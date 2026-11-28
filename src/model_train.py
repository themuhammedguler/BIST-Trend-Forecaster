# src/model_train.py
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report

try:
    import config  # type: ignore
    import ensemble  # type: ignore
    import features  # type: ignore
    import metrics  # type: ignore
    import model_metadata  # type: ignore
    import tune as tune_module  # type: ignore
except ImportError:
    from src import config, ensemble, features, metrics, model_metadata
    from src import tune as tune_module


def get_temporal_split(df_processed, features_list, train_ratio=0.9):
    """
    Veriyi benzersiz tarihlere göre (walk-forward mantığıyla) eğitim ve test setlerine ayırır.

    Tüm hisselerin belirli bir tarihten önceki verilerini eğitim, o tarih ve sonrasındaki
    verilerini ise test seti yapar. Böylece 'look-ahead bias' ve hisse bazlı veri sızıntısı
    (data leakage) engellenir.
    """
    unique_dates = np.sort(df_processed["Date"].unique())
    if len(unique_dates) < 2:
        raise ValueError(
            "Zaman serisi ayrımı (temporal split) için en az 2 farklı tarih gereklidir."
        )

    cutoff_index = max(1, min(int(len(unique_dates) * train_ratio), len(unique_dates) - 1))
    cutoff_date = unique_dates[cutoff_index]

    train_mask = df_processed["Date"] < cutoff_date
    test_mask = df_processed["Date"] >= cutoff_date

    X_train = df_processed.loc[train_mask, features_list].reset_index(drop=True)
    X_test = df_processed.loc[test_mask, features_list].reset_index(drop=True)
    y_train = df_processed.loc[train_mask, "target"].reset_index(drop=True)
    y_test = df_processed.loc[test_mask, "target"].reset_index(drop=True)
    dates_train = df_processed.loc[train_mask, "Date"].reset_index(drop=True)

    return X_train, X_test, y_train, y_test, dates_train, cutoff_date


def train_horizon_models(
    df_processed: pd.DataFrame,
    models_dir: str,
    params: dict | None = None,
    balance_classes: bool = False,
) -> dict[str, float]:
    """5 Günlük (1 Hafta) ve 20 Günlük (1 Ay) çoklu vade XGBoost modellerini eğitir ve kaydeder."""
    os.makedirs(models_dir, exist_ok=True)
    xgb_params = dict(params if params is not None else tune_module.DEFAULT_PARAMS)
    horizon_Map = {
        "5d": ("xgb_5d_model.json", 5),
        "20d": ("xgb_20d_model.json", 20),
    }
    accuracies: dict[str, float] = {}

    df_sorted = df_processed.sort_values(by=["ticker", "Date"]).copy()
    for h_key, (filename, days) in horizon_Map.items():
        future_close = df_sorted.groupby("ticker")["close"].shift(-days)
        valid_mask = future_close.notna()
        df_h = df_sorted.loc[valid_mask].copy()
        df_h["target"] = (future_close.loc[valid_mask] > df_h["close"]).astype(int)

        X_tr, X_te, y_tr, y_te, _, _ = get_temporal_split(df_h, config.FEATURES, train_ratio=0.9)
        w_params = tune_module.class_weight_params(y_tr, balance_classes)
        h_model = xgb.XGBClassifier(**xgb_params, **tune_module.FIXED_PARAMS, **w_params)
        print(f"Çoklu vade ({h_key} / {days} iş günü) XGBoost modeli eğitiliyor...")
        h_model.fit(X_tr, y_tr)

        h_proba = h_model.predict_proba(X_te)[:, 1]
        h_acc = float(metrics.evaluate(y_te, h_proba)["accuracy"])
        accuracies[h_key] = h_acc

        out_path = os.path.join(models_dir, filename)
        if hasattr(h_model, "save_model"):
            h_model.save_model(out_path)
        elif hasattr(h_model, "booster_") and hasattr(h_model.booster_, "save_model"):
            h_model.booster_.save_model(out_path)
        print(f"✅ {h_key} modeli kaydedildi: {out_path} (Test Doğruluğu: {h_acc:.4f})")

    return accuracies


def train_model(
    tune=False,
    n_trials=50,
    metric="accuracy",
    balance_classes=False,
    min_accuracy=None,
    model_type="xgb",
    include_ensemble=False,
    include_horizons=False,
):
    # 1. Veriyi Yükle
    print("Veri yükleniyor...")
    df = pd.read_csv(config.DATA_PATH)

    # 2. Feature Engineering
    print("Feature Engineering uygulanıyor...")
    df_processed = features.add_features(df)

    # 3. Eğitim Seti Hazırlığı
    # Geleceği görmeyi engellemek için tarihsel kesim yapıyoruz (TimeSeries Split mantığı)
    X_train, X_test, y_train, y_test, dates_train, cutoff_date = get_temporal_split(
        df_processed, config.FEATURES, train_ratio=0.9
    )

    cutoff_str = pd.to_datetime(cutoff_date).strftime("%Y-%m-%d")
    train_start = pd.to_datetime(dates_train.min()).strftime("%Y-%m-%d")
    train_end = pd.to_datetime(dates_train.max()).strftime("%Y-%m-%d")
    test_dates = df_processed.loc[df_processed["Date"] >= cutoff_date, "Date"]
    test_start = pd.to_datetime(test_dates.min()).strftime("%Y-%m-%d")
    test_end = pd.to_datetime(test_dates.max()).strftime("%Y-%m-%d")

    print(f"Tarih Kesimi (Cutoff): {cutoff_str}")
    print(f"Eğitim Verisi: {X_train.shape} (Tarih Aralığı: {train_start} - {train_end})")
    print(f"Test Verisi: {X_test.shape} (Tarih Aralığı: {test_start} - {test_end})")
    for name, y_part in (("Eğitim", y_train), ("Test", y_test)):
        dist = metrics.class_distribution(y_part)
        print(f"{name} Sınıf Dağılımı: Yükseliş %{dist[1] * 100:.1f} / Düşüş %{dist[0] * 100:.1f}")

    # 4. Model Tanımlama ve Eğitim (XGBoost / LightGBM / CatBoost / Ensemble)
    if model_type == "ensemble":
        print("Çoklu model ve topluluk (XGBoost + LightGBM + CatBoost) eğitiliyor...")
        trained_models, df_comparison = ensemble.evaluate_models_comparison(
            X_train, y_train, X_test, y_test
        )
        print("\n📊 Çoklu Model Karşılaştırma Karnesi:")
        print(df_comparison.to_string(index=False))
        ensemble.save_ensemble_models(
            trained_models,
            os.path.dirname(config.MODEL_PATH),
            comparison_df=df_comparison,
        )
        model = trained_models["xgboost"]
    elif model_type == "lgbm":
        print("LightGBM modeli eğitiliyor...")
        model = ensemble.build_lgbm_classifier()
        model.fit(X_train, y_train)
    elif model_type == "catboost":
        print("CatBoost modeli eğitiliyor...")
        model = ensemble.build_catboost_classifier()
        model.fit(X_train, y_train)
    else:
        # Varsayılan: XGBoost
        params = dict(tune_module.DEFAULT_PARAMS)
        if tune:
            print(
                f"Optuna ile hiperparametre optimizasyonu ({n_trials} deneme, hedef: {metric})..."
            )
            study = tune_module.run_study(
                X_train,
                y_train,
                dates_train,
                n_trials=n_trials,
                metric=metric,
                balance_classes=balance_classes,
            )
            print(f"En iyi CV skoru ({metric}): {study.best_value:.4f}")
            print(f"En iyi parametreler: {study.best_params}")
            params = study.best_params

        weight_params = tune_module.class_weight_params(y_train, balance_classes)
        if weight_params:
            print(
                f"Sınıf ağırlıklandırma: scale_pos_weight={weight_params['scale_pos_weight']:.4f}"
            )
        model = xgb.XGBClassifier(**params, **tune_module.FIXED_PARAMS, **weight_params)

        print("XGBoost modeli eğitiliyor...")
        model.fit(X_train, y_train)

    # 5. Değerlendirme
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    scores = metrics.evaluate(y_test, proba)
    acc = scores["accuracy"]
    print(f"\n🎯 Model Doğruluğu (Test Seti): {acc:.4f}")
    print(f"Dengeli Doğruluk (Balanced Accuracy): {scores['balanced_accuracy']:.4f}")
    print(f"ROC-AUC: {scores['roc_auc']:.4f}")
    print(f"Log Loss: {scores['log_loss']:.4f}")
    print(f"Kesinlik (Precision, Yükseliş): {scores['precision']:.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, preds))

    # Feature Importance (PDF Maddesi: Model Evaluation)
    importance = dict(zip(config.FEATURES, model.feature_importances_))
    print("\nÖnem Düzeyleri:")
    for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v:.4f}")

    # Kalite Kapısı: Minimum doğruluk eşiği kontrolü (#20)
    if min_accuracy is not None and acc < min_accuracy:
        raise ValueError(
            f"Model doğruluğu ({acc:.4f}) belirlenen minimum kalite eşiğinin "
            f"({min_accuracy:.4f}) altında kaldı. Model kaydedilmedi."
        )

    # 6. Modeli ve Meta-verisini Kaydet
    if not os.path.exists(os.path.dirname(config.MODEL_PATH)):
        os.makedirs(os.path.dirname(config.MODEL_PATH))

    if hasattr(model, "save_model"):
        model.save_model(config.MODEL_PATH)
    elif hasattr(model, "booster_") and hasattr(model.booster_, "save_model"):
        model.booster_.save_model(config.MODEL_PATH)

    print(f"\n✅ Model kaydedildi: {config.MODEL_PATH}")

    model_params = params if "params" in locals() else {"model_type": model_type}
    meta = model_metadata.build_metadata(
        cutoff_date=cutoff_str,
        train_rows=len(X_train),
        test_rows=len(X_test),
        test_accuracy=acc,
        metrics=scores,
        features=config.FEATURES,
        best_params=model_params,
        version="1.0",
    )
    model_metadata.save_metadata(meta, config.MODEL_META_PATH)
    print(f"✅ Model meta-verisi kaydedildi: {config.MODEL_META_PATH}")

    models_dir = os.path.dirname(config.MODEL_PATH)

    # 7. İsteğe Bağlı: Topluluk (LightGBM + CatBoost + Ensemble Karşılaştırma Özeti) Eğitimi
    if include_ensemble and model_type != "ensemble":
        print("\nÇoklu model topluluğu (LightGBM + CatBoost + Ensemble) güncelleniyor...")
        trained_models, df_comparison = ensemble.evaluate_models_comparison(
            X_train, y_train, X_test, y_test
        )
        print("\n📊 Çoklu Model Karşılaştırma Karnesi:")
        print(df_comparison.to_string(index=False))
        # Ana XGBoost modelini ezmeden LightGBM, CatBoost ve ensemble_meta.json kaydet
        remaining_models = {k: v for k, v in trained_models.items() if k != "xgboost"}
        ensemble.save_ensemble_models(
            remaining_models,
            models_dir,
            comparison_df=df_comparison,
        )

    # 8. İsteğe Bağlı: Çoklu Vade (5G ve 20G) Modellerinin Eğitimi
    if include_horizons:
        print("\nÇoklu vade (5G ve 20G) modelleri güncelleniyor...")
        active_params = params if "params" in locals() else None
        train_horizon_models(
            df_processed,
            models_dir=models_dir,
            params=active_params,
            balance_classes=balance_classes,
        )

    return model, acc


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="BIST Makine Öğrenmesi Model Eğitimi")
    parser.add_argument(
        "--model",
        choices=["xgb", "lgbm", "catboost", "ensemble"],
        default="xgb",
        help="Eğitilecek model mimarisi (xgb, lgbm, catboost, ensemble)",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Optuna ile hiperparametre optimizasyonu yap"
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna deneme sayısı")
    parser.add_argument(
        "--metric",
        choices=tune_module.METRICS,
        default="accuracy",
        help="Optuna'nın maksimize edeceği CV metriği",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Sınıf oranına göre scale_pos_weight ile ağırlıklandır",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Modelin kaydedilmesi için gereken minimum test doğruluğu eşiği (örn: 0.50)",
    )
    parser.add_argument(
        "--include-ensemble",
        action="store_true",
        help="Ana modele ek olarak LightGBM, CatBoost ve ensemble_meta.json dosyalarını da eğit",
    )
    parser.add_argument(
        "--include-horizons",
        action="store_true",
        help="1 günlük modele ek olarak 5 günlük (1H) ve 20 günlük (1A) modelleri de eğit",
    )
    args = parser.parse_args()
    train_model(
        tune=args.tune,
        n_trials=args.n_trials,
        metric=args.metric,
        balance_classes=args.balance_classes,
        min_accuracy=args.min_accuracy,
        model_type=args.model,
        include_ensemble=args.include_ensemble,
        include_horizons=args.include_horizons,
    )
