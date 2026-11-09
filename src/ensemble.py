"""Çoklu Model ve Topluluk (Ensemble) Tahmin Motoru.

XGBoost, LightGBM ve CatBoost algoritmalarını birleştirerek
finansal zaman serilerinde varyansı düşürür ve daha dengeli
yön tahmin olasılıkları üretir.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


def build_xgb_classifier(params: dict[str, Any] | None = None) -> Any:
    """XGBClassifier örneği oluşturur."""
    import xgboost as xgb

    default_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
    }
    if params:
        default_params.update(params)
    return xgb.XGBClassifier(**default_params)


def build_lgbm_classifier(params: dict[str, Any] | None = None) -> Any:
    """LightGBM LGBMClassifier örneği oluşturur."""
    import lightgbm as lgb

    default_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "num_leaves": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    return lgb.LGBMClassifier(**default_params)


def build_catboost_classifier(params: dict[str, Any] | None = None) -> Any:
    """CatBoostClassifier örneği oluşturur."""
    from catboost import CatBoostClassifier

    default_params = {
        "iterations": 100,
        "depth": 3,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": 0,
    }
    if params:
        default_params.update(params)
    return CatBoostClassifier(**default_params)


class EnsemblePredictor:
    """XGBoost, LightGBM ve CatBoost modellerini ağırlıklandıran topluluk sınıfı."""

    def __init__(
        self,
        models: dict[str, Any],
        weights: dict[str, float] | None = None,
    ):
        self.models = models
        if weights is None:
            # Varsayılan eşit ağırlıklar
            n = max(1, len(models))
            self.weights = {name: 1.0 / n for name in models}
        else:
            # Ağırlıkları normalize et
            total = sum(weights.values())
            self.weights = (
                {k: v / total for k, v in weights.items()} if total > 0 else weights.copy()
            )

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Her modelin sınıf olasılıklarını ağırlıklı ortalama ile birleştirir."""
        combined_prob_1 = np.zeros(len(X), dtype=float)

        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w <= 0:
                continue
            probs = model.predict_proba(X)
            # 1. sınıf (Yükseliş) olasılığı
            p1 = probs[:, 1] if probs.ndim == 2 else probs
            combined_prob_1 += w * p1

        combined_prob_0 = 1.0 - combined_prob_1
        return np.column_stack([combined_prob_0, combined_prob_1])

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Kompozit olasılığa göre ikili yön tahminini (0 veya 1) döndürür."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def evaluate_models_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    weights: dict[str, float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """XGBoost, LightGBM ve CatBoost modellerini eğitip karşılaştırma tablosu üretir."""
    models: dict[str, Any] = {
        "XGBoost": build_xgb_classifier(),
        "LightGBM": build_lgbm_classifier(),
        "CatBoost": build_catboost_classifier(),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        logger.info("%s modeli eğitiliyor...", name)
        model.fit(X_train, y_train)
        trained_models[name.lower()] = model

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        bal_acc = balanced_accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, proba)
        loss = log_loss(y_test, proba)

        results.append(
            {
                "Model": name,
                "Doğruluk (Accuracy)": round(acc, 4),
                "Dengeli Doğruluk": round(bal_acc, 4),
                "ROC-AUC": round(roc, 4),
                "Log Loss": round(loss, 4),
            }
        )

    # Topluluk Modeli
    ensemble_weights = weights or {"xgboost": 0.40, "lightgbm": 0.30, "catboost": 0.30}
    ensemble = EnsemblePredictor(trained_models, ensemble_weights)
    ens_proba = ensemble.predict_proba(X_test)[:, 1]
    ens_preds = (ens_proba >= 0.5).astype(int)

    ens_acc = accuracy_score(y_test, ens_preds)
    ens_bal_acc = balanced_accuracy_score(y_test, ens_preds)
    ens_roc = roc_auc_score(y_test, ens_proba)
    ens_loss = log_loss(y_test, ens_proba)

    results.append(
        {
            "Model": "Topluluk (Ensemble)",
            "Doğruluk (Accuracy)": round(ens_acc, 4),
            "Dengeli Doğruluk": round(ens_bal_acc, 4),
            "ROC-AUC": round(ens_roc, 4),
            "Log Loss": round(ens_loss, 4),
        }
    )

    df_comparison = pd.DataFrame(results)
    return trained_models, df_comparison
