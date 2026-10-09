# tests/test_integration.py - gerçek veri seti ile uçtan uca eğitim/optimizasyon
import os

import pytest
import xgboost as xgb

import config
import features
import model_metadata
import model_train
import tune

FEATURES = config.FEATURES

pytestmark = pytest.mark.skipif(not os.path.exists(config.DATA_PATH), reason="veri seti yok")


def test_features_and_study_on_real_data(small_dataset):
    processed = features.add_features(small_dataset)
    study = tune.run_study(
        processed[FEATURES], processed["target"], processed["Date"], n_trials=2, n_splits=2
    )
    assert 0.0 <= study.best_value <= 1.0


@pytest.mark.parametrize("use_tune", [False, True])
def test_train_model_saves_loadable_model(small_dataset, use_tune):
    model, acc = model_train.train_model(tune=use_tune, n_trials=2)
    assert 0.0 <= acc <= 1.0
    assert os.path.exists(config.MODEL_PATH)

    # app.py ile aynı şekilde yükle ve tahmin yap
    loaded = xgb.XGBClassifier()
    loaded.load_model(config.MODEL_PATH)
    X = features.add_features(small_dataset)[FEATURES].tail(5)
    proba = loaded.predict_proba(X)
    assert proba.shape == (5, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_shipped_model_still_loads():
    """Repodaki modelin eğitim kodundaki değişiklikten etkilenmediğini doğrular."""
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    assert model.n_features_in_ == len(FEATURES)


def test_train_model_reports_imbalance_aware_metrics(small_dataset, capsys):
    model_train.train_model()
    out = capsys.readouterr().out
    assert "Eğitim Sınıf Dağılımı" in out
    assert "Test Sınıf Dağılımı" in out
    for label in ("Dengeli Doğruluk", "ROC-AUC", "Log Loss"):
        assert label in out


def test_train_model_tunes_for_selected_metric(small_dataset, capsys):
    model_train.train_model(tune=True, n_trials=2, metric="roc_auc")
    assert "En iyi CV skoru (roc_auc)" in capsys.readouterr().out


@pytest.mark.parametrize("use_tune", [False, True])
def test_train_model_applies_class_weight(small_dataset, use_tune, capsys):
    model, _ = model_train.train_model(tune=use_tune, n_trials=2, balance_classes=True)
    weight = model.get_params()["scale_pos_weight"]
    assert weight is not None and weight > 0
    assert "scale_pos_weight=" in capsys.readouterr().out


def test_train_model_writes_metadata_next_to_model(small_dataset, tmp_path):
    model, acc = model_train.train_model()

    meta = model_metadata.load_metadata(str(tmp_path / "models" / "model_meta.json"))
    assert meta is not None
    assert meta["test_accuracy"] == round(acc, 4)
    assert meta["features"] == model.get_booster().feature_names


def test_untuned_training_records_default_params_in_metadata(small_dataset):
    model_train.train_model(tune=False)

    meta = model_metadata.load_metadata(config.MODEL_META_PATH)
    assert meta["best_params"] == tune.DEFAULT_PARAMS
