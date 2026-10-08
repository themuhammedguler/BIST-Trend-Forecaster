# tests/test_model_metadata.py - Model meta-veri kayıt ve okuma birim testleri
import os
import subprocess
import sys

import pandas as pd
import pytest
import xgboost as xgb

import config
from src import model_metadata


def test_build_metadata_schema_and_types():
    meta = model_metadata.build_metadata(
        cutoff_date="2025-02-25",
        train_rows=50000,
        test_rows=5000,
        test_accuracy=0.551234,
        metrics={"roc_auc": 0.56789, "balanced_accuracy": 0.54321},
        features=["rsi", "macd"],
        best_params={"max_depth": 5},
        version="1.0",
    )

    assert meta["version"] == "1.0"
    assert meta["data_cutoff_date"] == "2025-02-25"
    assert meta["train_rows"] == 50000
    assert meta["test_rows"] == 5000
    assert meta["test_accuracy"] == 0.5512
    assert meta["metrics"]["roc_auc"] == 0.5679
    assert meta["features"] == ["rsi", "macd"]
    assert meta["best_params"] == {"max_depth": 5}
    assert isinstance(meta["trained_at"], str)


def test_save_and_load_metadata_roundtrip(tmp_path):
    target = str(tmp_path / "test_meta.json")
    original = model_metadata.build_metadata(
        cutoff_date="2025-01-01",
        train_rows=100,
        test_rows=10,
        test_accuracy=0.60,
    )

    saved_path = model_metadata.save_metadata(original, path=target)
    assert saved_path == target
    assert os.path.exists(target)

    loaded = model_metadata.load_metadata(path=target)
    assert loaded == original


def test_load_metadata_returns_none_when_missing(tmp_path):
    non_existent = str(tmp_path / "does_not_exist.json")
    assert model_metadata.load_metadata(path=non_existent) is None


def test_load_metadata_returns_none_on_corrupt_json(tmp_path):
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{ this is not valid json", encoding="utf-8")
    assert model_metadata.load_metadata(path=str(corrupt_file)) is None


def test_format_metadata_summary():
    assert "bulunamadı" in model_metadata.format_metadata_summary(None)

    meta = {
        "version": "1.0",
        "data_cutoff_date": "2025-02-25",
        "test_accuracy": 0.5512,
    }
    summary = model_metadata.format_metadata_summary(meta)
    assert "v1.0" in summary
    assert "2025-02-25" in summary
    assert "%55.1" in summary


@pytest.mark.skipif(not os.path.exists(config.MODEL_PATH), reason="model dosyası yok")
def test_shipped_model_metadata_is_valid():
    assert os.path.exists(config.MODEL_META_PATH)
    meta = model_metadata.load_metadata(config.MODEL_META_PATH)
    assert meta is not None
    assert "version" in meta
    assert "test_accuracy" in meta
    assert "features" in meta

    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    assert meta["features"] == model.get_booster().feature_names


def test_training_script_entry_point_imports_cleanly():
    """model_train.py, src/ içinden betik olarak çalıştırılır; 'src' paketi
    sys.path'te olmadığından modüller 'from src import ...' kullanmamalıdır."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, os.path.join("src", "model_train.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not (os.path.exists(config.MODEL_PATH) and os.path.exists(config.DATA_PATH)),
    reason="model dosyası veya veri seti yok",
)
def test_shipped_metadata_matches_measured_test_scores():
    """Arayüzde gösterilen skorlar, gönderilen modelin zamansal test kesitindeki
    gerçek ölçümleriyle aynı olmalıdır."""
    import features
    import metrics
    import model_train

    meta = model_metadata.load_metadata(config.MODEL_META_PATH)
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    processed = features.add_features(pd.read_csv(config.DATA_PATH))
    X_train, X_test, _, y_test, _, cutoff = model_train.get_temporal_split(
        processed, meta["features"]
    )
    scores = metrics.evaluate(y_test, model.predict_proba(X_test)[:, 1])

    assert meta["data_cutoff_date"] == pd.to_datetime(cutoff).strftime("%Y-%m-%d")
    assert (meta["train_rows"], meta["test_rows"]) == (len(X_train), len(X_test))
    assert meta["test_accuracy"] == round(scores["accuracy"], 4)
    for name, value in meta["metrics"].items():
        assert value == round(scores[name], 4), name
