# tests/conftest.py
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Proje kök dizini ve src/ dizinini sys.path'e ekle
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import optuna  # noqa: E402
import streamlit as st  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    """@st.cache_data süreç genelinde kalıcıdır; testler arasında sızmasını önler."""
    st.cache_data.clear()
    yield


@pytest.fixture
def synthetic_data():
    """3 hisse x 120 gün; hedef ilk özelliğe bağlı, yani öğrenilebilir."""
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.date_range("2024-01-01", periods=120, freq="B"), 3)
    X = pd.DataFrame(rng.normal(size=(len(dates), 4)), columns=["f0", "f1", "f2", "f3"])
    y = pd.Series((X["f0"] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int))
    return X, y, pd.Series(dates)


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    """Yahoo isteklerindeki üstel geri çekilme beklemelerini testlerde atlar.
    src/ altındaki betikler modülü 'network' olarak, uygulama 'src.network'
    olarak yükler; ikisi ayrı modül nesneleridir."""
    import network
    from src import network as src_network

    for module in (network, src_network):
        monkeypatch.setattr(module, "_sleep", lambda seconds: None)


@pytest.fixture(autouse=True)
def isolate_audit_log(tmp_path, monkeypatch):
    """Testlerin gerçek data/prediction_audit.json dosyasını kirletmesini önler."""
    import tracker
    from src import tracker as src_tracker

    isolated_path = str(tmp_path / "prediction_audit.json")
    for module in (tracker, src_tracker):
        monkeypatch.setattr(module, "DEFAULT_AUDIT_LOG_PATH", isolated_path)


@pytest.fixture
def small_dataset(tmp_path, monkeypatch):
    """Gerçek CSV'den 3 hisse x ~1 yıllık küçük bir kesit; config yolları tmp'ye yönlendirilir."""
    import config
    from src import config as src_config

    if not os.path.exists(config.DATA_PATH):
        pytest.skip("veri seti yok")

    df = pd.read_csv(config.DATA_PATH)
    macro_tickers = {"XU100", "USDTRY", "XU100.IS", "USDTRY=X"}
    df = df[
        (df["ticker"].isin(["AKBNK", "GARAN", "THYAO"]) | df["ticker"].isin(macro_tickers))
        & (df["Date"] >= "2024-01-01")
    ]
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(models_dir / "model.json")
    model_meta_path = str(models_dir / "model_meta.json")

    for cfg in (config, src_config):
        monkeypatch.setattr(cfg, "DATA_PATH", str(data_path))
        monkeypatch.setattr(cfg, "MODEL_PATH", model_path)
        monkeypatch.setattr(cfg, "MODEL_META_PATH", model_meta_path)
    return df
