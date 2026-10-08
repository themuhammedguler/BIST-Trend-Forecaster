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
