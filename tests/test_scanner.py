# tests/test_scanner.py - scanner.py birim testleri
import pandas as pd
import numpy as np
import pytest
import xgboost as xgb
import yfinance

from src import scanner, config


@pytest.fixture
def dummy_model():
    """Test için basit bir XGBoost modeli simülasyonu."""
    class DummyModel:
        def predict_proba(self, X):
            # İlk özelliğe göre deterministik olasılık
            val = float(X.iloc[0, 0])
            prob = 1.0 / (1.0 + np.exp(-val))
            return np.array([[1.0 - prob, prob]])
    return DummyModel()


@pytest.fixture
def fake_panel():
    """120 iş günü OHLCV paneli."""
    dates = pd.bdate_range("2025-01-01", periods=120)
    close = 100.0 + np.arange(120) * 0.1
    df = pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": 1_000_000.0
    }, index=pd.Index(dates, name="Date"))
    return df


def test_predict_single_ticker_success(monkeypatch, dummy_model, fake_panel):
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: fake_panel.copy())

    res = scanner.predict_single_ticker("AKBNK.IS", dummy_model)
    assert res is not None
    assert res["Hisse"] == "AKBNK"
    assert res["ticker_code"] == "AKBNK.IS"
    assert "Son Fiyat (TL)" in res
    assert "Günlük Değişim (%)" in res
    assert "Yükseliş Olasılığı (%)" in res
    assert 0.0 <= res["_prob"] <= 1.0
    assert res["Tahmin"] in ["YUKARI 🚀", "DÜŞÜŞ 🔻"]


def test_predict_single_ticker_handles_empty_data(monkeypatch, dummy_model):
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: pd.DataFrame())

    res = scanner.predict_single_ticker("INVALID.IS", dummy_model)
    assert res is None


def test_scan_market_sorts_by_probability_descending(monkeypatch, dummy_model, fake_panel):
    # Farklı hisseler için farklı fiyatlar/indikatörler dönelim
    def custom_download(ticker, *args, **kwargs):
        df = fake_panel.copy()
        if "GARAN" in ticker:
            df["Close"] = df["Close"] * 2.0
        elif "THYAO" in ticker:
            df["Close"] = df["Close"] * 0.5
        return df

    monkeypatch.setattr(yfinance, "download", custom_download)

    tickers = ["AKBNK.IS", "GARAN.IS", "THYAO.IS"]
    df_scan = scanner.scan_market(tickers, dummy_model)

    assert len(df_scan) == 3
    # Azalan sırada sıralı olmalı
    probs = df_scan["_prob"].tolist()
    assert probs == sorted(probs, reverse=True)
    assert set(df_scan["ticker_code"]) == set(tickers)


def test_scan_market_handles_empty_list(dummy_model):
    df_scan = scanner.scan_market([], dummy_model)
    assert df_scan.empty
    assert "Hisse" in df_scan.columns
    assert "_prob" in df_scan.columns


def test_scan_market_continues_on_partial_failure(monkeypatch, dummy_model, fake_panel):
    def flappy_download(ticker, *args, **kwargs):
        if "FAIL" in ticker:
            return pd.DataFrame()
        return fake_panel.copy()

    monkeypatch.setattr(yfinance, "download", flappy_download)

    tickers = ["AKBNK.IS", "FAIL.IS", "THYAO.IS"]
    df_scan = scanner.scan_market(tickers, dummy_model)

    assert len(df_scan) == 2
    assert "FAIL.IS" not in df_scan["ticker_code"].values
