# tests/test_ticker_alias.py - sembol eşleme ve boş veri doğrulama testleri
import pandas as pd
import pytest
import yfinance

import app
import config


def test_ticker_yahoo_map_contains_renamed_tickers():
    """Kasım 2025'te değişen KOZAL (TRALT) ve KOZAA (TRMET) sembollerinin
    haritada olduğunu doğrular."""
    assert hasattr(config, "TICKER_YAHOO_MAP")
    assert config.TICKER_YAHOO_MAP.get("KOZAL.IS") == "TRALT.IS"
    assert config.TICKER_YAHOO_MAP.get("KOZAA.IS") == "TRMET.IS"


def test_get_prediction_data_uses_mapped_ticker(monkeypatch):
    """get_prediction_data fonksiyonunun Yahoo Finance'e sorgu atarken haritadaki
    güncel sembolü kullandığını doğrular."""
    requested_tickers = []

    def fake_download(ticker, *args, **kwargs):
        requested_tickers.append(ticker)
        # 120 günlük sahte veri dön
        dates = pd.bdate_range("2025-01-01", periods=120)
        return pd.DataFrame(
            {"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5, "Volume": 1000.0}, index=dates
        )

    monkeypatch.setattr(yfinance, "download", fake_download)

    app.get_prediction_data("KOZAL.IS")
    assert requested_tickers[-1] == "TRALT.IS"

    app.get_prediction_data("KOZAA.IS")
    assert requested_tickers[-1] == "TRMET.IS"

    app.get_prediction_data("THYAO.IS")
    assert requested_tickers[-1] == "THYAO.IS"


def test_get_prediction_data_raises_value_error_on_empty_df(monkeypatch):
    """Yahoo Finance boş veri döndüğünde KeyError: 'Date' yerine anlamlı ValueError fırlatılmalı."""
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(ValueError, match="piyasa verisi alınamadı"):
        app.get_prediction_data("INVALID.IS")
