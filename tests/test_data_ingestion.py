# tests/test_data_ingestion.py - toplu veri çekme (data_ingestion.py) testleri
import pandas as pd
import pytest
import yfinance

import config
import data_ingestion


@pytest.fixture
def two_tickers(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TICKERS", ["AKBNK.IS", "GARAN.IS"])
    monkeypatch.setattr(config, "MACRO_TICKERS", [])
    monkeypatch.setattr(config, "DATA_PATH", str(tmp_path / "data" / "bist.csv"))


def _frame(*args, **kwargs):
    dates = pd.bdate_range("2025-01-01", periods=3)
    return pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100.0},
        index=pd.Index(dates, name="Date"),
    )


def test_fetch_data_recovers_from_transient_errors(two_tickers, monkeypatch):
    failures = {"AKBNK.IS": 2}

    def download(ticker, *args, **kwargs):
        if failures.get(ticker, 0) > 0:
            failures[ticker] -= 1
            raise ConnectionError("ağ hatası")
        return _frame()

    monkeypatch.setattr(yfinance, "download", download)

    data_ingestion.fetch_data()

    saved = pd.read_csv(config.DATA_PATH)
    assert sorted(saved["ticker"].unique()) == ["AKBNK", "GARAN"]
    assert len(saved) == 6


def test_fetch_data_skips_ticker_that_stays_unreachable(two_tickers, monkeypatch, capsys):
    def download(ticker, *args, **kwargs):
        if ticker == "AKBNK.IS":
            raise ConnectionError("ağ hatası")
        return _frame()

    monkeypatch.setattr(yfinance, "download", download)

    data_ingestion.fetch_data()

    saved = pd.read_csv(config.DATA_PATH)
    assert list(saved["ticker"].unique()) == ["GARAN"]
    assert "AKBNK.IS hatası" in capsys.readouterr().out


def test_fetch_data_downloads_macro_tickers(two_tickers, monkeypatch):
    monkeypatch.setattr(config, "MACRO_TICKERS", ["XU100.IS", "USDTRY=X"])
    monkeypatch.setattr(yfinance, "download", lambda ticker, *args, **kwargs: _frame())

    data_ingestion.fetch_data(include_macro=True)

    saved = pd.read_csv(config.DATA_PATH)
    assert set(saved["ticker"].unique()) == {"AKBNK", "GARAN", "XU100", "USDTRY"}
    assert len(saved) == 12
