# tests/test_live_data.py - canlı veri hazırlama yardımcılarının birim testleri
import numpy as np
import pandas as pd
import pytest
import yfinance

from src import live_data


def _raw_panel(n_days=120, multiindex_ticker=None):
    """yf.download çıktısı biçiminde OHLCV; istenirse (Price, Ticker) MultiIndex sütunlu."""
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    close = 100.0 + np.arange(n_days) * 0.1
    df = pd.DataFrame(
        {
            "Close": close,
            "High": close + 1,
            "Low": close - 1,
            "Open": close - 0.5,
            "Volume": 1_000_000.0,
        },
        index=pd.Index(dates, name="Date"),
    )
    if multiindex_ticker:
        df.columns = pd.MultiIndex.from_product(
            [df.columns, [multiindex_ticker]], names=["Price", "Ticker"]
        )
    return df


def test_yahoo_symbol_maps_renamed_tickers():
    assert live_data.yahoo_symbol("KOZAL.IS") == "TRALT.IS"
    assert live_data.yahoo_symbol("THYAO.IS") == "THYAO.IS"


@pytest.mark.parametrize("multiindex_ticker", [None, "AKBNK.IS"])
def test_prepare_live_frame_normalises_columns_and_keeps_latest_day(multiindex_ticker):
    raw = _raw_panel(multiindex_ticker=multiindex_ticker)

    processed, ohlcv = live_data.prepare_live_frame(raw, "AKBNK.IS")

    assert {"Date", "ticker", "open", "high", "low", "close", "volume"} <= set(ohlcv.columns)
    assert (ohlcv["ticker"] == "AKBNK").all()
    # Canlı tahmin için en son işlem günü düşürülmemeli
    assert processed["Date"].iloc[-1] == raw.index[-1]
    assert "rsi" in processed.columns


@pytest.mark.parametrize("raw", [None, pd.DataFrame()])
def test_prepare_live_frame_rejects_missing_data(raw):
    with pytest.raises(ValueError, match="piyasa verisi alınamadı"):
        live_data.prepare_live_frame(raw, "AKBNK.IS")


def test_missing_data_message_suggests_retrying_later():
    """yfinance ağ kesintilerini boş veri olarak döndürür; mesaj bu durumu da kapsamalı."""
    with pytest.raises(ValueError, match="birkaç dakika sonra tekrar deneyin"):
        live_data.prepare_live_frame(pd.DataFrame(), "AKBNK.IS")


def test_prepare_live_frame_rejects_too_short_history():
    with pytest.raises(ValueError, match="yetersiz"):
        live_data.prepare_live_frame(_raw_panel(n_days=20), "AKBNK.IS")


def test_fetch_live_frame_downloads_mapped_symbol(monkeypatch):
    requested = []

    def fake_download(ticker, *args, **kwargs):
        requested.append(ticker)
        return _raw_panel()

    monkeypatch.setattr(yfinance, "download", fake_download)

    live_data.fetch_live_frame("KOZAA.IS")
    assert requested == ["TRMET.IS"]


def test_live_frame_has_six_months_of_indicators_for_backtest(monkeypatch):
    """Göstergelerin ısınma süresinden (sma_50) sonra en az 6 aylık satır kalmalı."""
    months = {"3mo": 3, "6mo": 6, "1y": 12, "2y": 24}
    end = pd.Timestamp("2026-10-06")

    def fake_download(ticker, period, **kwargs):
        dates = pd.bdate_range(end - pd.DateOffset(months=months[period]), end)
        close = 100.0 + np.arange(len(dates)) * 0.1
        return pd.DataFrame(
            {"Close": close, "High": close + 1, "Low": close - 1, "Open": close, "Volume": 1e6},
            index=pd.Index(dates, name="Date"),
        )

    monkeypatch.setattr(yfinance, "download", fake_download)

    df_processed, _ = live_data.fetch_live_frame("AKBNK.IS")
    assert df_processed["Date"].min() <= end - pd.DateOffset(months=6)


@pytest.mark.parametrize("multiindex_ticker", [None, "AKBNK.IS"])
def test_prepare_live_frame_drops_partial_rows_without_prices(multiindex_ticker):
    """Yahoo seans sonrası bazen günün satırını OHLC'si boş, yalnızca hacmi dolu
    döndürür. Bu satır fiyat/grafik ve tahmin için kullanılmamalı (#25)."""
    raw = _raw_panel()
    partial_day = raw.index[-1] + pd.offsets.BDay(1)
    raw.loc[partial_day] = [np.nan, np.nan, np.nan, np.nan, 125_773_102.0]
    if multiindex_ticker:
        raw.columns = pd.MultiIndex.from_product(
            [raw.columns, [multiindex_ticker]], names=["Price", "Ticker"]
        )

    processed, ohlcv = live_data.prepare_live_frame(raw, "AKBNK.IS")

    assert ohlcv[["open", "high", "low", "close"]].notna().all().all()
    assert ohlcv["Date"].iloc[-1] == raw.index[-2]
    # Fiyat metriği, grafik ve tahmin aynı son tam işlem gününü kullanmalı
    assert processed["Date"].iloc[-1] == ohlcv["Date"].iloc[-1]


def test_fetch_live_frame_recovers_from_transient_network_errors(monkeypatch):
    outcomes = [ConnectionError("ağ hatası"), ConnectionError("ağ hatası"), _raw_panel()]
    monkeypatch.setattr(
        yfinance, "download", lambda *args, **kwargs: _raise_or_return(outcomes.pop(0))
    )

    df_processed, _ = live_data.fetch_live_frame("AKBNK.IS")

    assert not df_processed.empty
    assert outcomes == []


def test_fetch_live_frame_reports_connection_problem_after_retries(monkeypatch):
    def download(*args, **kwargs):
        raise ConnectionError("ağ hatası")

    monkeypatch.setattr(yfinance, "download", download)

    with pytest.raises(ValueError, match="bağlantı sorunu"):
        live_data.fetch_live_frame("AKBNK.IS")


def _raise_or_return(outcome):
    if isinstance(outcome, Exception):
        raise outcome
    return outcome


def test_prepare_live_frame_with_macro_data():
    raw = _raw_panel(n_days=120)
    macro_tuples = [("XU100.IS", "Close"), ("USDTRY=X", "Close")]
    cols = pd.MultiIndex.from_tuples(macro_tuples, names=["Ticker", "Price"])
    macro_df = pd.DataFrame(
        [[1000.0, 30.0], [1020.0, 30.3]],
        index=raw.index[-2:],
        columns=cols,
    )

    processed, _ = live_data.prepare_live_frame(raw, "AKBNK.IS", macro_data=macro_df)

    assert "xu100_ret" in processed.columns
    assert "rel_strength_bist" in processed.columns
    assert "usdtry_change" in processed.columns


def test_fetch_macro_frame_downloads_macro_tickers(monkeypatch):
    called_tickers = []

    def fake_download(tickers, *args, **kwargs):
        called_tickers.append(tickers)
        return _raw_panel()

    monkeypatch.setattr(yfinance, "download", fake_download)

    live_data.fetch_macro_frame()
    assert len(called_tickers) == 1
    assert set(called_tickers[0]) == {"XU100.IS", "USDTRY=X"}
