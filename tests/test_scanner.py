# tests/test_scanner.py - scanner.py birim testleri
import numpy as np
import pandas as pd
import pytest
import yfinance

from src import scanner


@pytest.fixture
def dummy_model():
    """Test için basit bir XGBoost modeli simülasyonu."""

    class DummyModel:
        def predict_proba(self, X):
            # RSI'a göre deterministik olasılık (RSI 0-100 -> olasılık 0-1)
            prob = float(X["rsi"].iloc[0]) / 100.0
            return np.array([[1.0 - prob, prob]])

    return DummyModel()


@pytest.fixture
def fake_panel():
    """120 iş günü OHLCV paneli."""
    dates = pd.bdate_range("2025-01-01", periods=120)
    close = 100.0 + np.arange(120) * 0.1
    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=pd.Index(dates, name="Date"),
    )
    return df


def batch_download(make_frame, calls=None):
    """yf.download(liste, group_by="ticker") çıktısını taklit eder: (sembol, alan)
    MultiIndex sütunlu tek bir DataFrame. Verisi olmayan semboller, gerçek
    Yahoo yanıtında olduğu gibi tamamen NaN bir blok olarak döner."""

    def download(tickers, *args, **kwargs):
        if calls is not None:
            calls.append(tickers)
        symbols = [tickers] if isinstance(tickers, str) else list(tickers)
        frames = {sym: make_frame(sym) for sym in symbols}
        index = next((f.index for f in frames.values() if not f.empty), None)
        if index is None:
            return pd.DataFrame()
        cols = ["Open", "High", "Low", "Close", "Volume"]
        frames = {
            sym: (f if not f.empty else pd.DataFrame(np.nan, index=index, columns=cols))
            for sym, f in frames.items()
        }
        return pd.concat(frames, axis=1)

    return download


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
    # Farklı fiyat şekilleri farklı RSI, dolayısıyla farklı olasılık üretir.
    # (Fiyatı sabit bir katsayıyla ölçeklemek RSI'ı değiştirmez.)
    def make_frame(symbol):
        df = fake_panel.copy()
        if "GARAN" in symbol:  # sürekli düşüş -> RSI ~ 0
            df["Close"] = df["Close"].to_numpy()[::-1]
        elif "THYAO" in symbol:  # yatay dalgalanma -> RSI ~ 50
            df["Close"] = 100.0 + np.sin(np.arange(len(df)))
        return df

    monkeypatch.setattr(yfinance, "download", batch_download(make_frame))

    tickers = ["AKBNK.IS", "GARAN.IS", "THYAO.IS"]
    df_scan, failed = scanner.scan_market(tickers, dummy_model)

    assert len(df_scan) == 3
    # Olasılıklar birbirinden farklı olmalı ki sıralama gerçekten sınansın
    assert df_scan["_prob"].nunique() == 3
    assert df_scan["ticker_code"].tolist() == ["AKBNK.IS", "THYAO.IS", "GARAN.IS"]


def test_scan_market_handles_empty_list(dummy_model):
    df_scan, failed = scanner.scan_market([], dummy_model)
    assert df_scan.empty
    assert "Hisse" in df_scan.columns
    assert "_prob" in df_scan.columns


def test_scan_market_continues_on_partial_failure(monkeypatch, dummy_model, fake_panel):
    monkeypatch.setattr(
        yfinance,
        "download",
        batch_download(lambda symbol: pd.DataFrame() if "FAIL" in symbol else fake_panel.copy()),
    )

    tickers = ["AKBNK.IS", "FAIL.IS", "THYAO.IS"]
    df_scan, failed = scanner.scan_market(tickers, dummy_model)

    assert len(df_scan) == 2
    assert "FAIL.IS" not in df_scan["ticker_code"].values
    assert failed == ["FAIL.IS"]


def test_scan_market_downloads_all_symbols_in_one_request(monkeypatch, dummy_model, fake_panel):
    calls = []
    monkeypatch.setattr(
        yfinance, "download", batch_download(lambda symbol: fake_panel.copy(), calls)
    )

    df_scan, failed = scanner.scan_market(["AKBNK.IS", "KOZAL.IS", "THYAO.IS"], dummy_model)

    assert len(calls) == 1
    # Yahoo'da adı değişen hisse güncel sembolüyle istenmeli
    assert sorted(calls[0]) == ["AKBNK.IS", "THYAO.IS", "TRALT.IS"]
    assert set(df_scan["ticker_code"]) == {"AKBNK.IS", "KOZAL.IS", "THYAO.IS"}


def test_scan_market_reports_no_failures_on_success(monkeypatch, dummy_model, fake_panel):
    monkeypatch.setattr(yfinance, "download", batch_download(lambda symbol: fake_panel.copy()))

    _, failed = scanner.scan_market(["AKBNK.IS", "THYAO.IS"], dummy_model)

    assert failed == []


@pytest.mark.parametrize("error", [KeyError, ValueError])
def test_scan_market_does_not_hide_model_errors(monkeypatch, fake_panel, error):
    """Model hataları "veri alınamadı" gibi görünmemeli; çağırana ulaşmalı.
    XGBoost öznitelik uyumsuzluğunda ValueError fırlattığı için o da kapsanır."""

    class BrokenModel:
        def predict_proba(self, X):
            raise error("feature mismatch")

    monkeypatch.setattr(yfinance, "download", batch_download(lambda symbol: fake_panel.copy()))

    with pytest.raises(error):
        scanner.scan_market(["AKBNK.IS"], BrokenModel())


def _scan_frame(n):
    probs = np.linspace(0.9, 0.1, n)
    return pd.DataFrame({"ticker_code": [f"T{i}.IS" for i in range(n)], "_prob": probs})


@pytest.mark.parametrize(
    "n_rows, expected_size", [(30, 5), (10, 5), (9, 4), (3, 1), (1, 0), (0, 0)]
)
def test_top_and_bottom_never_overlap(n_rows, expected_size):
    bull, bear = scanner.top_and_bottom(_scan_frame(n_rows), n=5)

    assert len(bull) == len(bear) == expected_size
    assert not set(bull["ticker_code"]) & set(bear["ticker_code"])


def test_top_and_bottom_order():
    bull, bear = scanner.top_and_bottom(_scan_frame(30), n=5)

    assert bull["ticker_code"].tolist() == ["T0.IS", "T1.IS", "T2.IS", "T3.IS", "T4.IS"]
    # Düşüş listesi en düşük olasılıktan başlamalı
    assert bear["ticker_code"].tolist() == ["T29.IS", "T28.IS", "T27.IS", "T26.IS", "T25.IS"]


def test_scan_market_retries_batch_after_transient_error(monkeypatch, dummy_model, fake_panel):
    tickers = ["AKBNK.IS", "GARAN.IS"]
    healthy = batch_download(lambda sym: fake_panel.copy())
    attempts = []

    def flaky(tickers_arg, *args, **kwargs):
        attempts.append(tickers_arg)
        if len(attempts) == 1:
            raise ConnectionError("ağ hatası")
        return healthy(tickers_arg, *args, **kwargs)

    monkeypatch.setattr(yfinance, "download", flaky)

    df_scan, failed = scanner.scan_market(tickers, dummy_model)

    assert len(attempts) == 2
    assert failed == []
    assert len(df_scan) == 2


def test_scan_market_marks_all_failed_when_yahoo_unreachable(monkeypatch, dummy_model):
    def download(*args, **kwargs):
        raise ConnectionError("ağ hatası")

    monkeypatch.setattr(yfinance, "download", download)

    df_scan, failed = scanner.scan_market(["AKBNK.IS", "GARAN.IS"], dummy_model)

    assert df_scan.empty
    assert failed == ["AKBNK.IS", "GARAN.IS"]


def test_scan_market_with_macro_includes_index_and_fx(monkeypatch, dummy_model, fake_panel):
    calls = []
    monkeypatch.setattr(
        yfinance, "download", batch_download(lambda symbol: fake_panel.copy(), calls)
    )

    df_scan, failed = scanner.scan_market(["AKBNK.IS", "GARAN.IS"], dummy_model, include_macro=True)

    assert len(calls) == 1
    assert set(calls[0]) == {"AKBNK.IS", "GARAN.IS", "XU100.IS", "USDTRY=X"}
    assert len(df_scan) == 2
