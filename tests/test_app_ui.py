# tests/test_app_ui.py - app.py'yi streamlit.testing.v1.AppTest ile uçtan uca test eder
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost
import yfinance
from streamlit.testing.v1 import AppTest

import config
import explain
import features
from src import config as src_config

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

DISPLAYED_COLUMNS = ["rsi", "macd", "sma_10", "sma_50", "volatility"]


def _disable_macro(monkeypatch):
    for cfg in (config, src_config):
        monkeypatch.setattr(cfg, "MACRO_TICKERS", [])


def _fake_download_panel(n_days=130, ticker="AKBNK.IS"):
    """yf.download(period='6mo')'un gerçek şeklini (MultiIndex sütunlar, 'Date'
    index'i) taklit eden deterministik bir OHLCV paneli üretir. Fiyat serisine
    SON günde belirgin bir sıçrama eklenir ki 'bir gün önceki' veri ile
    'bugünkü' veri UI testinde kolayca ayırt edilebilsin."""
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    close = 100.0 + np.arange(n_days) * 0.05
    close[-1] += 15.0

    panel = pd.DataFrame(
        {
            "Close": close,
            "High": close + 1,
            "Low": close - 1,
            "Open": close - 0.5,
            "Volume": np.full(n_days, 1_000_000.0),
        },
        index=pd.Index(dates, name="Date"),
    )
    panel.columns = pd.MultiIndex.from_product([panel.columns, [ticker]], names=["Price", "Ticker"])
    return panel


def _expected_last_row(panel, ticker="AKBNK.IS"):
    """app.py'nin get_prediction_data() içindeki yeniden adlandırma adımlarını
    bire bir uygular, ardından CANLI TAHMİN için doğru davranışı (son günün
    düşürülmemesi) kullanarak beklenen son satırı üretir."""
    df = panel.copy()
    df.columns = df.columns.get_level_values(0)
    df["ticker"] = ticker.replace(".IS", "")
    df.reset_index(inplace=True)
    df.rename(
        columns={
            c: ("Date" if c.lower() == "date" else c if c == "ticker" else c.lower())
            for c in df.columns
        },
        inplace=True,
    )
    processed = features.add_features(df, drop_incomplete_target=False)
    return processed.iloc[[-1]]


@pytest.fixture
def patched_download(monkeypatch):
    """Tek sembol istekleri için `panel`, BIST 30 taramasının toplu isteği için
    (sembol, alan) sütunlu bir toplu panel döndürür."""
    _disable_macro(monkeypatch)
    panel = _fake_download_panel()

    def download(tickers, *args, **kwargs):
        if isinstance(tickers, str):
            return panel.copy()
        single = panel.copy()
        single.columns = single.columns.get_level_values(0)
        return pd.concat({sym: single for sym in tickers}, axis=1)

    monkeypatch.setattr(yfinance, "download", download)
    return panel


def test_app_renders_a_single_prediction_without_error(patched_download):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert list(at.exception) == []
    assert not any("Bir hata oluştu" in e.value for e in at.error)

    # Yön kutusu ve güven skoru, sinyalin rengini taşıyan iki kutu olarak render edilmeli.
    assert len(at.success) + len(at.warning) + len(at.error) == 2
    assert len(at.metric) == 1


def test_app_uses_most_recent_trading_day_for_prediction(patched_download):
    """Canlı tahminde kullanılan son gün göstergeleri, en güncel (bugünkü) güne
    ait olmalı - bir önceki güne değil."""
    expected = _expected_last_row(patched_download)[DISPLAYED_COLUMNS]

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    displayed = at.dataframe[0].value[DISPLAYED_COLUMNS]

    pd.testing.assert_frame_equal(
        displayed.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_exact=False,
    )


def _plotly_charts(at):
    return at.get("plotly_chart")


def _chart_traces(at):
    """Her Plotly grafiğinin iz (trace) listesi."""
    return [json.loads(chart.proto.spec)["data"] for chart in _plotly_charts(at)]


def test_explanation_replaces_static_rsi_rules(patched_download):
    """Fikstürde RSI 100'dür; eski kural tabanlı metin "aşırı alım" basardı."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    texts = [m.value for m in at.markdown]
    assert not any("aşırı alım" in t or "aşırı satım" in t for t in texts)
    assert any("tahminini" in t for t in texts)


def test_explanation_matches_model_contributions(patched_download):
    """UI'daki özet, aynı satır için modelin gerçek TreeSHAP katkılarıyla tutarlı olmalı."""
    model = xgboost.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    row = _expected_last_row(patched_download)[model.get_booster().feature_names]
    explanation, _ = explain.explain_prediction(model, row)
    expected_summary = explain.summarize_drivers(explanation, "AKBNK")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert expected_summary in [m.value for m in at.markdown]


def test_explanation_renders_contribution_waterfall(patched_download):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert list(at.exception) == []
    assert any(trace["type"] == "waterfall" for traces in _chart_traces(at) for trace in traces)


@pytest.fixture
def fixed_probability(monkeypatch):
    """Modelin artış olasılığını sabitler; UI'ın sinyal etiketlemesini
    eğitilmiş modelden bağımsız test etmeyi sağlar."""

    def _set(prob):
        monkeypatch.setattr(
            xgboost.XGBClassifier,
            "predict_proba",
            lambda self, X: np.tile([1.0 - prob, prob], (len(X), 1)),
        )

    return _set


def _boxes(at):
    return {
        "success": [e.value for e in at.success],
        "warning": [e.value for e in at.warning],
        "error": [e.value for e in at.error],
    }


@pytest.mark.parametrize(
    "prob, kind, label",
    [
        (0.60, "success", "YÜKSELİŞ"),
        (0.50, "warning", "NÖTR"),
        (0.40, "error", "DÜŞÜŞ"),
    ],
)
def test_direction_and_confidence_share_signal_color(
    patched_download, fixed_probability, prob, kind, label
):
    fixed_probability(prob)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert list(at.exception) == []
    boxes = _boxes(at)
    # Yön kutusu ve güven skoru aynı renkte (aynı türde) render edilmeli
    assert len(boxes[kind]) == 2
    assert any(label in v for v in boxes[kind])
    assert any(f"%{prob * 100:.1f}" in v for v in boxes[kind])
    assert sum(len(v) for v in boxes.values()) == 2


@pytest.mark.parametrize("prob", [0.49, 0.50, 0.51])
def test_coin_flip_probabilities_render_as_neutral(patched_download, fixed_probability, prob):
    fixed_probability(prob)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert any("NÖTR" in e.value for e in at.warning)
    assert not any("YÜKSELİŞ" in e.value for e in at.success)
    assert not any("DÜŞÜŞ" in e.value for e in at.error)


def test_min_confidence_slider_widens_neutral_band(patched_download, fixed_probability):
    fixed_probability(0.56)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    slider = at.sidebar.slider[0]
    assert slider.value == pytest.approx(0.53)
    assert any("YÜKSELİŞ" in e.value for e in at.success)

    slider.set_value(0.60).run()

    assert list(at.exception) == []
    assert any("NÖTR" in e.value for e in at.warning)
    assert not at.success


def _backtest_table(at):
    return next(df.value for df in at.dataframe if "Toplam Getiri (%)" in df.value.columns)


def test_backtest_compares_model_strategy_with_buy_and_hold(patched_download):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert list(at.exception) == []
    assert any("Backtest" in h.value for h in at.subheader)
    curves = [
        t
        for traces in _chart_traces(at)
        for t in traces
        if t.get("name") in ("Model Stratejisi", "Al ve Tut")
    ]
    assert sorted(t["name"] for t in curves) == ["Al ve Tut", "Model Stratejisi"]

    table = _backtest_table(at)
    assert list(table.index) == ["Model Stratejisi", "Al ve Tut"]
    assert list(table.columns) == ["Toplam Getiri (%)", "Sharpe Oranı", "Maks. Değer Kaybı (%)"]


def test_backtest_matches_buy_and_hold_when_always_bullish(patched_download, fixed_probability):
    fixed_probability(0.60)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    table = _backtest_table(at)
    assert table.loc["Al ve Tut", "Toplam Getiri (%)"] > 0
    pd.testing.assert_series_equal(
        table.loc["Model Stratejisi"], table.loc["Al ve Tut"], check_names=False
    )


def test_backtest_stays_in_cash_when_never_bullish(patched_download, fixed_probability):
    fixed_probability(0.50)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    table = _backtest_table(at)
    assert table.loc["Model Stratejisi", "Toplam Getiri (%)"] == 0
    assert table.loc["Model Stratejisi", "Maks. Değer Kaybı (%)"] == 0
    assert table.loc["Al ve Tut", "Toplam Getiri (%)"] > 0


def test_backtest_uses_selected_confidence_threshold(patched_download, fixed_probability):
    fixed_probability(0.56)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert _backtest_table(at).loc["Model Stratejisi", "Toplam Getiri (%)"] > 0

    at.sidebar.slider[0].set_value(0.60).run()
    assert _backtest_table(at).loc["Model Stratejisi", "Toplam Getiri (%)"] == 0


def test_backtest_caption_reports_exposure_and_trades(patched_download, fixed_probability):
    fixed_probability(0.60)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Hep yükseliş sinyali: tüm günlerde hissede, yalnızca ilk giriş işlemi
    assert any(
        "Hissede kalınan gün oranı: %100" in c.value and "Pozisyon değişikliği: 1" in c.value
        for c in at.caption
    )


def test_app_market_scanner_tab_triggers_and_renders(patched_download):
    """Fırsat Radarı sekmesindeki 'BIST 30 Taramasını Başlat' butonuna basıldığında
    Top 5 Yükseliş, Top 5 Düşüş ve tüm BIST 30 tablolarının hatasız render edildiğini doğrular."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Tarama butonuna tıkla ve yeniden çalıştır
    at.button(key="btn_start_scan").click().run()

    assert list(at.exception) == []
    assert not any("Bir hata oluştu" in e.value for e in at.error)

    # En az 4 tablo olmalı: Tab 1'deki indikatör tablosu + Tab 2'deki 3 sıralama tablosu
    assert len(at.dataframe) >= 4
    leaderboard = at.dataframe[-1].value
    assert "Hisse" in leaderboard.columns
    assert "Yükseliş Olasılığı (%)" in leaderboard.columns
    assert len(leaderboard) == len(config.TICKERS)


def test_cached_data_does_not_leak_between_tests_first(monkeypatch):
    """Aynı hisse için bir önceki testin önbelleğe aldığı veri, sonraki testte
    farklı bir veri ile yamalanan yf.download'ı gölgelememeli (bkz. _second)."""
    panel = _fake_download_panel()
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: panel.copy())

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert at.metric[0].value == f"{panel[('Close', 'AKBNK.IS')].iloc[-1]:.2f} TL"


def test_cached_data_does_not_leak_between_tests_second(monkeypatch):
    panel = _fake_download_panel()
    panel.iloc[:, :4] *= 3  # Fiyatlar 3 katı
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: panel.copy())

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert at.metric[0].value == f"{panel[('Close', 'AKBNK.IS')].iloc[-1]:.2f} TL"


def test_scanner_tab_stays_usable_when_selected_ticker_has_no_data(monkeypatch):
    """Yan menüde seçili hissenin verisi alınamadığında yalnızca o sekme uyarı
    göstermeli; Fırsat Radarı sekmesi kullanılabilir kalmalı."""
    panel = _fake_download_panel()
    monkeypatch.setattr(
        yfinance,
        "download",
        lambda ticker, *args, **kwargs: pd.DataFrame() if ticker == "AKBNK.IS" else panel.copy(),
    )

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert any("piyasa verisi alınamadı" in w.value for w in at.warning)
    assert any(b.key == "btn_start_scan" for b in at.button)


def test_scanner_lists_tickers_without_data(monkeypatch):
    panel = _fake_download_panel()
    single = panel.copy()
    single.columns = single.columns.get_level_values(0)

    def download(tickers, *args, **kwargs):
        if isinstance(tickers, str):
            return panel.copy()
        empty = pd.DataFrame(np.nan, index=single.index, columns=single.columns)
        return pd.concat({sym: (empty if sym == "SASA.IS" else single) for sym in tickers}, axis=1)

    monkeypatch.setattr(yfinance, "download", download)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="btn_start_scan").click().run()

    assert list(at.exception) == []
    assert any("1 hisse için veri alınamadı" in w.value and "SASA" in w.value for w in at.warning)
    assert len(at.dataframe[-1].value) == len(config.TICKERS) - 1


def test_refresh_button_appears_right_after_first_scan(patched_download):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not any(b.key == "btn_refresh_scan" for b in at.button)

    at.button(key="btn_start_scan").click().run()

    assert any(b.key == "btn_refresh_scan" for b in at.button)


def test_scanner_tables_use_consistent_number_formats(patched_download):
    import json

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="btn_start_scan").click().run()

    # Tab 1'deki indikatör tablosundan sonraki 3 tablo tarama tablolarıdır
    scan_tables = at.dataframe[-3:]
    assert len(scan_tables) == 3
    for table in scan_tables:
        formats = {
            col: cfg["type_config"]["format"]
            for col, cfg in json.loads(table.proto.columns).items()
            if "format" in cfg.get("type_config", {})
        }
        assert formats["Son Fiyat (TL)"] == "%.2f"
        assert formats["Günlük Değişim (%)"] == "%.2f"
        assert formats["Yükseliş Olasılığı (%)"] == "%.1f"


def test_unreachable_yahoo_shows_friendly_warning_instead_of_error(monkeypatch):
    def download(*args, **kwargs):
        raise ConnectionError("Max retries exceeded")

    monkeypatch.setattr(yfinance, "download", download)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert list(at.exception) == []
    assert list(at.error) == []
    assert any("bağlantı sorunu" in w.value for w in at.warning)


def test_app_renders_session_calendar_caption(patched_download):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert list(at.exception) == []
    assert any(
        "Analiz Edilen Son Kapanış" in c.value and "Hedef Seans" in c.value for c in at.caption
    )


def test_app_renders_weekend_notice_on_weekends(patched_download, monkeypatch):
    import src.session_calendar as sc

    orig_get_info = sc.get_session_info

    def fake_get_info(last_date, now=None):
        return orig_get_info(last_date, now=pd.Timestamp("2026-10-03 12:00:00"))  # Cumartesi

    monkeypatch.setattr(sc, "get_session_info", fake_get_info)

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert list(at.exception) == []
    assert any("Hafta Sonu Bildirimi" in i.value for i in at.info)


def test_app_renders_model_metadata_in_sidebar(patched_download):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert list(at.exception) == []
    sidebar_texts = [w.value for w in at.sidebar.get("markdown")]
    assert any("Sürüm:" in t for t in sidebar_texts)
    assert any("Test Doğruluğu:" in t for t in sidebar_texts)


@pytest.fixture
def counted_download(monkeypatch):
    """Tek sembol isteklerini sayan sahte yf.download; önbellek isabetini ölçmek için."""
    _disable_macro(monkeypatch)
    panel = _fake_download_panel()
    calls = []

    def download(tickers, *args, **kwargs):
        calls.append(tickers)
        return panel.copy()

    monkeypatch.setattr(yfinance, "download", download)
    return calls


def test_rerun_with_same_ticker_is_served_from_cache(counted_download):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert counted_download == ["AKBNK.IS"]

    at.sidebar.slider[0].set_value(0.60).run()

    assert list(at.exception) == []
    assert counted_download == ["AKBNK.IS"]


def test_refresh_data_button_refetches_selected_ticker(counted_download):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    at.sidebar.button(key="btn_refresh_data").click().run()

    assert list(at.exception) == []
    assert counted_download == ["AKBNK.IS", "AKBNK.IS"]


def test_refresh_data_button_keeps_other_tickers_cached(counted_download):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.selectbox[0].select("GARAN").run()
    at.sidebar.button(key="btn_refresh_data").click().run()

    at.sidebar.selectbox[0].select("AKBNK").run()

    assert list(at.exception) == []
    assert counted_download == ["AKBNK.IS", "GARAN.IS", "GARAN.IS"]


def test_app_renders_macro_indicators_expander(monkeypatch):
    panel = _fake_download_panel()
    macro_tuples = [("XU100.IS", "Close"), ("USDTRY=X", "Close")]
    cols = pd.MultiIndex.from_tuples(macro_tuples, names=["Ticker", "Price"])
    macro_df = pd.DataFrame(
        [[1000.0, 30.0], [1020.0, 30.3]],
        index=panel.index[-2:],
        columns=cols,
    )
    monkeypatch.setattr(config, "MACRO_TICKERS", ["XU100.IS", "USDTRY=X"])

    def download(tickers, *args, **kwargs):
        if isinstance(tickers, list) and set(tickers) == {"XU100.IS", "USDTRY=X"}:
            return macro_df
        return panel.copy()

    monkeypatch.setattr(yfinance, "download", download)
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert list(at.exception) == []
    expander_labels = [e.label for e in at.expander]
    assert any("Makro Piyasa Göstergeleri" in label for label in expander_labels)
