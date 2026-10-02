# tests/test_app_ui.py - app.py'yi streamlit.testing.v1.AppTest ile uçtan uca test eder
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yfinance
from streamlit.testing.v1 import AppTest

import features

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

DISPLAYED_COLUMNS = ['rsi', 'macd', 'sma_10', 'sma_50', 'volatility']


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
    df.rename(columns={c: ("Date" if c.lower() == "date" else c if c == "ticker" else c.lower())
                       for c in df.columns}, inplace=True)
    processed = features.add_features(df, drop_incomplete_target=False)
    return processed.iloc[[-1]]


@pytest.fixture
def patched_download(monkeypatch):
    panel = _fake_download_panel()
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: panel.copy())
    return panel


def test_app_renders_a_single_prediction_without_error(patched_download):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert list(at.exception) == []
    assert not any("Bir hata oluştu" in e.value for e in at.error)

    # Tam olarak bir yön kutusu render edilmeli: yukarı (success) ya da düşüş (error).
    assert len(at.success) + len(at.error) == 1
    assert len(at.metric) == 1
    assert len(at.info) == 1


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
    assert len(leaderboard) > 0
