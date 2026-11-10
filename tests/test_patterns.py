"""BIST 30 Mum Formasyonları ve Destek/Direnç modülü testleri."""

import pandas as pd
import pytest

from src import patterns


@pytest.fixture
def sample_ohlc_df():
    """Örnek OHLCV veri çerçevesi."""
    data = {
        "Date": pd.date_range("2026-11-01", periods=10, freq="B").strftime("%Y-%m-%d"),
        "open": [100.0, 102.0, 101.0, 99.0, 95.0, 96.0, 98.0, 101.0, 103.0, 105.0],
        "high": [103.0, 104.0, 102.0, 100.0, 96.0, 99.0, 101.0, 104.0, 106.0, 108.0],
        "low": [99.0, 100.0, 98.0, 94.0, 92.0, 93.0, 97.0, 100.0, 102.0, 104.0],
        "close": [102.0, 101.0, 99.0, 95.0, 96.0, 98.0, 101.0, 103.0, 105.0, 107.0],
        "volume": [1000] * 10,
    }
    return pd.DataFrame(data)


def test_detect_candlestick_patterns_empty():
    assert patterns.detect_candlestick_patterns(None) == []
    assert patterns.detect_candlestick_patterns(pd.DataFrame()) == []


def test_detect_candlestick_patterns_doji():
    # Son mum açık ve kapanışı eşit (Doji)
    df = pd.DataFrame(
        {
            "Date": ["2026-11-01", "2026-11-02", "2026-11-03"],
            "open": [100.0, 101.0, 100.0],
            "high": [102.0, 103.0, 105.0],
            "low": [98.0, 99.0, 95.0],
            "close": [101.0, 100.0, 100.05],  # Body çok küçük
        }
    )
    detected = patterns.detect_candlestick_patterns(df)
    assert any(p.name == "Doji" for p in detected)


def test_detect_candlestick_patterns_hammer():
    # Düşüş sonrası uzun alt fitil ve üstte küçük gövde (Hammer)
    df = pd.DataFrame(
        {
            "Date": ["2026-11-01", "2026-11-02", "2026-11-03"],
            "open": [105.0, 100.0, 95.0],
            "high": [106.0, 101.0, 96.5],
            "low": [99.0, 94.0, 88.0],  # Uzun alt fitil (7 birim)
            "close": [100.0, 95.0, 96.0],  # Gövde: 1.0 (95-96)
        }
    )
    detected = patterns.detect_candlestick_patterns(df)
    assert any(p.name == "Hammer" for p in detected)


def test_detect_candlestick_patterns_bullish_engulfing():
    # 1. gün kırmızı mum, 2. gün yeşil mum önceki gövdeyi yutuyor
    df = pd.DataFrame(
        {
            "Date": ["2026-11-01", "2026-11-02", "2026-11-03"],
            "open": [100.0, 102.0, 96.0],
            "high": [102.0, 103.0, 105.0],
            "low": [98.0, 97.0, 95.0],
            "close": [101.0, 98.0, 104.0],  # 96'dan açılıp 104'e gitti, 102-98'i yuttu
        }
    )
    detected = patterns.detect_candlestick_patterns(df)
    assert any(p.name == "Bullish_Engulfing" for p in detected)


def test_calculate_support_resistance(sample_ohlc_df):
    sr = patterns.calculate_support_resistance(sample_ohlc_df, window=5)
    assert sr is not None
    assert sr.pivot > 0
    assert sr.resistance_1 >= sr.pivot
    assert sr.support_1 <= sr.pivot
    assert sr.status in [
        "BREAKOUT_BULLISH",
        "BREAKDOWN_BEARISH",
        "TESTING_RESISTANCE",
        "TESTING_SUPPORT",
        "IN_RANGE",
    ]


def test_calculate_relative_strength(sample_ohlc_df):
    index_df = sample_ohlc_df.copy()
    index_df["close"] = index_df["close"] * 10.0

    rs_df = patterns.calculate_relative_strength(sample_ohlc_df, index_df, window=5)
    assert rs_df is not None
    assert "rs_ratio" in rs_df.columns
    assert "outperforming" in rs_df.columns
