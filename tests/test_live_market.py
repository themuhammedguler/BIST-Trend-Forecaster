"""Unit tests for the zero-delay live market quotes module."""

import numpy as np
import pandas as pd

from src import features
from src.live_market import (
    LiveQuote,
    apply_live_quote_to_frames,
    build_scanner_payload,
    clean_ticker_name,
    fetch_live_quotes,
    is_live_frame_recent,
)


def test_clean_ticker_name():
    assert clean_ticker_name("THYAO.IS") == "THYAO"
    assert clean_ticker_name("BIST:AKBNK") == "AKBNK"
    assert clean_ticker_name("  eregl.is  ") == "EREGL"


def test_live_quote_badges():
    q_pos = LiveQuote("THYAO", 310.0, 1.25, 1000000, 312.0, 305.0, 306.0)
    assert "🟢" in q_pos.badge
    assert "+1.25" in q_pos.badge

    q_neg = LiveQuote("AKBNK", 70.0, -0.85, 500000, 71.0, 69.5, 70.5)
    assert "🔴" in q_neg.badge
    assert "-0.85" in q_neg.badge

    q_zero = LiveQuote("EREGL", 30.0, 0.0, 200000, 30.2, 29.8, 30.0)
    assert "⚪" in q_zero.badge


def test_build_scanner_payload():
    payload = build_scanner_payload(["THYAO", "AKBNK"])
    assert payload["filter"][0]["right"] == ["THYAO", "AKBNK"]
    assert "close" in payload["columns"]
    assert "change" in payload["columns"]


def test_fetch_live_quotes_with_mock_fetcher():
    mock_response = {
        "data": [
            {
                "s": "BIST:THYAO",
                "d": ["THYAO", 315.5, 2.1, 45000000, 318.0, 310.0, 311.0, 0.5],
            },
            {
                "s": "BIST:GARAN",
                "d": ["GARAN", 155.0, -1.2, 30000000, 157.0, 154.0, 156.5, -0.2],
            },
        ]
    }

    def mock_fetcher(url, payload, timeout):
        return mock_response

    quotes = fetch_live_quotes(
        tickers=["THYAO.IS", "GARAN.IS"],
        custom_fetcher=mock_fetcher,
    )

    assert len(quotes) == 2
    assert "THYAO" in quotes
    assert quotes["THYAO"].last_price == 315.5
    assert quotes["THYAO"].change_pct == 2.1
    assert "GARAN" in quotes
    assert quotes["GARAN"].last_price == 155.0


def test_fetch_live_quotes_handles_network_error():
    def failing_fetcher(url, payload, timeout):
        raise ConnectionResetError("Remote server closed")

    quotes = fetch_live_quotes(
        tickers=["THYAO"],
        custom_fetcher=failing_fetcher,
    )
    assert quotes == {}


def test_is_live_frame_recent():
    old_df = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=10)})
    assert not is_live_frame_recent(old_df)

    today = pd.Timestamp.now(tz="Europe/Istanbul").tz_localize(None).normalize()
    recent_df = pd.DataFrame({"Date": [today - pd.Timedelta(days=1), today]})
    assert is_live_frame_recent(recent_df)


def test_apply_live_quote_to_frames_updates_candle_and_recomputes_indicators():
    dates = pd.bdate_range("2025-01-01", periods=120)
    close = 100.0 + np.linspace(0.0, 10.0, 120) + 3.0 * np.sin(np.arange(120))
    last_close_val = float(close[-1])
    full_df = pd.DataFrame(
        {
            "Date": dates,
            "ticker": "THYAO.IS",
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(120, 1_000_000.0),
        }
    )
    df_processed = features.add_features(full_df.copy(), drop_incomplete_target=False)
    orig_rsi = float(df_processed["rsi"].iloc[-1])

    quote = LiveQuote(
        ticker="THYAO",
        last_price=135.0,
        change_pct=12.5,
        volume=5_000_000.0,
        high=136.0,
        low=105.0,
        open_price=112.0,
    )

    # Without force=True on an old 2025 fixture, frame remains untouched
    unchanged_proc, unchanged_full = apply_live_quote_to_frames(df_processed, full_df, quote)
    assert float(unchanged_full["close"].iloc[-1]) == last_close_val

    # With force=True (or recent live dates), latest candle and indicators are updated
    updated_proc, updated_full = apply_live_quote_to_frames(
        df_processed, full_df, quote, force=True
    )
    assert float(updated_full["close"].iloc[-1]) == 135.0
    assert float(updated_full["high"].iloc[-1]) == 136.0
    assert float(updated_full["low"].iloc[-1]) == 105.0
    assert float(updated_full["volume"].iloc[-1]) == 5_000_000.0
    assert float(updated_proc["rsi"].iloc[-1]) > orig_rsi
