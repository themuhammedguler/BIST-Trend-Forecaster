"""Unit tests for the zero-delay live market quotes module."""

from src.live_market import (
    LiveQuote,
    build_scanner_payload,
    clean_ticker_name,
    fetch_live_quotes,
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
