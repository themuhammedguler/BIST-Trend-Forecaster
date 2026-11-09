"""Tests for the Financial Analyst Chatbot & Fundamental RAG integration module."""

import json
from pathlib import Path
from unittest.mock import patch

from src.fundamental_rag import (
    FundamentalReport,
    KeyMetrics,
    build_chatbot_deeplink,
    clean_ticker,
    fetch_fundamental_report,
    parse_report_payload,
)


def test_clean_ticker_normalizes_symbols():
    assert clean_ticker("THYAO.IS") == "THYAO"
    assert clean_ticker("akbnk.is") == "AKBNK"
    assert clean_ticker("tuprs") == "TUPRS"
    assert clean_ticker("  kchol.is  ") == "KCHOL"


def test_build_chatbot_deeplink():
    link = build_chatbot_deeplink("THYAO.IS", base_url="https://example-chatbot.streamlit.app/")
    assert link == "https://example-chatbot.streamlit.app/?ticker=THYAO"

    default_link = build_chatbot_deeplink("akbnk")
    assert "?ticker=AKBNK" in default_link


def test_parse_report_payload_complete():
    payload = {
        "ticker": "THYAO",
        "company_name": "Türk Hava Yolları",
        "report_date": "2026-11-08",
        "fundamental_summary": "Güçlü yolcu büyümesi ve FAVÖK artışı.",
        "sentiment_score": 0.85,
        "sentiment_label": "Pozitif",
        "key_metrics": {
            "pe_ratio": 4.8,
            "pb_ratio": 1.15,
            "net_profit_margin_trend": "Artış",
            "debt_to_equity": "Düşük/Yönetilebilir",
            "extra_metrics": {"roe": 0.28},
        },
        "catalysts": ["Filo büyümesi", "Yaz rezervasyonları"],
        "risk_factors": ["Yakıt maliyetleri"],
    }
    report = parse_report_payload(payload, source="test")
    assert report.ticker == "THYAO"
    assert report.company_name == "Türk Hava Yolları"
    assert report.sentiment_score == 0.85
    assert report.sentiment_label == "Pozitif"
    assert "🟢 Pozitif (+0.85)" in report.badge
    assert report.key_metrics.pe_ratio == 4.8
    assert report.key_metrics.pb_ratio == 1.15
    assert report.key_metrics.net_profit_margin_trend == "Artış"
    assert report.catalysts == ["Filo büyümesi", "Yaz rezervasyonları"]
    assert report.risk_factors == ["Yakıt maliyetleri"]
    assert report.source == "test"


def test_parse_report_payload_resilient_to_missing_fields():
    payload = {"ticker": "XYZ"}
    report = parse_report_payload(payload, source="partial")
    assert report.ticker == "XYZ"
    assert report.sentiment_score == 0.0
    assert report.sentiment_label == "Nötr"
    assert report.key_metrics.pe_ratio is None
    assert report.key_metrics.pb_ratio is None
    assert report.catalysts == []
    assert report.risk_factors == []
    assert "🟡 Nötr" in report.badge


def test_badge_variants():
    km = KeyMetrics()
    pos = FundamentalReport(
        ticker="T",
        company_name="T",
        report_date="2026",
        fundamental_summary="",
        sentiment_score=0.6,
        sentiment_label="Pozitif",
        key_metrics=km,
    )
    assert "🟢" in pos.badge

    neg = FundamentalReport(
        ticker="T",
        company_name="T",
        report_date="2026",
        fundamental_summary="",
        sentiment_score=-0.4,
        sentiment_label="Negatif",
        key_metrics=km,
    )
    assert "🔴" in neg.badge

    neutral = FundamentalReport(
        ticker="T",
        company_name="T",
        report_date="2026",
        fundamental_summary="",
        sentiment_score=0.0,
        sentiment_label="Nötr",
        key_metrics=km,
    )
    assert "🟡" in neutral.badge


def test_fetch_fundamental_report_from_local_dir(tmp_path: Path):
    sample_data = {
        "ticker": "AKBNK",
        "company_name": "Akbank",
        "report_date": "2026-11-08",
        "fundamental_summary": "Net faiz marjı toparlanması.",
        "sentiment_score": 0.5,
        "sentiment_label": "Pozitif",
        "key_metrics": {"pe_ratio": 3.2, "pb_ratio": 0.8},
        "catalysts": ["Kredi büyümesi"],
        "risk_factors": ["Enflasyon muhasebesi"],
    }
    report_file = tmp_path / "AKBNK_latest.json"
    report_file.write_text(json.dumps(sample_data), encoding="utf-8")

    report = fetch_fundamental_report("AKBNK.IS", local_reports_dir=tmp_path)
    assert report is not None
    assert report.ticker == "AKBNK"
    assert report.source == "local_cache"
    assert report.key_metrics.pe_ratio == 3.2


def test_fetch_fundamental_report_via_mocked_github_raw():
    sample = {
        "ticker": "THYAO",
        "company_name": "THY",
        "report_date": "2026-11-08",
        "fundamental_summary": "Yüksek doluluk.",
        "sentiment_score": 0.7,
        "sentiment_label": "Pozitif",
    }
    with patch("src.fundamental_rag._fetch_from_url", return_value=sample):
        report = fetch_fundamental_report("THYAO", api_url="", raw_repo="user/repo")
        assert report is not None
        assert report.ticker == "THYAO"
        assert report.source == "github_raw"


def test_fetch_fundamental_report_via_mocked_api():
    sample = {
        "ticker": "TUPRS",
        "company_name": "Tüpraş",
        "report_date": "2026-11-08",
        "fundamental_summary": "Rafineri marjları güçlü.",
        "sentiment_score": 0.4,
        "sentiment_label": "Pozitif",
    }
    with patch("src.fundamental_rag._fetch_from_url", return_value=sample):
        report = fetch_fundamental_report("TUPRS", api_url="https://api.example.com")
        assert report is not None
        assert report.ticker == "TUPRS"
        assert report.source == "api"


def test_fetch_fundamental_report_graceful_fallback_on_error():
    with patch("src.fundamental_rag._fetch_from_url", return_value=None):
        report = fetch_fundamental_report("UNKNOWN", api_url="", raw_repo="user/repo")
        assert report is None
