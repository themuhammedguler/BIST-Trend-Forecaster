"""Tests for the Financial Analyst Chatbot & Fundamental RAG integration module."""

import json
from pathlib import Path
from unittest.mock import patch

from src.fundamental_rag import (
    FundamentalReport,
    KeyMetrics,
    ask_hybrid_analyst,
    build_chatbot_deeplink,
    clean_ticker,
    convert_disclosures_to_kap,
    enrich_key_metrics_with_live_market_data,
    fetch_fundamental_report,
    fetch_macro_stress_test,
    fetch_peer_comparison,
    fetch_quant_metrics,
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


def test_enrich_key_metrics_skips_when_already_populated():
    original = KeyMetrics(pe_ratio=5.5, pb_ratio=1.2)
    with patch("yfinance.Ticker") as mock_ticker:
        result = enrich_key_metrics_with_live_market_data("AKBNK", original)
        assert result == original
        mock_ticker.assert_not_called()


def test_enrich_key_metrics_fills_missing_ratios():
    original = KeyMetrics(pe_ratio=None, pb_ratio=None)
    fake_info = {"trailingPE": 6.85, "priceToBook": 1.42}
    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = mock_ticker_cls.return_value
        mock_instance.info = fake_info
        result = enrich_key_metrics_with_live_market_data("THYAO", original)
        assert result.pe_ratio == 6.85
        assert result.pb_ratio == 1.42


def test_enrich_key_metrics_handles_yfinance_error_gracefully():
    original = KeyMetrics(pe_ratio=None, pb_ratio=None)
    with patch("yfinance.Ticker", side_effect=RuntimeError("Network down")):
        result = enrich_key_metrics_with_live_market_data("THYAO", original)
        assert result.pe_ratio is None
        assert result.pb_ratio is None


def test_convert_disclosures_to_kap_positive_negative_neutral():
    raw = [
        {
            "Tarih": "2026-11-05",
            "Kategori": "Özel Durum Açıklaması",
            "Başlık": "Yeni filo alımı",
            "Etki": "Pozitif",
        },
        {
            "Tarih": "2026-10-20",
            "Kategori": "Dava / Ceza",
            "Başlık": "Vergi cezası tebliği",
            "Etki": "Negatif",
        },
        {
            "Tarih": "2026-10-01",
            "Kategori": "Genel Kurul",
            "Başlık": "Olağan genel kurul toplantısı",
            "Etki": "Nötr",
        },
    ]
    kap_list = convert_disclosures_to_kap("THYAO.IS", raw)
    assert len(kap_list) == 3
    assert kap_list[0].ticker == "THYAO"
    assert "🟢" in kap_list[0].badge
    assert kap_list[0].category.value == "positive"
    assert "🔴" in kap_list[1].badge
    assert kap_list[1].category.value == "negative"
    assert "⚪" in kap_list[2].badge
    assert kap_list[2].category.value == "neutral"


def test_fundamental_report_kap_disclosures_property():
    report = parse_report_payload(
        {
            "ticker": "EREGL",
            "recent_disclosures": [
                {
                    "Tarih": "2026-11-02",
                    "Kategori": "Yatırım",
                    "Başlık": "Peletleme tesisi",
                    "Etki": "Pozitif",
                }
            ],
        }
    )
    disclosures = report.kap_disclosures
    assert len(disclosures) == 1
    assert disclosures[0].ticker == "EREGL"
    assert disclosures[0].title == "Peletleme tesisi"


def test_fetch_fundamental_report_fallback_to_data_reports_path():
    called_urls = []

    def mock_fetch(url, timeout=5.0):
        called_urls.append(url)
        if "data/reports/SISE.json" in url:
            return {"ticker": "SISE", "sentiment_score": 0.5, "sentiment_label": "Pozitif"}
        return None

    with patch("src.fundamental_rag._fetch_from_url", side_effect=mock_fetch):
        rep = fetch_fundamental_report("SISE", api_url="", raw_repo="org/repo")
        assert rep is not None
        assert rep.ticker == "SISE"
        assert any("data/reports/SISE.json" in u for u in called_urls)


def test_fetch_quant_metrics_deterministic_and_api():
    quant = fetch_quant_metrics("THYAO.IS", api_url="")
    assert quant["ticker"] == "THYAO"
    assert quant["source"] == "deterministic_engine"
    assert quant["dupont"]["roe_pct"] > 0
    assert "z_score" in quant["altman_z"]
    assert quant["dcf_valuation"]["estimated_fair_value_million_tl"] > 0
    assert "dividend_safety_score" in quant["dividend_safety"]
    assert "recommended_portfolio_allocation_pct" in quant["kelly_allocation"]

    mock_api_resp = {
        "ticker": "THYAO",
        "dupont": {"roe_pct": 31.5},
        "altman_z": {"z_score": 2.95, "zone": "Güvenli Bölge"},
    }
    with patch("src.fundamental_rag._fetch_from_url", return_value=mock_api_resp):
        quant_api = fetch_quant_metrics("THYAO", api_url="http://localhost:8000")
        assert quant_api["source"] == "api_v2"
        assert quant_api["dupont"]["roe_pct"] == 31.5


def test_fetch_macro_stress_test_deterministic():
    stress_thy = fetch_macro_stress_test(
        "THYAO",
        usd_change_pct=10.0,
        rate_change_bps=250.0,
        oil_change_pct=15.0,
        api_url="",
    )
    assert stress_thy["ticker"] == "THYAO"
    assert "total_impact_pct" in stress_thy
    assert stress_thy["scenarios"]["usd_try"]["impact_pct"] == 7.2
    assert stress_thy["scenarios"]["brent_oil"]["impact_pct"] == -8.5

    stress_tuprs = fetch_macro_stress_test(
        "TUPRS",
        usd_change_pct=10.0,
        rate_change_bps=0.0,
        oil_change_pct=15.0,
        api_url="",
    )
    assert stress_tuprs["total_impact_pct"] > 5.0
    assert "🚀" in stress_tuprs["resilience_badge"]


def test_fetch_peer_comparison_deterministic():
    peer = fetch_peer_comparison("FROTO.IS", "EREGL.IS", api_url="")
    assert peer["company_a"]["ticker"] == "FROTO"
    assert peer["company_b"]["ticker"] == "EREGL"
    assert peer["quality_leader"] == "FROTO"
    assert "FROTO" in peer["analyst_rationale"]


def test_ask_hybrid_analyst_fallback_with_quant_context():
    with patch("src.colab_llm.query_colab_llm", return_value="THYAO ROE %30.1") as mock_q:
        res = ask_hybrid_analyst("THYAO ROE nedir?", ticker="THYAO", api_url="")
        assert res["answer"] == "THYAO ROE %30.1"
        assert res["source"] == "colab_llm_with_quant_context"
        sent_prompt = mock_q.call_args[0][0]
        assert "ROE=%30.1" in sent_prompt
