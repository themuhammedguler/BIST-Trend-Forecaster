"""tests/test_kap_llm.py - LLM destekli KAP bildirim analiz modülü birim testleri (#70)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.kap_llm import (
    LLMAnalysisResult,
    analyze_disclosures_with_llm,
    build_prompt,
    call_gemini_api,
    fallback_rule_based_analysis,
    get_impact_badge,
    parse_llm_json_response,
)
from src.kap_news import EventCategory, KAPDisclosure


@pytest.fixture
def sample_disclosures():
    return [
        KAPDisclosure(
            ticker="THYAO",
            title="Yeni Uçak Alım Sözleşmesi İmzalandı",
            summary="Filo genişletme kapsamında 20 yeni uçak alımı tamamlandı.",
            publish_date="2026-11-01",
            category=EventCategory.POSITIVE,
            badge="🟢 Yeni İş İlişkisi",
            event_type="yeni_is",
        ),
        KAPDisclosure(
            ticker="THYAO",
            title="Kâr Payı Dağıtımına İlişkin Yönetim Kurulu Kararı",
            summary="Hisse başına net 4.5 TL temettü dağıtılması genel kurula sunulacak.",
            publish_date="2026-11-02",
            category=EventCategory.POSITIVE,
            badge="🟢 Temettü Dağıtımı",
            event_type="temettu",
        ),
    ]


def test_get_impact_badge():
    assert "🟢" in get_impact_badge("Çok Olumlu", 0.6)
    assert "🟢" in get_impact_badge("Olumlu", 0.2)
    assert "⚪" in get_impact_badge("Nötr", 0.0)
    assert "🔴" in get_impact_badge("Olumsuz", -0.2)
    assert "🔴" in get_impact_badge("Çok Olumsuz", -0.7)


def test_build_prompt(sample_disclosures):
    prompt = build_prompt("THYAO", sample_disclosures)
    assert "THYAO" in prompt
    assert "Yeni Uçak Alım Sözleşmesi" in prompt
    assert "Kâr Payı Dağıtımına" in prompt
    assert "impact_rating" in prompt
    assert "impact_score" in prompt


def test_parse_llm_json_response_clean():
    raw_json = """
    {
        "bullets": [
            "Güçlü operasyonel büyüme ve filo genişlemesi kârlılığı artıracak.",
            "Temettü verimi yatırımcı algısını destekliyor."
        ],
        "impact_rating": "Olumlu",
        "impact_score": 0.45,
        "confidence": 0.88,
        "summary": "Pozitif haber akışı hisseyi destekliyor."
    }
    """
    result = parse_llm_json_response(raw_json, "THYAO")
    assert isinstance(result, LLMAnalysisResult)
    assert result.ticker == "THYAO"
    assert len(result.bullets) == 2
    assert result.impact_rating == "Olumlu"
    assert result.impact_score == 0.45
    assert result.confidence == 0.88
    assert result.provider == "gemini"
    assert "🟢" in result.badge


def test_parse_llm_json_response_with_markdown_fences():
    raw_json = """```json
    {
        "bullets": [
            "Şirket hakkında SPK incelemesi başlatıldı.",
            "Kısa vadede volatilite artabilir."
        ],
        "impact_rating": "Olumsuz",
        "impact_score": -0.35,
        "confidence": 0.75,
        "summary": "Hukuki riskler baskı oluşturabilir."
    }
    ```"""
    result = parse_llm_json_response(raw_json, "GARAN")
    assert result.impact_rating == "Olumsuz"
    assert result.impact_score == -0.35
    assert result.provider == "gemini"
    assert "🔴" in result.badge


def test_parse_llm_json_response_clamping_and_empty_defaults():
    raw_json = """
    {
        "bullets": [],
        "impact_rating": "BilinmeyenRating",
        "impact_score": 5.0,
        "confidence": 2.5,
        "summary": ""
    }
    """
    result = parse_llm_json_response(raw_json, "ASELS")
    assert result.impact_score == 1.0  # Clamped to 1.0
    assert result.confidence == 1.0  # Clamped to 1.0
    assert result.impact_rating == "Çok Olumlu"  # Inferred from score >= 0.40
    assert len(result.bullets) == 1
    assert "ASELS" in result.bullets[0]


def test_fallback_rule_based_analysis_empty():
    res = fallback_rule_based_analysis("EREGL", [])
    assert res.ticker == "EREGL"
    assert res.provider == "no_data"
    assert res.impact_score == 0.0
    assert res.impact_rating == "Nötr"


def test_fallback_rule_based_analysis_with_disclosures(sample_disclosures):
    res = fallback_rule_based_analysis("THYAO", sample_disclosures)
    assert res.ticker == "THYAO"
    assert res.provider == "rule_based"
    assert res.impact_score > 0.0
    assert res.impact_rating in ("Olumlu", "Çok Olumlu")
    assert len(res.bullets) > 0


def test_analyze_disclosures_with_llm_custom_caller_success(sample_disclosures):
    def mock_caller(prompt: str) -> str:
        return """{
            "bullets": ["Sözleşme hacmi güçlü."],
            "impact_rating": "Olumlu",
            "impact_score": 0.30,
            "confidence": 0.80,
            "summary": "Güçlü kurumsal görünüm."
        }"""

    res = analyze_disclosures_with_llm("THYAO", sample_disclosures, custom_caller=mock_caller)
    assert res.provider == "gemini"
    assert res.impact_score == 0.30
    assert res.bullets == ["Sözleşme hacmi güçlü."]


def test_analyze_disclosures_with_llm_custom_caller_failure_fallback(sample_disclosures):
    def faulty_caller(prompt: str) -> str:
        raise TimeoutError("LLM zaman aşımına uğradı")

    res = analyze_disclosures_with_llm("THYAO", sample_disclosures, custom_caller=faulty_caller)
    assert res.provider == "rule_based"
    assert res.error_message is not None
    assert "LLM zaman aşımına uğradı" in res.error_message


def test_analyze_disclosures_with_llm_no_api_key_graceful_fallback(sample_disclosures):
    with patch("src.kap_llm.get_gemini_api_key", return_value=""):
        res = analyze_disclosures_with_llm(
            "THYAO", sample_disclosures, api_key="", custom_caller=None
        )
        assert res.provider == "rule_based"
        assert res.error_message is not None
        assert "GEMINI_API_KEY tanımlanmadığı için" in res.error_message


def test_call_gemini_api_success():
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"impact_score": 0.25}'}]}}]
    }
    with patch("src.kap_llm.requests.post", return_value=fake_response):
        out = call_gemini_api("Test prompt", api_key="fake-key-123")
        assert '{"impact_score": 0.25}' in out


def test_call_gemini_api_http_error():
    fake_response = MagicMock()
    fake_response.status_code = 403
    fake_response.text = "Forbidden / Quota Exceeded"
    with patch("src.kap_llm.requests.post", return_value=fake_response):
        with pytest.raises(RuntimeError) as exc_info:
            call_gemini_api("Test prompt", api_key="fake-key-123")
        assert "HTTP 403" in str(exc_info.value)


def test_analyze_disclosures_with_llm_network_error_fallback(sample_disclosures):
    with patch("src.kap_llm.requests.post", side_effect=requests.ConnectionError("Ağ koptu")):
        res = analyze_disclosures_with_llm(
            "THYAO", sample_disclosures, api_key="fake_key", custom_caller=None
        )
        assert res.provider == "rule_based"
        assert res.error_message is not None
        assert "Ağ koptu" in res.error_message
