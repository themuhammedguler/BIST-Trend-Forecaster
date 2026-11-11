"""Unit tests for the Google Colab Pro LLM client."""

from unittest.mock import patch

from src.colab_llm import check_colab_health, get_colab_url, query_colab_llm


def test_get_colab_url_default():
    url = get_colab_url()
    assert url.startswith("http")


def test_check_colab_health_empty_url():
    assert check_colab_health(base_url="") is False


def test_query_colab_llm_custom_caller():
    def dummy_caller(prompt):
        return f"Echo: {prompt}"

    res = query_colab_llm("Test prompt", custom_caller=dummy_caller)
    assert res == "Echo: Test prompt"


def test_query_colab_llm_missing_url():
    with patch("src.colab_llm.get_colab_url", return_value=""):
        res = query_colab_llm("Hello")
        assert "tanımlanmadı" in res


def test_query_colab_llm_auto_continue_on_length_truncation():
    # İlk çağrı token sınırına takıldı (finish_reason = length)
    call_count = 0

    def mock_send(url, messages, model, temperature, max_tokens, timeout, ctx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "DuPont analizi sonuçları:", "length"
        return "ROE %28.5 ve riskler düşüktür.", "stop"

    with patch("src.colab_llm._send_chat_completion", side_effect=mock_send):
        res = query_colab_llm("ARCLK analizi yap", auto_continue=True)
        assert call_count == 2
        assert "DuPont analizi sonuçları:" in res
        assert "ROE %28.5" in res


def test_query_colab_llm_respects_auto_continue_false():
    call_count = 0

    def mock_send(url, messages, model, temperature, max_tokens, timeout, ctx):
        nonlocal call_count
        call_count += 1
        return "Yarım kalan metin:", "length"

    with patch("src.colab_llm._send_chat_completion", side_effect=mock_send):
        res = query_colab_llm("ARCLK analizi yap", auto_continue=False)
        assert call_count == 1
        assert res == "Yarım kalan metin:"
