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
