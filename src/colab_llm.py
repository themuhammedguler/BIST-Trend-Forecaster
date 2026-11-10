"""Google Colab Pro fine-tuned BIST LLM istemcisi.

Kullanıcının Colab üzerinde eğittiği ve ngrok üzerinden dışa aktardığı
'bist-financial-llm' modeline bağlanır; bilanço, KAP ve hisse analiz
sorgularını anlık olarak yanıtlar.
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.request
from typing import Any

from src import config

logger = logging.getLogger(__name__)


def get_colab_url() -> str:
    """Aktif Colab ngrok URL'sini döndürür."""
    return getattr(config, "COLAB_LLM_URL", "").rstrip("/")


def check_colab_health(base_url: str | None = None, timeout: float = 4.0) -> bool:
    """Colab LLM sunucusunun aktif ve erişilebilir olup olmadığını doğrular."""
    url = (get_colab_url() if base_url is None else base_url).rstrip("/")
    if not url:
        return False
    endpoint = f"{url}/v1/models"
    req = urllib.request.Request(
        endpoint,
        headers={
            "User-Agent": "BIST-Trend-Forecaster/1.0",
            "ngrok-skip-browser-warning": "true",
        },
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.status == 200
    except Exception:
        return False


def query_colab_llm(
    prompt: str,
    base_url: str | None = None,
    model: str = "bist-financial-llm",
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 250,
    timeout: float = 25.0,
    custom_caller: Any = None,
) -> str:
    """Colab üzerinde koşan fine-tuned modele sorgu gönderir ve yanıtı döndürür."""
    if custom_caller is not None:
        return custom_caller(prompt)

    url = (get_colab_url() if base_url is None else base_url).rstrip("/")
    if not url:
        return "Colab LLM URL'si tanımlanmadı."

    endpoint = f"{url}/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "BIST-Trend-Forecaster/1.0",
            "ngrok-skip-browser-warning": "true",
        },
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            if response.status == 200:
                body = json.loads(response.read().decode("utf-8"))
                choices = body.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message", {})
                    return str(msg.get("content", "")).strip()
    except Exception as exc:
        logger.debug("Colab LLM çağrısı başarısız (%s): %s", url, exc)
        return f"Colab LLM yanıt veremedi: {exc}"

    return "Modelden geçerli bir yanıt alınamadı."
