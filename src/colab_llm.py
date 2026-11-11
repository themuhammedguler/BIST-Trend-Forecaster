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


def _send_chat_completion(
    url: str,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    ctx: ssl.SSLContext,
) -> tuple[str, str]:
    """Colab /v1/chat/completions uç noktasına tek bir istek gönderir.

    Dönüş: (content, finish_reason)
    """
    endpoint = f"{url}/v1/chat/completions"
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
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
        if response.status == 200:
            body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                content = str(msg.get("content", "")).strip()
                finish_reason = str(choices[0].get("finish_reason", "stop")).lower()
                return content, finish_reason
    return "", "error"


def query_colab_llm(
    prompt: str,
    base_url: str | None = None,
    model: str = "bist-financial-llm",
    system_prompt: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: float = 45.0,
    auto_continue: bool = True,
    max_continuations: int = 2,
    custom_caller: Any = None,
) -> str:
    """Colab üzerinde koşan fine-tuned modele sorgu gönderir ve yanıtı döndürür."""
    if custom_caller is not None:
        return custom_caller(prompt)

    url = (get_colab_url() if base_url is None else base_url).rstrip("/")
    if not url:
        return "Colab LLM URL'si tanımlanmadı."

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        content, finish_reason = _send_chat_completion(
            url=url,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            ctx=ctx,
        )
        if not content:
            return "Modelden geçerli bir yanıt alınamadı."

        # Kesilme Kontrolü & Otomatik Tamamlama (Auto-Healer)
        continuations_left = max_continuations if auto_continue else 0
        while continuations_left > 0:
            is_cut_off = (
                finish_reason == "length"
                or content.rstrip().endswith((":", ",", "-", "(", "[", "\\"))
                or (
                    len(content) > 150
                    and not content.rstrip().endswith(
                        (".", "!", "?", "```", "---", "*", ")", "]", "}")
                    )
                )
            )
            if not is_cut_off:
                break

            logger.info(
                "Colab LLM yanıtı token sınırına (finish_reason=%s) takıldı; "
                "kaldığı yerden otomatik tamamlama başlatılıyor...",
                finish_reason,
            )
            cont_messages = list(messages)
            cont_messages.append({"role": "assistant", "content": content})
            cont_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Lütfen cümleni ve analizini kaldığın yerden aynen devam ettirerek "
                        "eksiksiz tamamla."
                    ),
                }
            )
            try:
                next_content, next_finish = _send_chat_completion(
                    url=url,
                    messages=cont_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    ctx=ctx,
                )
                if next_content:
                    sep = (
                        "\n\n"
                        if next_content.startswith(("#", "-", "*", "1.", "2.", "3."))
                        else " "
                    )
                    content = content.rstrip() + sep + next_content.lstrip()
                    finish_reason = next_finish
                else:
                    break
            except Exception as cont_exc:
                logger.debug("Otomatik tamamlama ek çağrısı başarısız: %s", cont_exc)
                break
            continuations_left -= 1

        return content
    except Exception as exc:
        logger.debug("Colab LLM çağrısı başarısız (%s): %s", url, exc)
        return f"Colab LLM yanıt veremedi: {exc}"
