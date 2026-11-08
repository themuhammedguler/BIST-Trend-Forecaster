"""LLM destekli KAP bildirim ve finansal rapor analiz modülü (#70).

KAP ve şirket haber metinlerini Google Gemini API kullanarak analiz eder,
kârlılık, borçluluk, yeni anlaşmalar ve riskler üzerine 2-3 maddelik özet
ile etki skoru üretir. API anahtarı girilmediğinde veya kota aşıldığında
kural tabanlı analize şeffaf biçimde geri döner (graceful degradation).
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import requests

from src import config
from src.kap_news import EventCategory, KAPDisclosure, summarize_disclosures

VALID_RATINGS = {"Çok Olumlu", "Olumlu", "Nötr", "Olumsuz", "Çok Olumsuz"}


@dataclass
class LLMAnalysisResult:
    """LLM veya kural tabanlı yedek analiz sonucunu temsil eden veri sınıfı."""

    ticker: str
    bullets: List[str]
    impact_rating: str
    impact_score: float
    confidence: float
    summary: str
    provider: str
    badge: str
    error_message: Optional[str] = None


def get_gemini_api_key() -> str:
    """Ortam değişkenlerinden veya Streamlit gizli dizilerinden API anahtarını çeker."""
    env_key = os.getenv("GEMINI_API_KEY", "") or config.GEMINI_API_KEY
    if env_key:
        return env_key.strip()

    # Streamlit secrets kontrolü (eğer streamlit çalışma anında mevcutsa)
    st_mod = sys.modules.get("streamlit")
    if st_mod is not None:
        try:
            secrets = getattr(st_mod, "secrets", None)
            if secrets is not None and "GEMINI_API_KEY" in secrets:
                return str(secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass

    return ""


def get_impact_badge(rating: str, score: float) -> str:
    """Etki derecesi ve puanına göre görsel rozet metni üretir."""
    if score >= 0.40 or rating == "Çok Olumlu":
        return f"🟢 Çok Olumlu ({score:+.2f})"
    if score > 0.05 or rating == "Olumlu":
        return f"🟢 Olumlu ({score:+.2f})"
    if score <= -0.40 or rating == "Çok Olumsuz":
        return f"🔴 Çok Olumsuz ({score:+.2f})"
    if score < -0.05 or rating == "Olumsuz":
        return f"🔴 Olumsuz ({score:+.2f})"
    return f"⚪ Nötr ({score:+.2f})"


def build_prompt(ticker: str, disclosures: Sequence[KAPDisclosure]) -> str:
    """Finansal analiz için yapılandırılmış LLM istemi (prompt) oluşturur."""
    formatted_items = []
    for i, d in enumerate(disclosures, start=1):
        formatted_items.append(
            f"{i}. [{d.publish_date}] {d.badge} | Başlık: {d.title}\n"
            f"   Özet: {d.summary if d.summary else 'Detay bulunmuyor.'}"
        )
    formatted_text = "\n".join(formatted_items)

    prompt = (
        f"Sen Borsa İstanbul (BIST) ve Türk finansal piyasaları konusunda uzman kıdemli bir "
        f"finansal analistsin. Aşağıda {ticker} şirketine ait en güncel KAP "
        f"(Kamuyu Aydınlatma Platformu) bildirimleri ve haber başlıkları verilmiştir:\n\n"
        f"{formatted_text}\n\n"
        "GÖREVİN:\n"
        "1. Bu bildirimleri şirket kârlılığı, borçluluk yapısı, sipariş/ihale büyümesi, "
        "temettü/sermaye politikası ve yasal riskler açısından değerlendir.\n"
        "2. Profesyonel, somut ve Türkçe 2 veya 3 maddelik analiz özeti oluştur (bullets).\n"
        "3. Hissesi üzerindeki genel etkiyi şu alanlarla JSON olarak derecelendir:\n"
        "   - 'impact_rating': 'Çok Olumlu', 'Olumlu', 'Nötr', 'Olumsuz', 'Çok Olumsuz' "
        "seçeneklerinden biri olmalıdır.\n"
        "   - 'impact_score': -1.0 (en negatif) ile +1.0 (en pozitif) "
        "arasında ondalıklı bir sayı.\n"
        "   - 'confidence': 0.0 ile 1.0 arasında güven skoru.\n"
        "   - 'summary': 1-2 cümlelik genel kurumsal değerlendirme.\n\n"
        "LÜTFEN YALNIZCA AŞAĞIDAKİ ŞABLONA UYGUN GEÇERLİ BİR JSON YANITI DÖN "
        "(Markdown veya ek metin ekleme):\n"
        "{\n"
        '  "bullets": [\n'
        '    "Operasyonel kârlılık ve sipariş akışı kısa vadeli teknik trendi destekliyor.",\n'
        '    "Finansman giderleri ve borçluluk rasyoları yakından izlenmeli."\n'
        "  ],\n"
        '  "impact_rating": "Olumlu",\n'
        '  "impact_score": 0.35,\n'
        '  "confidence": 0.85,\n'
        '  "summary": "Son dönem bildirimleri kurumsal büyümeyi teyit ediyor."\n'
        "}"
    )
    return prompt


def parse_llm_json_response(raw_text: str, ticker: str) -> LLMAnalysisResult:
    """LLM çıktısından JSON bloğunu ayıklar, doğrular ve LLMAnalysisResult üretir."""
    cleaned = raw_text.strip()
    # Markdown kod bloğu temizleme
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM yanıtı bir JSON nesnesi (dict) değil.")

    raw_bullets = data.get("bullets", [])
    if isinstance(raw_bullets, list):
        bullets = [str(b).strip() for b in raw_bullets if str(b).strip()]
    elif isinstance(raw_bullets, str) and raw_bullets.strip():
        bullets = [raw_bullets.strip()]
    else:
        bullets = []

    if not bullets:
        bullets = [f"{ticker} bildirimleri nötr finansal görünüm sergilemektedir."]

    bullets = bullets[:3]

    try:
        score = float(data.get("impact_score", 0.0))
        score = max(-1.0, min(1.0, score))
    except (ValueError, TypeError):
        score = 0.0

    try:
        conf = float(data.get("confidence", 0.70))
        conf = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        conf = 0.70

    rating = str(data.get("impact_rating", "")).strip()
    if rating not in VALID_RATINGS:
        if score >= 0.40:
            rating = "Çok Olumlu"
        elif score > 0.05:
            rating = "Olumlu"
        elif score <= -0.40:
            rating = "Çok Olumsuz"
        elif score < -0.05:
            rating = "Olumsuz"
        else:
            rating = "Nötr"

    summary = str(data.get("summary", "")).strip()
    if not summary:
        summary = f"{ticker} son KAP bildirimleri doğrultusunda {rating.lower()} eğilimdedir."

    badge = get_impact_badge(rating, score)

    return LLMAnalysisResult(
        ticker=ticker,
        bullets=bullets,
        impact_rating=rating,
        impact_score=round(score, 3),
        confidence=round(conf, 2),
        summary=summary,
        provider="gemini",
        badge=badge,
        error_message=None,
    )


def fallback_rule_based_analysis(
    ticker: str,
    disclosures: Sequence[KAPDisclosure],
    error_message: Optional[str] = None,
) -> LLMAnalysisResult:
    """Kural tabanlı anahtar kelime motoru ile yedek temel analiz üretir (#70)."""
    norm_ticker = ticker.replace(".IS", "")
    if not disclosures:
        return LLMAnalysisResult(
            ticker=norm_ticker,
            bullets=[
                "Son dönemde şirket için dikkat çeken yeni bir KAP bildirimi veya finansal haber "
                "bulunmuyor."
            ],
            impact_rating="Nötr",
            impact_score=0.0,
            confidence=0.50,
            summary="İncelenen dönemde haber akışı sakin; teknik göstergeler izlenmeli.",
            provider="no_data",
            badge="⚪ Nötr (+0.00)",
            error_message=error_message,
        )

    summary_info = summarize_disclosures(list(disclosures))
    pos = summary_info["positive_count"]
    neg = summary_info["negative_count"]
    total = max(1, summary_info["total"])

    score = (pos - neg) / total
    score = max(-1.0, min(1.0, score))

    if score >= 0.40:
        rating = "Çok Olumlu"
    elif score > 0.05:
        rating = "Olumlu"
    elif score <= -0.40:
        rating = "Çok Olumsuz"
    elif score < -0.05:
        rating = "Olumsuz"
    else:
        rating = "Nötr"

    badge = get_impact_badge(rating, score)

    bullets: List[str] = []
    pos_items = [d for d in disclosures if d.category == EventCategory.POSITIVE]
    neg_items = [d for d in disclosures if d.category == EventCategory.NEGATIVE]

    if pos_items:
        titles = ", ".join(f"'{d.title[:35]}...'" for d in pos_items[:2])
        bullets.append(f"Pozitif Gelişmeler ({len(pos_items)} bildirim): {titles}")
    if neg_items:
        titles = ", ".join(f"'{d.title[:35]}...'" for d in neg_items[:2])
        bullets.append(f"Dikkat Edilmesi Gereken Riskler ({len(neg_items)} bildirim): {titles}")

    bullets.append(
        f"Kural tabanlı genel piyasa algısı: {summary_info['overall_sentiment']} "
        f"({pos} pozitif, {neg} negatif bildirim)."
    )

    return LLMAnalysisResult(
        ticker=norm_ticker,
        bullets=bullets[:3],
        impact_rating=rating,
        impact_score=round(score, 3),
        confidence=0.65,
        summary=summary_info["warning_note"],
        provider="rule_based",
        badge=badge,
        error_message=error_message,
    )


def call_gemini_api(
    prompt: str,
    api_key: str,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
) -> str:
    """Google Gemini REST API üzerinden istemi çalıştırır."""
    if not api_key:
        raise ValueError("Google Gemini API anahtarı boş olamaz.")

    chosen_model = model or config.GEMINI_MODEL or "gemini-1.5-flash"
    timeout_sec = timeout or config.GEMINI_TIMEOUT_SECONDS or 10

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{chosen_model}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800,
            "responseMimeType": "application/json",
        },
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API yanıt hatası (HTTP {response.status_code}): {response.text[:200]}"
        )

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini API boş aday listesi döndürdü.")

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts or "text" not in parts[0]:
        raise ValueError("Gemini API yanıtında metin alanı bulunamadı.")

    return parts[0]["text"]


def analyze_disclosures_with_llm(
    ticker: str,
    disclosures: Sequence[KAPDisclosure],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
    custom_caller: Optional[Callable[[str], str]] = None,
) -> LLMAnalysisResult:
    """KAP bildirimlerini LLM ile analiz eder, hata veya anahtar yokluğunda yedek moda geçer."""
    norm_ticker = ticker.replace(".IS", "")
    if not disclosures:
        return fallback_rule_based_analysis(norm_ticker, [])

    prompt = build_prompt(norm_ticker, disclosures)

    # 1. Özel çağırıcı (Test mock veya özel motor)
    if custom_caller is not None:
        try:
            raw_output = custom_caller(prompt)
            return parse_llm_json_response(raw_output, norm_ticker)
        except Exception as exc:
            return fallback_rule_based_analysis(
                norm_ticker,
                disclosures,
                error_message=f"Özel LLM çağrısı başarısız: {exc}",
            )

    # 2. Standart Gemini API
    active_key = api_key or get_gemini_api_key()
    if not active_key:
        return fallback_rule_based_analysis(
            norm_ticker,
            disclosures,
            error_message="GEMINI_API_KEY tanımlanmadığı için kural tabanlı analiz uygulandı.",
        )

    try:
        raw_output = call_gemini_api(prompt, active_key, model=model, timeout=timeout)
        return parse_llm_json_response(raw_output, norm_ticker)
    except Exception as exc:
        return fallback_rule_based_analysis(
            norm_ticker,
            disclosures,
            error_message=(
                f"Gemini API erişiminde hata oluştu ({exc}); kural tabanlı analize geçildi."
            ),
        )
