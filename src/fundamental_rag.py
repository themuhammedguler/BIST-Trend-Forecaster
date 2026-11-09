"""Financial Analyst ChatBot Project ile RAG temel analiz entegrasyon istemcisi.

Bu modül, themuhammedguler/Financial-Analyst-ChatBot-Project tarafından üretilen
KAP, bilanço ve faaliyet raporu özetlerini (FundamentalReport) yapılandırılmış
JSON formatında çeker, önbelleğe alır ve arayüzde sunulmasını sağlar.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyMetrics:
    """Temel finansal rasyolar."""

    pe_ratio: float | None = None
    pb_ratio: float | None = None
    net_profit_margin_trend: str = "Yatay"
    debt_to_equity: str = "Yönetilebilir"
    extra_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FundamentalReport:
    """Chatbot projesinden alınan yapılandırılmış temel analiz raporu."""

    ticker: str
    company_name: str
    report_date: str
    fundamental_summary: str
    sentiment_score: float
    sentiment_label: str
    key_metrics: KeyMetrics
    catalysts: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    source: str = "unknown"
    error_message: str | None = None

    @property
    def badge(self) -> str:
        """Duygu skoruna ve etiketine göre görsel rozet döndürür."""
        score_str = f"{self.sentiment_score:+.2f}"
        if self.sentiment_label.lower() in ("pozitif", "positive"):
            return f"🟢 Pozitif ({score_str})"
        if self.sentiment_label.lower() in ("negatif", "negative"):
            return f"🔴 Negatif ({score_str})"
        return f"🟡 Nötr ({score_str})"


def clean_ticker(ticker: str) -> str:
    """Hisse kodunu temizler ve standartlaştırır (örn: THYAO.IS -> THYAO)."""
    return ticker.upper().replace(".IS", "").strip()


def build_chatbot_deeplink(ticker: str, base_url: str | None = None) -> str:
    """Seçili hisse için Chatbot uygulamasına giden derin bağlantı (deep link) üretir."""
    norm_ticker = clean_ticker(ticker)
    url = (base_url or config.FINANCIAL_CHATBOT_URL).rstrip("/")
    return f"{url}/?ticker={norm_ticker}"


def parse_report_payload(payload: dict[str, Any], source: str = "unknown") -> FundamentalReport:
    """Ham sözlük verisini FundamentalReport nesnesine dönüştürür."""
    ticker = clean_ticker(str(payload.get("ticker", "BIST")))
    company_name = str(payload.get("company_name", ticker))
    report_date = str(payload.get("report_date", ""))
    fundamental_summary = str(payload.get("fundamental_summary", ""))

    try:
        sentiment_score = float(payload.get("sentiment_score", 0.0))
    except (TypeError, ValueError):
        sentiment_score = 0.0

    sentiment_label = str(payload.get("sentiment_label", "Nötr")).capitalize()

    raw_metrics = payload.get("key_metrics", {})
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}

    def _to_float_or_none(val: Any) -> float | None:
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    key_metrics = KeyMetrics(
        pe_ratio=_to_float_or_none(raw_metrics.get("pe_ratio")),
        pb_ratio=_to_float_or_none(raw_metrics.get("pb_ratio")),
        net_profit_margin_trend=str(raw_metrics.get("net_profit_margin_trend", "Yatay")),
        debt_to_equity=str(raw_metrics.get("debt_to_equity", "Yönetilebilir")),
        extra_metrics=raw_metrics.get("extra_metrics", {})
        if isinstance(raw_metrics.get("extra_metrics"), dict)
        else {},
    )

    catalysts = [str(c) for c in payload.get("catalysts", []) if isinstance(c, (str, int, float))]
    risk_factors = [
        str(r) for r in payload.get("risk_factors", []) if isinstance(r, (str, int, float))
    ]

    return FundamentalReport(
        ticker=ticker,
        company_name=company_name,
        report_date=report_date,
        fundamental_summary=fundamental_summary,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        key_metrics=key_metrics,
        catalysts=catalysts,
        risk_factors=risk_factors,
        source=source,
    )


def _fetch_from_url(url: str, timeout: float) -> dict[str, Any] | None:
    """HTTP GET isteği gönderip JSON cevabını döndürür."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "BIST-Trend-Forecaster/1.0 (Integration Client)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                body = response.read().decode("utf-8")
                return json.loads(body)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        logger.debug("URL'den temel analiz çekilemedi (%s): %s", url, exc)
    return None


def fetch_fundamental_report(
    ticker: str,
    api_url: str | None = None,
    raw_repo: str | None = None,
    timeout: float | None = None,
    local_reports_dir: str | Path | None = None,
) -> FundamentalReport | None:
    """
    Belirtilen hisse için en güncel temel analiz raporunu getirir.

    Öncelik Sırası:
    1. Yerel rapor dizini (local_reports_dir)
    2. FastAPI servisi (FINANCIAL_CHATBOT_API_URL)
    3. GitHub Raw deposu (FINANCIAL_CHATBOT_RAW_REPO)
    """
    clean = clean_ticker(ticker)
    t_out = timeout if timeout is not None else config.FINANCIAL_CHATBOT_TIMEOUT_SECONDS

    # 1. Yerel rapor kontrolü (testler veya paylaşımlı disk için)
    if local_reports_dir:
        local_path = Path(local_reports_dir) / f"{clean}_latest.json"
        if local_path.is_file():
            try:
                data = json.loads(local_path.read_text(encoding="utf-8"))
                return parse_report_payload(data, source="local_cache")
            except Exception as e:
                logger.debug("Yerel rapor okunamadı (%s): %s", local_path, e)

    # 2. FastAPI Servisi Kontrolü
    active_api = api_url if api_url is not None else config.FINANCIAL_CHATBOT_API_URL
    if active_api:
        endpoint = f"{active_api.rstrip('/')}/api/v1/fundamental/{clean}"
        data = _fetch_from_url(endpoint, timeout=t_out)
        if data:
            return parse_report_payload(data, source="api")

    # 3. GitHub Raw Deposu Kontrolü
    active_repo = raw_repo or config.FINANCIAL_CHATBOT_RAW_REPO
    if active_repo:
        raw_url = (
            f"https://raw.githubusercontent.com/{active_repo}/main/reports/{clean}_latest.json"
        )
        data = _fetch_from_url(raw_url, timeout=t_out)
        if data:
            return parse_report_payload(data, source="github_raw")

    return None
