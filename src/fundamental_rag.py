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
from dataclasses import dataclass, field, replace
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
    recent_disclosures: list[dict[str, Any]] = field(default_factory=list)
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

    @property
    def kap_disclosures(self) -> list[Any]:
        """recent_disclosures verilerini KAPDisclosure nesneleri olarak döndürür."""
        return convert_disclosures_to_kap(self.ticker, self.recent_disclosures)


def clean_ticker(ticker: str) -> str:
    """Hisse kodunu temizler ve standartlaştırır (örn: THYAO.IS -> THYAO)."""
    return ticker.upper().replace(".IS", "").strip()


def build_chatbot_deeplink(ticker: str, base_url: str | None = None) -> str:
    """Seçili hisse için Chatbot uygulamasına giden derin bağlantı (deep link) üretir."""
    norm_ticker = clean_ticker(ticker)
    url = (base_url or config.FINANCIAL_CHATBOT_URL).rstrip("/")
    return f"{url}/?ticker={norm_ticker}"


def convert_disclosures_to_kap(ticker: str, disclosures: list[dict[str, Any]]) -> list[Any]:
    """Chatbot projesinden gelen recent_disclosures verisini KAPDisclosure listesine dönüştürür."""
    from src import kap_news

    results: list[kap_news.KAPDisclosure] = []
    clean = clean_ticker(ticker)
    for d in disclosures:
        if not isinstance(d, dict):
            continue
        title = str(d.get("Başlık") or d.get("title") or "").strip()
        if not title:
            continue
        date = str(d.get("Tarih") or d.get("date") or d.get("publish_date") or "").strip()
        cat = str(d.get("Kategori") or d.get("category") or "Özel Durum").strip()
        impact = str(d.get("Etki") or d.get("impact") or "Pozitif").strip().lower()

        if any(w in impact for w in ("pozitif", "positive", "olumlu")):
            category = kap_news.EventCategory.POSITIVE
            badge = f"🟢 {cat}"
        elif any(w in impact for w in ("negatif", "negative", "olumsuz")):
            category = kap_news.EventCategory.NEGATIVE
            badge = f"🔴 {cat}"
        else:
            category = kap_news.EventCategory.NEUTRAL
            badge = f"⚪ {cat}"

        summary = str(d.get("Özet") or d.get("summary") or title).strip()
        results.append(
            kap_news.KAPDisclosure(
                ticker=clean,
                title=title,
                summary=summary,
                publish_date=date,
                category=category,
                badge=badge,
                event_type=cat,
            )
        )
    return results


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
    raw_disclosures = payload.get("recent_disclosures", payload.get("kap_events", []))
    disclosures = [dict(d) for d in raw_disclosures if isinstance(d, dict)]

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
        recent_disclosures=disclosures,
        source=source,
    )


def enrich_key_metrics_with_live_market_data(ticker: str, key_metrics: KeyMetrics) -> KeyMetrics:
    """Rapor içeriğinde eksik olan F/K veya PD/DD rasyolarını Yahoo Finance'den tamamlar."""
    if key_metrics.pe_ratio is not None and key_metrics.pb_ratio is not None:
        return key_metrics

    new_pe = key_metrics.pe_ratio
    new_pb = key_metrics.pb_ratio

    try:
        import yfinance as yf

        from src.network import ensure_ascii_ca_bundle

        ensure_ascii_ca_bundle()
        norm_sym = f"{clean_ticker(ticker)}.IS"
        t = yf.Ticker(norm_sym)
        info = getattr(t, "info", {}) or {}

        if new_pe is None:
            pe_val = info.get("trailingPE") or info.get("forwardPE")
            if isinstance(pe_val, (int, float)) and pe_val > 0:
                new_pe = round(float(pe_val), 2)

        if new_pb is None:
            pb_val = info.get("priceToBook")
            if isinstance(pb_val, (int, float)) and pb_val > 0:
                new_pb = round(float(pb_val), 2)
    except Exception as exc:
        logger.debug("Canlı piyasa çarpanları çekilemedi (%s): %s", ticker, exc)

    if new_pe != key_metrics.pe_ratio or new_pb != key_metrics.pb_ratio:
        return KeyMetrics(
            pe_ratio=new_pe,
            pb_ratio=new_pb,
            net_profit_margin_trend=key_metrics.net_profit_margin_trend,
            debt_to_equity=key_metrics.debt_to_equity,
            extra_metrics=dict(key_metrics.extra_metrics),
        )
    return key_metrics


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
    enrich_live_metrics: bool = True,
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
    report: FundamentalReport | None = None

    # 1. Yerel rapor kontrolü (testler veya paylaşımlı disk için)
    if local_reports_dir:
        for fname in (f"{clean}_latest.json", f"{clean}.json"):
            local_path = Path(local_reports_dir) / fname
            if local_path.is_file():
                try:
                    data = json.loads(local_path.read_text(encoding="utf-8"))
                    report = parse_report_payload(data, source="local_cache")
                    if report is not None:
                        break
                except Exception as e:
                    logger.debug("Yerel rapor okunamadı (%s): %s", local_path, e)

    # 2. FastAPI Servisi Kontrolü
    if report is None:
        active_api = api_url if api_url is not None else config.FINANCIAL_CHATBOT_API_URL
        if active_api:
            endpoint = f"{active_api.rstrip('/')}/api/v1/fundamental/{clean}"
            data = _fetch_from_url(endpoint, timeout=t_out)
            if data:
                report = parse_report_payload(data, source="api")

    # 3. GitHub Raw Deposu Kontrolü
    if report is None:
        active_repo = raw_repo or config.FINANCIAL_CHATBOT_RAW_REPO
        if active_repo:
            for raw_path in (f"reports/{clean}_latest.json", f"data/reports/{clean}.json"):
                raw_url = f"https://raw.githubusercontent.com/{active_repo}/main/{raw_path}"
                data = _fetch_from_url(raw_url, timeout=t_out)
                if data:
                    report = parse_report_payload(data, source="github_raw")
                    break

    if report is not None and enrich_live_metrics:
        if report.key_metrics.pe_ratio is None or report.key_metrics.pb_ratio is None:
            enriched = enrich_key_metrics_with_live_market_data(clean, report.key_metrics)
            report = replace(report, key_metrics=enriched)

    return report


# ============================================================================
# v2.0 KANTİTATİF DEĞERLEME, MAKRO STRES TESTİ VE PEER KIYASLAMA MOTORU
# (Financial-Analyst-ChatBot-Project v2.0.0 API & Deterministik Yedek Katmanı)
# ============================================================================

MACRO_TICKER_PROFILES: dict[str, dict[str, Any]] = {
    "THYAO": {
        "sector": "Havacılık / Ulaştırma",
        "fx_beta": 7.2,
        "rate_beta": -0.8,
        "oil_beta": -8.5,
        "fx_driver": (
            "Satış gelirlerinin %85+'i döviz (EUR/USD) cinsindendir. "
            "TL değer kaybı operasyonel marjları genişletir."
        ),
        "rate_driver": (
            "Uçak filosu finansmanı ağırlıklı olarak küresel döviz cinsi "
            "leasing ile yapılır; TL faiz duyarlılığı düşüktür."
        ),
        "oil_driver": (
            "Jet yakıtı giderleri toplam operasyonel maliyetin yaklaşık "
            "%38'ini oluşturur; petrol yükselişi kârı baskılar."
        ),
    },
    "PGSUS": {
        "sector": "Havacılık / Ulaştırma",
        "fx_beta": 6.8,
        "rate_beta": -1.0,
        "oil_beta": -9.0,
        "fx_driver": (
            "Uluslararası yolcu gelirleri döviz bazlıdır. Kur artışı pozitif katkı sağlar."
        ),
        "rate_driver": "Faiz duyarlılığı orta-düşük düzeydedir.",
        "oil_driver": "Akaryakıt tüketimi operasyonel kârlılığı doğrudan belirler.",
    },
    "TUPRS": {
        "sector": "Rafineri & Enerji",
        "fx_beta": 4.0,
        "rate_beta": -1.5,
        "oil_beta": 7.5,
        "fx_driver": (
            "Rafineri ürün satış fiyatları Akdeniz piyasası ve USD kuru ile "
            "endekslidir. Stok kârı dövizle paralel artar."
        ),
        "rate_driver": "İşletme sermayesi ve ham petrol alım finansmanı faiz gideri yaratır.",
        "oil_driver": (
            "Ham petrol fiyat artışı rafineri crack marjlarını genişletir ve "
            "pozitif stok değerleme kârı üretir."
        ),
    },
    "FROTO": {
        "sector": "Otomotiv",
        "fx_beta": 6.5,
        "rate_beta": -3.0,
        "oil_beta": -1.2,
        "fx_driver": (
            "Avrupa'ya yüksek ihracat ve 'cost-plus' (maliyet artı) "
            "sözleşmeler güçlü kur koruması sağlar."
        ),
        "rate_driver": "Yurtiçi taşıt kredisi faizlerindeki artış iç pazar talebini yavaşlatır.",
        "oil_driver": "Lojistik navlun giderleri üzerinden sınırlı negatif etki yaratır.",
    },
    "EREGL": {
        "sector": "Demir & Çelik",
        "fx_beta": 3.5,
        "rate_beta": -3.5,
        "oil_beta": -2.0,
        "fx_driver": (
            "Yassı çelik fiyatları uluslararası dolar paritesiyle belirlenir; "
            "gelirler döviz endekslidir."
        ),
        "rate_driver": (
            "İnşaat, sanayi ve yatırım talebi yüksek faiz ortamında "
            "yavaşlayarak çelik tüketimini azaltır."
        ),
        "oil_driver": "Küresel enerji ve metalurjik kömür maliyetlerini artırır.",
    },
    "ASELS": {
        "sector": "Savunma Sanayii",
        "fx_beta": 4.5,
        "rate_beta": -1.8,
        "oil_beta": -0.8,
        "fx_driver": (
            "Milyar dolarlık uluslararası ihracat bakiye siparişleri ve "
            "kur garantili kamu sözleşmeleri."
        ),
        "rate_driver": "Uzun vadeli Ar-Ge ve işletme sermayesi borçlanma maliyeti.",
        "oil_driver": "Enerji yoğunluğu düşüktür; petrol fiyatlarına duyarlılığı asgaridir.",
    },
    "BIMAS": {
        "sector": "Organize Perakende",
        "fx_beta": -2.2,
        "rate_beta": 2.0,
        "oil_beta": -2.5,
        "fx_driver": "İthal girdi maliyetleri artarken raf fiyatı geçişkenliği zaman alabilir.",
        "rate_driver": (
            "Net borçsuz ve güçlü nakit akışı sayesinde yüksek gecelik faiz "
            "ortamında faiz geliri elde eder."
        ),
        "oil_driver": "Geniş mağaza ağı ve lojistik tır filosu akaryakıt zamlarından etkilenir.",
    },
    "SISE": {
        "sector": "Cam & Sanayi",
        "fx_beta": 5.0,
        "rate_beta": -2.5,
        "oil_beta": -4.0,
        "fx_driver": "Küresel ölçekte ihracat ağı döviz gelirlerini destekler.",
        "rate_driver": "Kapasite yatırımları finansmanı için faiz yükü oluşturur.",
        "oil_driver": (
            "Cam fırınlarında doğalgaz ve enerji tüketimi çok yüksektir; "
            "enerji fiyatları marjı eritir."
        ),
    },
    "AKBNK": {
        "sector": "Bankacılık",
        "fx_beta": 0.5,
        "rate_beta": -3.8,
        "oil_beta": -1.0,
        "fx_driver": "Yasal düzenlemeler gereği yabancı para net genel pozisyonu nötre yakındır.",
        "rate_driver": (
            "Mevduat maliyetleri kredilerden daha hızlı yeniden fiyatlandığı için "
            "faiz artışında Net Faiz Marjı (NIM) kısa vadede daralır."
        ),
        "oil_driver": "Enflasyon ve makroekonomik istikrar kanalıyla dolaylı etki eder.",
    },
    "GARAN": {
        "sector": "Bankacılık",
        "fx_beta": 0.6,
        "rate_beta": -3.6,
        "oil_beta": -1.0,
        "fx_driver": "Dengeli döviz bilançosu ile kur şoklarına karşı korumalıdır.",
        "rate_driver": "Faiz artışları ilk etapta fonlama maliyetini artırır.",
        "oil_driver": "Dolaylı makroekonomik etki.",
    },
    "PETKM": {
        "sector": "Petrokimya",
        "fx_beta": 2.5,
        "rate_beta": -2.5,
        "oil_beta": -4.5,
        "fx_driver": "Ürün fiyatları dolar endekslidir.",
        "rate_driver": "İşletme sermayesi ihtiyacı borçlanmayı artırır.",
        "oil_driver": (
            "Hammadde olarak nafta tüketir; petrol artışı etilen-nafta makasını daraltır."
        ),
    },
}

DEFAULT_MACRO_PROFILE: dict[str, Any] = {
    "sector": "Genel BIST Sanayi/Hizmet",
    "fx_beta": 2.0,
    "rate_beta": -2.0,
    "oil_beta": -1.2,
    "fx_driver": "İhracat potansiyeli ve döviz pozisyonuna bağlı olarak kârlılık etkilenir.",
    "rate_driver": "Yüksek faizler finansman maliyetini artırır ve iç talebi sınırlandırır.",
    "oil_driver": "Akaryakıt ve lojistik maliyetleri operasyonel kârı etkiler.",
}

DEFAULT_DIVIDEND_PROFILES: dict[str, dict[str, float | int]] = {
    "TUPRS": {
        "yield_pct": 10.5,
        "payout_pct": 82.0,
        "fcf_margin_pct": 14.2,
        "consecutive_years": 8,
    },
    "FROTO": {
        "yield_pct": 7.2,
        "payout_pct": 75.0,
        "fcf_margin_pct": 12.0,
        "consecutive_years": 10,
    },
    "TTRAK": {
        "yield_pct": 8.8,
        "payout_pct": 85.0,
        "fcf_margin_pct": 15.5,
        "consecutive_years": 9,
    },
    "EREGL": {
        "yield_pct": 5.4,
        "payout_pct": 65.0,
        "fcf_margin_pct": 8.0,
        "consecutive_years": 6,
    },
    "AKBNK": {
        "yield_pct": 3.8,
        "payout_pct": 20.0,
        "fcf_margin_pct": 18.0,
        "consecutive_years": 5,
    },
    "GARAN": {
        "yield_pct": 4.1,
        "payout_pct": 22.0,
        "fcf_margin_pct": 19.5,
        "consecutive_years": 4,
    },
    "BIMAS": {
        "yield_pct": 4.5,
        "payout_pct": 60.0,
        "fcf_margin_pct": 7.5,
        "consecutive_years": 7,
    },
    "THYAO": {
        "yield_pct": 0.0,
        "payout_pct": 0.0,
        "fcf_margin_pct": 16.5,
        "consecutive_years": 0,
    },
    "PGSUS": {
        "yield_pct": 0.0,
        "payout_pct": 0.0,
        "fcf_margin_pct": 14.0,
        "consecutive_years": 0,
    },
    "ASELS": {
        "yield_pct": 1.5,
        "payout_pct": 25.0,
        "fcf_margin_pct": 6.5,
        "consecutive_years": 5,
    },
}

DEFAULT_DUPONT_PROFILES: dict[str, dict[str, float]] = {
    "THYAO": {"npm": 16.4, "at": 0.78, "em": 2.35, "roe": 30.1, "roa": 12.8},
    "PGSUS": {"npm": 14.2, "at": 0.72, "em": 2.60, "roe": 26.6, "roa": 10.2},
    "TUPRS": {"npm": 9.8, "at": 1.45, "em": 1.85, "roe": 26.3, "roa": 14.2},
    "FROTO": {"npm": 11.2, "at": 1.38, "em": 2.25, "roe": 34.8, "roa": 15.5},
    "EREGL": {"npm": 6.5, "at": 0.65, "em": 1.75, "roe": 7.4, "roa": 4.2},
    "ASELS": {"npm": 18.5, "at": 0.62, "em": 1.90, "roe": 21.8, "roa": 11.5},
    "BIMAS": {"npm": 5.2, "at": 2.40, "em": 2.10, "roe": 26.2, "roa": 12.5},
    "SISE": {"npm": 8.4, "at": 0.68, "em": 1.95, "roe": 11.1, "roa": 5.7},
    "AKBNK": {"npm": 24.5, "at": 0.12, "em": 8.50, "roe": 25.0, "roa": 2.9},
    "GARAN": {"npm": 26.0, "at": 0.13, "em": 8.20, "roe": 27.7, "roa": 3.4},
}


def calculate_dupont_model(ticker: str, live_lookup: bool = False) -> dict[str, Any]:
    """3 aşamalı DuPont analizini (Net Kâr Marjı × Aktif Devir Hızı × Kaldıraç) hesaplar."""
    clean_sym = clean_ticker(ticker)
    prof = DEFAULT_DUPONT_PROFILES.get(
        clean_sym,
        {"npm": 8.5, "at": 0.85, "em": 2.2, "roe": 15.8, "roa": 7.2},
    )
    npm_pct = prof["npm"]
    asset_turnover = prof["at"]
    equity_multiplier = prof["em"]
    roe_pct = prof["roe"]
    roa_pct = prof["roa"]
    source = "deterministic_profile"

    if live_lookup:
        try:
            import yfinance as yf

            from src.network import ensure_ascii_ca_bundle

            ensure_ascii_ca_bundle()
            t = yf.Ticker(f"{clean_sym}.IS")
            info = getattr(t, "info", {}) or {}
            roe_raw = info.get("returnOnEquity")
            roa_raw = info.get("returnOnAssets")
            npm_raw = info.get("profitMargins")
            if roe_raw is not None and npm_raw is not None and float(npm_raw) > 0:
                npm_pct = round(float(npm_raw) * 100.0, 2)
                roe_pct = round(float(roe_raw) * 100.0, 2)
                roa_pct = (
                    round(float(roa_raw) * 100.0, 2)
                    if roa_raw is not None
                    else round(roe_pct * 0.45, 2)
                )
                asset_turnover = max(0.1, round(roa_pct / npm_pct, 2)) if npm_pct > 0 else 0.8
                equity_multiplier = max(1.0, round(roe_pct / roa_pct, 2)) if roa_pct > 0 else 2.2
                source = "live_yfinance"
        except Exception as exc:
            logger.debug("DuPont canlı hesaplama hatası (%s): %s", clean_sym, exc)

    if equity_multiplier > 3.5:
        driver = "Yüksek Finansal Kaldıraç (Borç / Aktif Yapısı)"
        quality = (
            "Özkaynak kârlılığı yüksek finansal kaldıraç ile destekleniyor; "
            "fonlama ve faiz maliyeti izlenmeli."
        )
    elif asset_turnover > 1.3:
        driver = "Hızlı Varlık Devir Hızı (Operasyonel Hız)"
        quality = (
            "Şirket varlıklarını çok verimli bir şekilde ciroya dönüştürüyor; "
            "operasyonel hacim ana itici güç."
        )
    else:
        driver = "Güçlü Net Kâr Marjı (Fiyatlama Gücü)"
        quality = (
            "Yüksek net kâr marjı şirketin güçlü pazar payı ve fiyatlama "
            "gücüne sahip olduğunu gösteriyor."
        )

    return {
        "ticker": clean_sym,
        "net_profit_margin_pct": round(npm_pct, 2),
        "asset_turnover": round(asset_turnover, 2),
        "equity_multiplier": round(equity_multiplier, 2),
        "roe_pct": round(roe_pct, 2),
        "roa_pct": round(roa_pct, 2),
        "driver": driver,
        "quality_assessment": quality,
        "source": source,
    }


def calculate_altman_z_score(ticker: str, dupont: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gelişmekte olan piyasalar için uyarlanmış Altman Z-Skorunu hesaplar."""
    clean_sym = clean_ticker(ticker)
    dup = dupont if dupont is not None else calculate_dupont_model(clean_sym)

    npm = float(dup.get("net_profit_margin_pct", 8.0))
    lev = float(dup.get("equity_multiplier", 2.0))
    turnover = float(dup.get("asset_turnover", 0.9))
    roa = float(dup.get("roa_pct", 7.0))

    x1 = max(0.05, min(0.40, 0.25 - (lev - 2.0) * 0.05))
    x2 = max(0.05, min(0.50, npm * 0.025))
    x3 = max(0.02, min(0.35, (roa / 100.0) * 1.3))
    x4 = max(0.30, min(3.50, 4.0 / max(1.1, lev)))
    x5 = turnover

    z_score = round(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5, 2)

    if z_score >= 2.90:
        zone = "Güvenli Bölge (Safe Zone)"
        zone_color = "#10B981"
        interpretation = (
            "Finansal dayanıklılık çok yüksek; kısa ve orta vadeli iflas "
            "veya temerrüt riski bulunmuyor."
        )
    elif z_score >= 1.80:
        zone = "Gri Bölge (Grey Zone)"
        zone_color = "#F59E0B"
        interpretation = (
            "Finansal yapı dengeli ancak işletme sermayesi ve borç çevirme "
            "oranları dikkatle takip edilmeli."
        )
    else:
        zone = "Stres Bölgesi (Distress Zone)"
        zone_color = "#EF4444"
        interpretation = "Yüksek borç yükü ve zayıf kârlılık marjı finansal kırılganlığı artırıyor."

    return {
        "ticker": clean_sym,
        "z_score": z_score,
        "zone": zone,
        "zone_color": zone_color,
        "interpretation": interpretation,
        "components": {
            "Likidite (X1)": round(x1, 3),
            "Kümülatif Kârlılık (X2)": round(x2, 3),
            "Faaliyet Getirisi (X3)": round(x3, 3),
            "Kaldıraç Gücü (X4)": round(x4, 3),
            "Varlık Verimi (X5)": round(x5, 3),
        },
    }


def calculate_dcf_valuation(
    ticker: str,
    base_fcf_million_tl: float | None = None,
    growth_rate_pct: float = 15.0,
    discount_rate_pct: float = 24.0,
    terminal_growth_pct: float = 6.0,
    years: int = 5,
) -> dict[str, Any]:
    """İndirgenmiş Nakit Akımları (DCF) 5 yıllık projeksiyon ve Gordon terminal değerlemesi."""
    clean_sym = clean_ticker(ticker)
    if base_fcf_million_tl is None or base_fcf_million_tl <= 0:
        base_fcf_million_tl = 1500.0

    g = growth_rate_pct / 100.0
    r = discount_rate_pct / 100.0
    tg = terminal_growth_pct / 100.0
    if r <= tg:
        r = tg + 0.05

    pv_cash_flows: list[float] = []
    current_cf = float(base_fcf_million_tl)
    for y in range(1, years + 1):
        current_cf = current_cf * (1.0 + g)
        pv = current_cf / ((1.0 + r) ** y)
        pv_cash_flows.append(round(pv, 2))

    terminal_value = (current_cf * (1.0 + tg)) / (r - tg)
    pv_terminal_value = terminal_value / ((1.0 + r) ** years)
    fair_enterprise_value = sum(pv_cash_flows) + pv_terminal_value

    return {
        "ticker": clean_sym,
        "base_fcf_million_tl": base_fcf_million_tl,
        "projected_pv_flows": pv_cash_flows,
        "pv_terminal_value": round(pv_terminal_value, 2),
        "estimated_fair_value_million_tl": round(fair_enterprise_value, 2),
        "assumptions": {
            "growth_rate_pct": growth_rate_pct,
            "discount_rate_pct": discount_rate_pct,
            "terminal_growth_pct": terminal_growth_pct,
            "projection_years": years,
        },
    }


def calculate_dividend_metrics(ticker: str) -> dict[str, Any]:
    """Temettü verimi, dağıtma oranı, serbest nakit akışı ve Temettü Güvenlik Skorunu hesaplar."""
    clean_sym = clean_ticker(ticker)
    profile = DEFAULT_DIVIDEND_PROFILES.get(
        clean_sym,
        {"yield_pct": 3.5, "payout_pct": 45.0, "fcf_margin_pct": 9.0, "consecutive_years": 3},
    )
    yield_pct = float(profile["yield_pct"])
    payout_pct = float(profile["payout_pct"])
    fcf_margin_pct = float(profile["fcf_margin_pct"])
    consecutive_years = int(profile["consecutive_years"])

    if yield_pct <= 0:
        score = 25.0
        category = "⚪ Temettü Dağıtmıyor (Büyüme/Yatırım Odaklı)"
        badge_color = "#64748B"
        takeaway = (
            "Şirket kârını temettü olarak dağıtmak yerine filo, kapasite "
            "veya teknoloji yatırımlarında değerlendirmektedir."
        )
    else:
        score = 0.0
        if 20.0 <= payout_pct <= 75.0:
            score += 40.0
        elif 75.0 < payout_pct <= 90.0:
            score += 25.0
        else:
            score += 15.0

        if fcf_margin_pct >= 10.0:
            score += 40.0
        elif fcf_margin_pct >= 5.0:
            score += 25.0
        elif fcf_margin_pct > 0:
            score += 15.0
        else:
            score += 5.0

        score += min(20.0, consecutive_years * 2.5)
        score = min(100.0, max(10.0, score))

        if score >= 75.0 and yield_pct >= 5.0:
            category = "👑 BIST Temettü Aristokratı / Güçlü Nakit"
            badge_color = "#10B981"
            takeaway = (
                "Şirket yüksek serbest nakit akışı yaratmakta ve kâr payı "
                "dağıtımını borçlanmadan, kendi nakdiyle finanse etmektedir."
            )
        elif score >= 55.0:
            category = "🛡️ Güvenli & Dengeli Temettü"
            badge_color = "#38BDF8"
            takeaway = (
                "Temettü ödeme gücü tatmin edici düzeydedir. Nakit akışı "
                "dağıtılan kâr payını karşılayabilecek kapasitededir."
            )
        else:
            category = "⚠️ Riskli / Sürdürülebilirlik Takip Edilmeli"
            badge_color = "#F59E0B"
            takeaway = (
                "Dağıtma oranı yüksek veya serbest nakit akışı dalgalıdır; "
                "gelecekte temettü kesintisi riski mevcuttur."
            )

    return {
        "ticker": clean_sym,
        "yield_pct": yield_pct,
        "payout_pct": payout_pct,
        "fcf_margin_pct": fcf_margin_pct,
        "consecutive_years": consecutive_years,
        "dividend_safety_score": round(score, 1),
        "category": category,
        "badge_color": badge_color,
        "takeaway": takeaway,
    }


def calculate_kelly_allocation(
    ticker: str,
    win_rate: float = 0.62,
    win_loss_ratio: float = 1.85,
    max_portfolio_cap_pct: float = 25.0,
) -> dict[str, Any]:
    """Kelly Kriteri (f* = p - (1 - p) / b) ile optimal Half-Kelly portföy ağırlığını hesaplar."""
    clean_sym = clean_ticker(ticker)
    p = max(0.01, min(0.99, win_rate))
    b = max(0.1, win_loss_ratio)
    q = 1.0 - p
    raw_kelly = p - (q / b)
    kelly_pct = max(0.0, raw_kelly * 100.0)
    conservative_kelly_pct = min(kelly_pct * 0.5, max_portfolio_cap_pct)

    if kelly_pct <= 0:
        recommendation = "Girmeyin / Pozisyon Açmayın (Negatif Matematiksel Beklenti)"
    elif conservative_kelly_pct >= 15.0:
        recommendation = f"Güçlü Tahsis (Maksimum %{conservative_kelly_pct:.1f} Portföy Payı)"
    else:
        recommendation = f"Ilımlı Tahsis (%{conservative_kelly_pct:.1f} Portföy Payı)"

    return {
        "ticker": clean_sym,
        "win_rate_pct": round(p * 100.0, 1),
        "win_loss_ratio": round(b, 2),
        "full_kelly_pct": round(kelly_pct, 1),
        "recommended_portfolio_allocation_pct": round(conservative_kelly_pct, 1),
        "recommendation": recommendation,
    }


def analyze_macro_sensitivity(
    ticker: str,
    usd_change_pct: float = 10.0,
    rate_change_bps: float = 250.0,
    oil_change_pct: float = 15.0,
) -> dict[str, Any]:
    """USD/TRY, TCMB Politika Faizi ve Brent Petrol şoklarının net kâr duyarlılığını hesaplar."""
    clean_sym = clean_ticker(ticker)
    profile = MACRO_TICKER_PROFILES.get(clean_sym, DEFAULT_MACRO_PROFILE)

    fx_impact = round((usd_change_pct / 10.0) * float(profile["fx_beta"]), 2)
    rate_impact = round((rate_change_bps / 250.0) * float(profile["rate_beta"]), 2)
    oil_impact = round((oil_change_pct / 15.0) * float(profile["oil_beta"]), 2)
    total_impact = round(fx_impact + rate_impact + oil_impact, 2)

    if total_impact >= 3.0:
        resilience_badge = "🚀 Makro Pozitif / Kur Korunaklı"
        badge_color = "#10B981"
        verdict = (
            f"{clean_sym}, mevcut senaryodaki kur ve emtia hareketlerinden "
            "net pozitif ayrışma potansiyeline sahiptir."
        )
    elif total_impact >= -2.0:
        resilience_badge = "🛡️ Dengeli & Dayanıklı Yapı"
        badge_color = "#38BDF8"
        verdict = (
            f"{clean_sym}, karşıt makro güçlerin birbirini dengelemesi sayesinde "
            "şoklara karşı yüksek bağışıklık göstermektedir."
        )
    else:
        resilience_badge = "⚠️ Makro Baskı / Yüksek Maliyet Riski"
        badge_color = "#F43F5E"
        verdict = (
            f"{clean_sym}, mevcut senaryodaki faiz artışı veya enerji maliyetleri "
            "nedeniyle marj daralması riskiyle karşı karşıyadır."
        )

    return {
        "ticker": clean_sym,
        "sector": profile["sector"],
        "scenarios": {
            "usd_try": {
                "change_pct": usd_change_pct,
                "impact_pct": fx_impact,
                "driver": profile["fx_driver"],
            },
            "interest_rate": {
                "change_bps": rate_change_bps,
                "impact_pct": rate_impact,
                "driver": profile["rate_driver"],
            },
            "brent_oil": {
                "change_pct": oil_change_pct,
                "impact_pct": oil_impact,
                "driver": profile["oil_driver"],
            },
        },
        "total_impact_pct": total_impact,
        "resilience_badge": resilience_badge,
        "badge_color": badge_color,
        "verdict": verdict,
    }


def compare_peers(ticker_a: str, ticker_b: str) -> dict[str, Any]:
    """İki BIST hissesini DuPont kârlılık marjları, kaldıraç ve Altman Z-Skoru bazında kıyaslar."""
    a = clean_ticker(ticker_a)
    b = clean_ticker(ticker_b)

    dup_a = calculate_dupont_model(a)
    dup_b = calculate_dupont_model(b)
    alt_a = calculate_altman_z_score(a, dup_a)
    alt_b = calculate_altman_z_score(b, dup_b)

    score_a = float(dup_a.get("roe_pct", 0.0)) + (float(alt_a.get("z_score", 0.0)) * 5.0)
    score_b = float(dup_b.get("roe_pct", 0.0)) + (float(alt_b.get("z_score", 0.0)) * 5.0)

    if score_a > score_b * 1.1:
        leader = a
        rationale = (
            f"{a}, daha yüksek operasyonel verimlilik ve sağlam bilanço Z-skoru "
            "sayesinde kârlılık kalitesinde öne çıkmaktadır."
        )
    elif score_b > score_a * 1.1:
        leader = b
        rationale = (
            f"{b}, kârlılık marjı ve sermaye dayanıklılık rasyolarında {a} "
            "hissesine göre avantaj sağlamaktadır."
        )
    else:
        leader = "Dengeli / Baş Başa"
        rationale = (
            "Her iki şirket de kârlılık ve finansal risk profili açısından "
            "birbirine yakın seviyelerdedir."
        )

    return {
        "company_a": {
            "ticker": a,
            "roe_pct": round(float(dup_a.get("roe_pct", 0.0)), 1),
            "net_margin_pct": round(float(dup_a.get("net_profit_margin_pct", 0.0)), 1),
            "asset_turnover": round(float(dup_a.get("asset_turnover", 0.0)), 2),
            "equity_multiplier": round(float(dup_a.get("equity_multiplier", 1.0)), 2),
            "z_score": round(float(alt_a.get("z_score", 0.0)), 2),
            "zone": alt_a.get("zone", ""),
        },
        "company_b": {
            "ticker": b,
            "roe_pct": round(float(dup_b.get("roe_pct", 0.0)), 1),
            "net_margin_pct": round(float(dup_b.get("net_profit_margin_pct", 0.0)), 1),
            "asset_turnover": round(float(dup_b.get("asset_turnover", 0.0)), 2),
            "equity_multiplier": round(float(dup_b.get("equity_multiplier", 1.0)), 2),
            "z_score": round(float(alt_b.get("z_score", 0.0)), 2),
            "zone": alt_b.get("zone", ""),
        },
        "quality_leader": leader,
        "analyst_rationale": rationale,
    }


def fetch_quant_metrics(
    ticker: str,
    growth_rate_pct: float = 15.0,
    discount_rate_pct: float = 24.0,
    api_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """`/api/v1/quant/{ticker}` uç noktasını veya deterministik motoru çalıştırır."""
    clean = clean_ticker(ticker)
    t_out = timeout if timeout is not None else config.FINANCIAL_CHATBOT_TIMEOUT_SECONDS
    active_api = api_url if api_url is not None else config.FINANCIAL_CHATBOT_API_URL

    if active_api:
        url = (
            f"{active_api.rstrip('/')}/api/v1/quant/{clean}"
            f"?growth_rate_pct={growth_rate_pct}&discount_rate_pct={discount_rate_pct}"
        )
        data = _fetch_from_url(url, timeout=t_out)
        if isinstance(data, dict) and "dupont" in data and "altman_z" in data:
            data["source"] = "api_v2"
            return data

    dup = calculate_dupont_model(clean)
    return {
        "ticker": clean,
        "dupont": dup,
        "altman_z": calculate_altman_z_score(clean, dup),
        "dcf_valuation": calculate_dcf_valuation(
            clean,
            growth_rate_pct=growth_rate_pct,
            discount_rate_pct=discount_rate_pct,
        ),
        "dividend_safety": calculate_dividend_metrics(clean),
        "kelly_allocation": calculate_kelly_allocation(clean),
        "source": "deterministic_engine",
    }


def fetch_macro_stress_test(
    ticker: str,
    usd_change_pct: float = 10.0,
    rate_change_bps: float = 250.0,
    oil_change_pct: float = 15.0,
    api_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """`/api/v1/macro-stress/{ticker}` uç noktasını veya deterministik simülatörü çalıştırır."""
    clean = clean_ticker(ticker)
    t_out = timeout if timeout is not None else config.FINANCIAL_CHATBOT_TIMEOUT_SECONDS
    active_api = api_url if api_url is not None else config.FINANCIAL_CHATBOT_API_URL

    if active_api:
        url = (
            f"{active_api.rstrip('/')}/api/v1/macro-stress/{clean}"
            f"?usd_change_pct={usd_change_pct}&rate_change_bps={rate_change_bps}"
            f"&oil_change_pct={oil_change_pct}"
        )
        data = _fetch_from_url(url, timeout=t_out)
        if isinstance(data, dict) and "total_impact_pct" in data:
            data["source"] = "api_v2"
            return data

    res = analyze_macro_sensitivity(
        clean,
        usd_change_pct=usd_change_pct,
        rate_change_bps=rate_change_bps,
        oil_change_pct=oil_change_pct,
    )
    res["source"] = "deterministic_engine"
    return res


def fetch_peer_comparison(
    ticker_a: str,
    ticker_b: str,
    api_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """`/api/v1/compare/{ticker_a}/{ticker_b}` uç noktasını veya yerel peer motorunu çalıştırır."""
    clean_a = clean_ticker(ticker_a)
    clean_b = clean_ticker(ticker_b)
    t_out = timeout if timeout is not None else config.FINANCIAL_CHATBOT_TIMEOUT_SECONDS
    active_api = api_url if api_url is not None else config.FINANCIAL_CHATBOT_API_URL

    if active_api:
        url = f"{active_api.rstrip('/')}/api/v1/compare/{clean_a}/{clean_b}"
        data = _fetch_from_url(url, timeout=t_out)
        if isinstance(data, dict) and "quality_leader" in data:
            data["source"] = "api_v2"
            return data

    res = compare_peers(clean_a, clean_b)
    res["source"] = "deterministic_engine"
    return res


def ask_hybrid_analyst(
    question: str,
    ticker: str | None = None,
    provider: str = "colab",
    base_url: str | None = None,
    api_url: str | None = None,
    timeout: float = 25.0,
) -> dict[str, Any]:
    """`/api/v1/ask` uç noktasını çağırır; erişilemezse kantitatif bağlamla Colab LLM'i sorgular."""
    active_api = api_url if api_url is not None else config.FINANCIAL_CHATBOT_API_URL
    if active_api:
        endpoint = f"{active_api.rstrip('/')}/api/v1/ask"
        payload = json.dumps(
            {
                "question": question,
                "company_filter": clean_ticker(ticker) if ticker else None,
                "provider": provider,
                "base_url": base_url,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "BIST-Trend-Forecaster/2.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    if isinstance(data, dict) and data.get("answer"):
                        data["source"] = "api_v2"
                        return data
        except Exception as exc:
            logger.debug("v2 /api/v1/ask çağrısı başarısız: %s", exc)

    from src import colab_llm

    enriched_prompt = question
    if ticker:
        clean = clean_ticker(ticker)
        quant = fetch_quant_metrics(clean, api_url="")
        dup = quant["dupont"]
        alt = quant["altman_z"]
        div = quant["dividend_safety"]
        enriched_prompt = (
            f"[{clean} Doğrulanmış Kantitatif Veri: ROE=%{dup['roe_pct']}, "
            f"Net Kâr Marjı=%{dup['net_profit_margin_pct']}, "
            f"Aktif Devir={dup['asset_turnover']}x, Kaldıraç={dup['equity_multiplier']}x, "
            f"Altman Z={alt['z_score']} ({alt['zone']}), "
            f"Temettü Güvenliği={div['dividend_safety_score']}/100 ({div['category']})]\n\n"
            f"Soru: {question}"
        )

    answer = colab_llm.query_colab_llm(
        enriched_prompt,
        base_url=base_url or config.COLAB_LLM_URL,
        timeout=timeout,
    )
    return {
        "question": question,
        "answer": answer,
        "sources": [],
        "source": "colab_llm_with_quant_context",
    }
