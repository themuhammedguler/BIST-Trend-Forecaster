"""BIST Pay Piyasası Anlık (0 Gecikmeli) Canlı Kotasyon ve Fiyat Motoru.

Borsa İstanbul'daki hisselerin anlık son fiyatını, günlük yüzde değişimini,
hacmini ve gün içi en yüksek/en düşük seviyelerini sıfır gecikmeyle (real-time) çeker.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

SCANNER_URL = "https://scanner.tradingview.com/turkey/scan"


@dataclass(frozen=True)
class LiveQuote:
    """Tek bir hissenin anlık kotasyon verisi."""

    ticker: str
    last_price: float
    change_pct: float
    volume: float
    high: float
    low: float
    open_price: float
    technical_rating: float = 0.0

    @property
    def badge(self) -> str:
        """Yüzde değişim rozeti."""
        if self.change_pct > 0:
            return f"🟢 %{self.change_pct:+.2f}"
        if self.change_pct < 0:
            return f"🔴 %{self.change_pct:+.2f}"
        return f"⚪ %{self.change_pct:+.2f}"


def clean_ticker_name(symbol: str) -> str:
    """Sembol kodunu temizler (örn: THYAO.IS veya BIST:THYAO -> THYAO)."""
    return symbol.upper().replace(".IS", "").replace("BIST:", "").strip()


def build_scanner_payload(clean_tickers: list[str]) -> dict[str, Any]:
    """TradingView scanner API için POST yükü üretir."""
    return {
        "filter": [{"left": "name", "operation": "in_range", "right": clean_tickers}],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name", "close", "change", "volume", "high", "low", "open", "Recommend.All"],
        "sort": {"sortBy": "change", "sortOrder": "desc"},
        "range": [0, max(50, len(clean_tickers))],
    }


def fetch_live_quotes(
    tickers: list[str] | None = None,
    timeout: float = 6.0,
    custom_fetcher: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
) -> dict[str, LiveQuote]:
    """BIST hisselerinin anlık canlı fiyat verilerini sıfır gecikmeyle çeker."""
    from src import config

    symbols = tickers if tickers is not None else config.TICKERS
    clean_map = {clean_ticker_name(t): t for t in symbols}
    clean_list = list(clean_map.keys())

    if not clean_list:
        return {}

    payload = build_scanner_payload(clean_list)
    results: dict[str, LiveQuote] = {}

    try:
        if custom_fetcher is not None:
            raw_data = custom_fetcher(SCANNER_URL, payload, timeout)
        else:
            req = urllib.request.Request(
                SCANNER_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    raw_data = json.loads(response.read().decode("utf-8"))
                else:
                    raw_data = {}

        rows = raw_data.get("data", []) if isinstance(raw_data, dict) else []
        for item in rows:
            if not isinstance(item, dict):
                continue
            cols = item.get("d", [])
            if len(cols) < 7:
                continue
            name = str(cols[0]).upper()
            try:
                last_price = float(cols[1])
                change_pct = float(cols[2])
                volume = float(cols[3])
                high = float(cols[4])
                low = float(cols[5])
                open_price = float(cols[6])
                rec = float(cols[7]) if len(cols) > 7 and cols[7] is not None else 0.0
            except (ValueError, TypeError):
                continue

            quote = LiveQuote(
                ticker=name,
                last_price=last_price,
                change_pct=change_pct,
                volume=volume,
                high=high,
                low=low,
                open_price=open_price,
                technical_rating=rec,
            )
            results[name] = quote

    except Exception as exc:
        logger.debug("Canlı BIST kotasyonları çekilirken hata: %s", exc)

    return results
