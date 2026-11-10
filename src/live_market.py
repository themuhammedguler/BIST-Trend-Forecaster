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


def is_live_frame_recent(full_df: Any, max_age_days: int = 10) -> bool:
    """Veri çerçevesinin son tarihinin güncel seans dönemine ait olup olmadığını kontrol eder."""
    import pandas as pd

    if full_df is None or full_df.empty or "Date" not in full_df.columns:
        return False
    try:
        last_date = pd.to_datetime(full_df["Date"].iloc[-1]).date()
        today_date = pd.Timestamp.now(tz="Europe/Istanbul").tz_localize(None).date()
        return abs((today_date - last_date).days) <= max_age_days
    except Exception:
        return False


def apply_live_quote_to_frames(
    df_processed: Any,
    full_df: Any,
    quote: LiveQuote | None,
    macro_data: Any = None,
    force: bool = False,
) -> tuple[Any, Any]:
    """Anlık 0 gecikmeli kotasyonu son gün mumuna entegre eder ve indikatörleri günceller.

    Eğer Yahoo Finance henüz bugünün mumunu açmamışsa (örn. seans açılışının ilk 15 dakikasında
    son satır dünün kapanışıysa ve canlı fiyat/değişim bugünün yeni mumuna işaret ediyorsa)
    yeni gün satırını ekler; aksi halde son gün mumunun OHLCV değerlerini 0 gecikmeli
    TradingView verisiyle günceller ve teknik indikatörleri yeniden hesaplar.
    """
    import pandas as pd

    from src import features

    if quote is None or full_df is None or full_df.empty or quote.last_price <= 0:
        return df_processed, full_df

    if not force and not is_live_frame_recent(full_df):
        return df_processed, full_df

    df_updated = full_df.copy()
    last_idx = df_updated.index[-1]
    last_close = float(df_updated.loc[last_idx, "close"])

    implied_prev_close = (
        quote.last_price / (1.0 + quote.change_pct / 100.0)
        if abs(1.0 + quote.change_pct / 100.0) > 1e-6
        else last_close
    )

    last_date_ts = pd.to_datetime(df_updated.loc[last_idx, "Date"])
    now_tr = pd.Timestamp.now(tz="Europe/Istanbul").tz_localize(None)
    today_date = now_tr.date()

    should_append_new_day = (
        last_date_ts.date() < today_date
        and today_date.weekday() < 5
        and now_tr.hour >= 10
        and last_close > 0
        and abs(last_close - implied_prev_close) / last_close < 0.03
        and abs(quote.change_pct) > 0.01
    )

    open_val = float(quote.open_price) if quote.open_price > 0 else float(quote.last_price)
    high_val = max(float(quote.high), float(quote.last_price), open_val)
    low_val = (
        min(float(quote.low), float(quote.last_price), open_val)
        if quote.low > 0
        else float(quote.last_price)
    )
    vol_val = max(0.0, float(quote.volume))

    if should_append_new_day:
        new_row = df_updated.iloc[[-1]].copy()
        new_idx = new_row.index[0]
        new_row.loc[new_idx, "Date"] = pd.Timestamp(today_date)
        if "open" in new_row.columns:
            new_row.loc[new_idx, "open"] = open_val
        if "high" in new_row.columns:
            new_row.loc[new_idx, "high"] = high_val
        if "low" in new_row.columns:
            new_row.loc[new_idx, "low"] = low_val
        new_row.loc[new_idx, "close"] = float(quote.last_price)
        if "volume" in new_row.columns:
            new_row.loc[new_idx, "volume"] = vol_val
        df_updated = pd.concat([df_updated, new_row], ignore_index=True)
    else:
        df_updated.loc[last_idx, "close"] = float(quote.last_price)
        if "open" in df_updated.columns and quote.open_price > 0:
            df_updated.loc[last_idx, "open"] = open_val
        if "high" in df_updated.columns and quote.high > 0:
            df_updated.loc[last_idx, "high"] = max(
                float(df_updated.loc[last_idx, "high"]), high_val
            )
        if "low" in df_updated.columns and quote.low > 0:
            cur_low = float(df_updated.loc[last_idx, "low"])
            df_updated.loc[last_idx, "low"] = min(cur_low, low_val) if cur_low > 0 else low_val
        if "volume" in df_updated.columns and quote.volume > 0:
            df_updated.loc[last_idx, "volume"] = max(
                float(df_updated.loc[last_idx, "volume"]), vol_val
            )

    try:
        if "ticker" not in df_updated.columns:
            df_updated["ticker"] = f"{quote.ticker}.IS"
        recomputed = features.add_features(
            df_updated,
            macro_df=macro_data,
            drop_incomplete_target=False,
        )
        if not recomputed.empty:
            return recomputed, df_updated
    except Exception as exc:
        logger.debug("Canlı mum indikatör güncellemesi başarısız: %s", exc)

    return df_processed, df_updated
