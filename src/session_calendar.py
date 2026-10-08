# src/session_calendar.py
"""BIST seans takvimi ve işlem günü farkındalığı modülü.

Hafta sonları ve seans dışı zamanlarda en son kapanış verisi ile
tahmin yapılan bir sonraki işlem seansını belirler, Türkçe tarih
formatında kullanıcı arayüzüne sunar.
"""

from dataclasses import dataclass

import pandas as pd

# Hafta sonu kontrolü sunucu saatine göre değil, borsanın saatine göre yapılır
BIST_TZ = "Europe/Istanbul"

TURKISH_DAYS = {
    0: "Pazartesi",
    1: "Salı",
    2: "Çarşamba",
    3: "Perşembe",
    4: "Cuma",
    5: "Cumartesi",
    6: "Pazar",
}

TURKISH_MONTHS = {
    1: "Ocak",
    2: "Şubat",
    3: "Mart",
    4: "Nisan",
    5: "Mayıs",
    6: "Haziran",
    7: "Temmuz",
    8: "Ağustos",
    9: "Eylül",
    10: "Ekim",
    11: "Kasım",
    12: "Aralık",
}


def _normalize_date(dt) -> pd.Timestamp:
    ts = pd.to_datetime(dt)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(BIST_TZ).tz_localize(None)
    return ts.normalize()


def format_turkish_date(dt) -> str:
    """Verilen tarihi '2 Ekim 2026, Cuma' biçiminde Türkçe döndürür."""
    ts = _normalize_date(dt)
    day = ts.day
    month_name = TURKISH_MONTHS.get(ts.month, str(ts.month))
    year = ts.year
    day_name = TURKISH_DAYS.get(ts.weekday(), "")
    return f"{day} {month_name} {year}, {day_name}"


def get_next_trading_session(last_date) -> pd.Timestamp:
    """Son işlem gününden sonraki ilk BIST seans gününü hesaplar.
    Hafta sonlarını (Cuma -> Pazartesi) otomatik atlar.
    """
    ts = _normalize_date(last_date)
    return ts + pd.offsets.BDay(1)


@dataclass
class SessionInfo:
    last_close_date: pd.Timestamp
    next_session_date: pd.Timestamp
    last_close_str: str
    next_session_str: str
    is_weekend: bool
    badge_text: str
    weekend_notice: str | None


def get_session_info(last_date, now=None) -> SessionInfo:
    """Son kapanış tarihi ve şimdiki zamana göre seans takvimi özetini oluşturur."""
    last_ts = _normalize_date(last_date)
    next_ts = get_next_trading_session(last_ts)

    now_ts = _normalize_date(pd.Timestamp.now(tz=BIST_TZ) if now is None else now)

    is_wknd = now_ts.weekday() >= 5  # 5: Cumartesi, 6: Pazar

    last_str = format_turkish_date(last_ts)
    next_str = format_turkish_date(next_ts)

    badge = f"📅 Analiz Edilen Son Kapanış: {last_str} | Hedef Seans: {next_str}"

    weekend_notice = None
    if is_wknd:
        weekend_notice = (
            f"ℹ️ **Hafta Sonu Bildirimi:** Borsa İstanbul şu anda kapalıdır. "
            f"En son kapanış verisi ({last_str}) baz alınarak üretilen tahmin, "
            f"bir sonraki işlem seansını (**{next_str}**) hedeflemektedir."
        )

    return SessionInfo(
        last_close_date=last_ts,
        next_session_date=next_ts,
        last_close_str=last_str,
        next_session_str=next_str,
        is_weekend=is_wknd,
        badge_text=badge,
        weekend_notice=weekend_notice,
    )
