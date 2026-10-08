# tests/test_session_calendar.py - BIST seans takvimi ve tarih formatlama testleri
from datetime import datetime

import pandas as pd

from src import session_calendar


def test_format_turkish_date_formats_correctly():
    dt = pd.Timestamp("2026-10-02")  # Cuma
    formatted = session_calendar.format_turkish_date(dt)
    assert formatted == "2 Ekim 2026, Cuma"


def test_format_turkish_date_handles_various_months_and_days():
    cases = [
        ("2025-01-01", "1 Ocak 2025, Çarşamba"),
        ("2025-04-23", "23 Nisan 2025, Çarşamba"),
        ("2025-05-19", "19 Mayıs 2025, Pazartesi"),
        ("2025-08-30", "30 Ağustos 2025, Cumartesi"),
        ("2025-10-29", "29 Ekim 2025, Çarşamba"),
        ("2025-12-31", "31 Aralık 2025, Çarşamba"),
    ]
    for date_str, expected in cases:
        assert session_calendar.format_turkish_date(date_str) == expected


def test_format_turkish_date_handles_tz_aware_timestamp():
    dt = pd.Timestamp("2026-10-02 18:15:00", tz="Europe/Istanbul")
    formatted = session_calendar.format_turkish_date(dt)
    assert formatted == "2 Ekim 2026, Cuma"


def test_get_next_trading_session_advances_friday_to_monday():
    friday = pd.Timestamp("2026-10-02")
    next_session = session_calendar.get_next_trading_session(friday)
    assert next_session == pd.Timestamp("2026-10-05")
    assert next_session.day_name() == "Monday"


def test_get_next_trading_session_advances_midweek():
    thursday = pd.Timestamp("2026-10-01")
    next_session = session_calendar.get_next_trading_session(thursday)
    assert next_session == pd.Timestamp("2026-10-02")
    assert next_session.day_name() == "Friday"


def test_get_session_info_weekday_without_weekend_notice():
    last_close = "2026-10-01"  # Perşembe
    now_midweek = datetime(2026, 10, 1, 20, 0)  # Perşembe akşamı

    info = session_calendar.get_session_info(last_close, now=now_midweek)

    assert not info.is_weekend
    assert info.weekend_notice is None
    assert "Analiz Edilen Son Kapanış: 1 Ekim 2026, Perşembe" in info.badge_text
    assert "Hedef Seans: 2 Ekim 2026, Cuma" in info.badge_text
    assert info.last_close_str == "1 Ekim 2026, Perşembe"
    assert info.next_session_str == "2 Ekim 2026, Cuma"


def test_get_session_info_on_weekend_includes_notice():
    last_close = "2026-10-02"  # Cuma kapanışı
    now_saturday = datetime(2026, 10, 3, 14, 30)  # Cumartesi günü

    info = session_calendar.get_session_info(last_close, now=now_saturday)

    assert info.is_weekend
    assert info.weekend_notice is not None
    assert "Borsa İstanbul şu anda kapalıdır" in info.weekend_notice
    assert "5 Ekim 2026, Pazartesi" in info.weekend_notice
    assert "Analiz Edilen Son Kapanış: 2 Ekim 2026, Cuma" in info.badge_text
    assert "Hedef Seans: 5 Ekim 2026, Pazartesi" in info.badge_text


def test_get_session_info_on_sunday_includes_notice():
    last_close = "2026-10-02"  # Cuma
    now_sunday = pd.Timestamp("2026-10-04 11:00")

    info = session_calendar.get_session_info(last_close, now=now_sunday)

    assert info.is_weekend
    assert info.weekend_notice is not None
    assert "5 Ekim 2026, Pazartesi" in info.next_session_str


def test_get_session_info_uses_istanbul_time_for_weekend_check():
    # UTC'de hâlâ Cuma 22:30, İstanbul'da Cumartesi 01:30
    friday_night_utc = pd.Timestamp("2026-10-02 22:30", tz="UTC")
    assert session_calendar.get_session_info("2026-10-02", now=friday_night_utc).is_weekend

    # UTC'de hâlâ Pazar 22:30, İstanbul'da Pazartesi 01:30
    sunday_night_utc = pd.Timestamp("2026-10-04 22:30", tz="UTC")
    assert not session_calendar.get_session_info("2026-10-02", now=sunday_night_utc).is_weekend


def test_get_session_info_defaults_to_istanbul_now(monkeypatch):
    def fake_now(tz=None):
        assert tz is not None, "now() saat dilimi olmadan çağrıldı"
        return pd.Timestamp("2026-10-02 22:30", tz="UTC").tz_convert(tz)

    monkeypatch.setattr(session_calendar.pd.Timestamp, "now", staticmethod(fake_now))
    assert session_calendar.get_session_info("2026-10-02").is_weekend
