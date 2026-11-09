"""Otonom Günlük Tahmin & Ledger Güncelleme (src/daily_updater.py) birim testleri."""

import json

import pandas as pd
import pytest

from src import daily_updater, network


@pytest.fixture
def mock_scan_results():
    """Örnek piyasa tarama sonuç tablosu."""
    return pd.DataFrame(
        [
            {
                "Hisse": "THYAO",
                "ticker_code": "THYAO.IS",
                "Son Fiyat (TL)": 310.50,
                "_prob": 0.65,
                "Tahmin": "YUKARI 🚀",
            },
            {
                "Hisse": "GARAN",
                "ticker_code": "GARAN.IS",
                "Son Fiyat (TL)": 125.00,
                "_prob": 0.42,
                "Tahmin": "DÜŞÜŞ 🔻",
            },
            {
                "Hisse": "KCHOL",
                "ticker_code": "KCHOL.IS",
                "Son Fiyat (TL)": 180.20,
                "_prob": 0.58,
                "Tahmin": "YUKARI 🚀",
            },
        ]
    )


def test_run_daily_forecast_dry_run(monkeypatch, mock_scan_results):
    from src import scanner

    monkeypatch.setattr(scanner, "scan_market", lambda *args, **kwargs: (mock_scan_results, []))

    result = daily_updater.run_daily_forecast(dry_run=True, send_email=False)
    assert result["success"] is True
    assert result["recorded_count"] == 3
    assert result["dry_run"] is True
    assert len(result["failed_tickers"]) == 0
    assert len(result["top_picks"]) == 2  # prob >= 0.52 (THYAO %65, KCHOL %58)


def test_run_daily_forecast_records_to_audit_log(monkeypatch, mock_scan_results, tmp_path):
    from src import scanner

    monkeypatch.setattr(scanner, "scan_market", lambda *args, **kwargs: (mock_scan_results, []))

    custom_log = str(tmp_path / "custom_audit.json")
    result = daily_updater.run_daily_forecast(
        dry_run=False,
        send_email=False,
        audit_log_path=custom_log,
    )

    assert result["success"] is True
    assert result["recorded_count"] == 3

    with open(custom_log, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 3
    tickers = {entry["ticker"] for entry in data}
    assert tickers == {"THYAO", "GARAN", "KCHOL"}


def test_run_daily_forecast_handles_market_unavailable(monkeypatch):
    from src import scanner

    def failing_scan(*args, **kwargs):
        raise network.MarketDataUnavailableError("Sunucuya ulaşılamadı")

    monkeypatch.setattr(scanner, "scan_market", failing_scan)

    result = daily_updater.run_daily_forecast(dry_run=True, send_email=False)
    assert result["success"] is False
    assert result["recorded"] == 0
    assert "Sunucuya ulaşılamadı" in result["error"]


def test_run_daily_forecast_sends_email_when_configured(monkeypatch, mock_scan_results, tmp_path):
    from src import email_alerts, scanner

    monkeypatch.setattr(scanner, "scan_market", lambda *args, **kwargs: (mock_scan_results, []))

    sent_emails = []

    def mock_send(*args, **kwargs):
        sent_emails.append(kwargs)
        return True

    monkeypatch.setattr(email_alerts, "send_email_alert", mock_send)
    monkeypatch.setenv("ALERT_RECIPIENT_EMAIL", "test@example.com")
    monkeypatch.setenv("SMTP_USER", "smtp_tester")

    custom_log = str(tmp_path / "custom_audit.json")
    result = daily_updater.run_daily_forecast(
        dry_run=False,
        send_email=True,
        audit_log_path=custom_log,
    )

    assert result["success"] is True
    assert result["email_sent"] is True
    assert len(sent_emails) == 1
