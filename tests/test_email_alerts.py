"""E-posta piyasa bülteni modülü (src/email_alerts.py) birim testleri."""

from unittest.mock import MagicMock, patch

from src import email_alerts


def test_build_newsletter_html_renders_picks():
    picks = [
        {
            "ticker": "THYAO",
            "price": 312.50,
            "prob_pct": 64.2,
            "verdict": "🚀 GÜÇLÜ AL",
            "allocation_pct": 14.5,
        },
        {
            "ticker": "AKBNK",
            "price": 58.90,
            "prob_pct": 59.8,
            "verdict": "📈 ALIM YÖNLÜ",
            "allocation_pct": 10.0,
        },
    ]

    html = email_alerts.build_newsletter_html(
        top_picks=picks,
        report_date="2026-11-09",
        model_version="v2.0.0",
        model_accuracy=0.525,
    )

    assert "<!DOCTYPE html>" in html
    assert "BIST Trend Forecaster" in html
    assert "THYAO" in html
    assert "312.50 TL" in html
    assert "AKBNK" in html
    assert "58.90 TL" in html
    assert "%52.5" in html
    assert "Kelly Payı" in html
    assert "v2.0.0" in html


def test_build_newsletter_html_handles_empty_picks():
    html = email_alerts.build_newsletter_html(top_picks=[])
    assert "<!DOCTYPE html>" in html
    assert "BIST Trend Forecaster" in html


def test_send_email_alert_missing_credentials(monkeypatch):
    monkeypatch.delenv("ALERT_RECIPIENT_EMAIL", raising=False)
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)

    success = email_alerts.send_email_alert(
        subject="Test",
        html_content="<p>Test</p>",
        recipient_email="",
        smtp_user="",
        smtp_password="",
    )
    assert success is False


def test_send_email_alert_success(monkeypatch):
    mock_smtp_instance = MagicMock()
    mock_smtp_class = MagicMock(return_value=mock_smtp_instance)
    mock_smtp_instance.__enter__.return_value = mock_smtp_instance

    with patch("smtplib.SMTP", mock_smtp_class):
        success = email_alerts.send_email_alert(
            subject="Bülten Test",
            html_content="<h1>Test Bülteni</h1>",
            recipient_email="investor@example.com",
            smtp_server="smtp.example.com",
            smtp_port=587,
            smtp_user="bot@example.com",
            smtp_password="secretpassword",
        )

        assert success is True
        mock_smtp_class.assert_called_once_with("smtp.example.com", 587, timeout=15)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("bot@example.com", "secretpassword")
        mock_smtp_instance.send_message.assert_called_once()


def test_send_email_alert_handles_smtp_exception():
    mock_smtp_class = MagicMock(side_effect=ConnectionRefusedError("SMTP sunucusuna bağlanılamadı"))

    with patch("smtplib.SMTP", mock_smtp_class):
        success = email_alerts.send_email_alert(
            subject="Hata Testi",
            html_content="<p>Hata</p>",
            recipient_email="investor@example.com",
            smtp_port=587,
            smtp_user="bot@example.com",
            smtp_password="password",
        )

        assert success is False


def test_send_email_alert_ssl_port_465_resend():
    mock_ssl_instance = MagicMock()
    mock_ssl_class = MagicMock(return_value=mock_ssl_instance)
    mock_ssl_instance.__enter__.return_value = mock_ssl_instance

    with patch("smtplib.SMTP_SSL", mock_ssl_class):
        success = email_alerts.send_email_alert(
            subject="Resend SSL Test",
            html_content="<h1>Resend Test</h1>",
            recipient_email="target@example.com",
            smtp_server="smtp.resend.com",
            smtp_port=465,
            smtp_user="resend",
            smtp_password="re_secret_key",
        )

        assert success is True
        mock_ssl_class.assert_called_once_with("smtp.resend.com", 465, timeout=15)
        mock_ssl_instance.login.assert_called_once_with("resend", "re_secret_key")
        mock_ssl_instance.send_message.assert_called_once()
        msg = mock_ssl_instance.send_message.call_args[0][0]
        assert "onboarding@resend.dev" in msg["From"]
        assert msg["To"] == "target@example.com"


def test_send_email_alert_custom_from_email(monkeypatch):
    mock_ssl_instance = MagicMock()
    mock_ssl_class = MagicMock(return_value=mock_ssl_instance)
    mock_ssl_instance.__enter__.return_value = mock_ssl_instance
    monkeypatch.setenv("SMTP_FROM_EMAIL", "custom_sender@mybist.com")

    with patch("smtplib.SMTP_SSL", mock_ssl_class):
        success = email_alerts.send_email_alert(
            subject="Custom From Test",
            html_content="<h1>Custom From</h1>",
            recipient_email="target@example.com",
            smtp_server="smtp.resend.com",
            smtp_port=465,
            smtp_user="resend",
            smtp_password="re_secret_key",
        )

        assert success is True
        msg = mock_ssl_instance.send_message.call_args[0][0]
        assert "custom_sender@mybist.com" in msg["From"]


def test_send_test_email(monkeypatch):
    sent_args = {}

    def mock_send_email_alert(**kwargs):
        sent_args.update(kwargs)
        return True

    monkeypatch.setattr(email_alerts, "send_email_alert", mock_send_email_alert)
    success = email_alerts.send_test_email(recipient_email="test_investor@example.com")

    assert success is True
    assert "TEST" in sent_args["subject"]
    assert sent_args["recipient_email"] == "test_investor@example.com"
    assert "<!DOCTYPE html>" in sent_args["html_content"]
    assert "THYAO" in sent_args["html_content"]


def test_build_newsletter_html_with_multi_horizon_data():
    picks = [
        {
            "ticker": "THYAO",
            "price": 310.50,
            "prob_1d": 64.2,
            "prob_5d": 68.5,
            "prob_20d": 72.0,
            "confluence": "🏛️ GÜÇLÜ BOĞA",
            "verdict": "🏛️ GÜÇLÜ BOĞA",
            "corridor_1d": "295.00 TL — 326.00 TL",
            "allocation_pct": 14.0,
            "sector": "Ulaştırma",
            "adx_text": "ADX: 28.5 (Güçlü Trend)",
        }
    ]

    html = email_alerts.build_newsletter_html(
        top_picks=picks,
        report_date="2026-11-09",
        model_version="v2.1-ensemble",
        model_accuracy=0.528,
    )

    assert "THYAO" in html
    assert "310.50 TL" in html
    assert "GÜÇLÜ BOĞA" in html
    assert "1 GÜN (YARIN)" in html
    assert "1 HAFTA (5 GÜN)" in html
    assert "1 AY (20 GÜN)" in html
    assert "%64.2" in html
    assert "%68.5" in html
    assert "%72.0" in html
    assert "295.00 TL — 326.00 TL" in html
    assert "ADX: 28.5" in html
    assert "%14.0" in html
