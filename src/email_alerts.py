"""BIST 30 piyasa bülteni ve fırsat radarı e-posta bildirim modülü.

Her seans kapanışı sonrasında veya talep edildiğinde çoklu vade konsensüsü
(1 Gün, 1 Hafta, 1 Ay), fiyat koridorları, teknik göstergeler ve Kelly sermaye
dağılımı özetini yatırımcının e-posta adresine modern ve şık bir HTML bülteni
olarak iletir.
"""

from __future__ import annotations

import base64
import email.message
import logging
import os
import smtplib
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")
LOGO_PUBLIC_URL = (
    "https://cdn.jsdelivr.net/gh/themuhammedguler/BIST-Trend-Forecaster@main/assets/logo.png"
)


def _get_logo_base64() -> str | None:
    """Logo dosyasını base64 formatında okur."""
    if os.path.exists(LOGO_PATH):
        try:
            with open(LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None
    return None


def build_newsletter_html(
    top_picks: list[dict[str, Any]],
    report_date: str = "",
    model_version: str = "v2.1-ensemble",
    model_accuracy: float | None = None,
    market_summary: str | None = None,
) -> str:
    """Çoklu vade, fiyat koridorları ve risk masası içeren şık HTML bülteni üretir."""
    accuracy_text = f"%{model_accuracy * 100:.1f}" if model_accuracy else "%52.1"
    app_url = "https://bist-trend-forecaster.streamlit.app"

    # Logo HTML: E-posta istemcileri (özellikle Gmail ve Outlook) base64 data: URI'leri
    # güvenlik sebebiyle engellediğinden CDN üzerinden herkese açık HTTPS URL kullanılır.
    logo_html = (
        f'<img src="{LOGO_PUBLIC_URL}" width="60" height="60" '
        f'style="border-radius: 50%; box-shadow: 0 0 16px rgba(0, 229, 255, 0.35); '
        f'border: 2px solid #00E5FF; vertical-align: middle; display: inline-block;" '
        f'alt="BIST Trend Forecaster" />'
    )

    # Hisse Kartları HTML Üretimi
    cards_html = ""
    for idx, pick in enumerate(top_picks, 1):
        ticker = pick.get("ticker", pick.get("Hisse", "BIST"))
        price = float(pick.get("price", pick.get("Son Fiyat (TL)", 0.0)))
        prob_1d = float(pick.get("prob_1d", pick.get("prob_pct", 0.0)))
        prob_5d = float(pick.get("prob_5d", prob_1d))
        prob_20d = float(pick.get("prob_20d", prob_1d))
        verdict = pick.get("verdict", pick.get("confluence", "🚀 GÜÇLÜ BOĞA"))
        corridor = pick.get("corridor_1d", pick.get("corridor", ""))
        allocation = float(pick.get("allocation_pct", 0.0))
        adx_text = pick.get("adx_text", "")
        sector = pick.get("sector", pick.get("Sektör", ""))

        # Vade Konsensüs rozet rengi
        badge_bg = "rgba(5, 150, 105, 0.22)"
        badge_border = "#059669"
        badge_color = "#34D399"
        if "DÜŞÜŞ" in verdict or "AYI" in verdict:
            badge_bg = "rgba(220, 38, 38, 0.22)"
            badge_border = "#DC2626"
            badge_color = "#F87171"
        elif "NÖTR" in verdict or "YATAY" in verdict:
            badge_bg = "rgba(217, 119, 6, 0.22)"
            badge_border = "#D97706"
            badge_color = "#FBBF24"

        alloc_badge = f"%{allocation:.1f}" if allocation > 0 else "Nötr"

        extra_info = ""
        if corridor:
            extra_info += (
                f'<div style="font-size: 0.80rem; color: #94A3B8; margin-top: 6px;">'
                f"🛡️ <strong>%95 Fiyat Koridoru (1G):</strong> "
                f'<span style="color: #38BDF8; font-weight: 600;">{corridor}</span></div>'
            )
        if adx_text:
            extra_info += (
                f'<div style="font-size: 0.80rem; color: #94A3B8; margin-top: 4px;">'
                f"📐 <strong>Trend & Göstergeler:</strong> {adx_text}</div>"
            )

        cards_html += f"""
        <div style="background-color: #0F131D; border: 1px solid #1E293B; border-radius: 10px;
                    padding: 16px 18px; margin-bottom: 14px;
                    border-left: 4px solid {badge_border};">
            <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%"
                   style="margin-bottom: 8px;">
                <tr>
                    <td align="left" style="vertical-align: middle;">
                        <span style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">
                            #{idx} {ticker}
                        </span>
                        <span style="font-size: 0.82rem; color: #64748B; margin-left: 6px;">
                            {sector}
                        </span>
                        <span style="font-size: 1.05rem; font-weight: 700; color: #38BDF8;
                                     margin-left: 10px;">
                            {price:.2f} TL
                        </span>
                    </td>
                    <td align="right" style="vertical-align: middle;">
                        <span style="background-color: {badge_bg}; color: {badge_color};
                                     font-size: 0.78rem; font-weight: 700; padding: 4px 10px;
                                     border-radius: 6px; border: 1px solid {badge_border};
                                     display: inline-block; white-space: nowrap;">
                            {verdict}
                        </span>
                    </td>
                </tr>
            </table>

            <!-- 3 Vade İhtimal Çubukları (Email Table) -->
            <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%"
                   style="background-color: #07090E; border: 1px solid #1A2233; border-radius: 8px;
                          margin: 12px 0 10px 0; table-layout: fixed; border-collapse: separate;">
                <tr>
                    <td align="center" style="padding: 10px 4px; width: 25%;
                                              border-right: 1px solid #1E293B;">
                        <div style="font-size: 0.68rem; color: #94A3B8; font-weight: 600;
                                    letter-spacing: 0.01em;">
                            1 GÜN (YARIN)
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC;
                                    margin-top: 4px;">
                            %{prob_1d:.1f}
                        </div>
                    </td>
                    <td align="center" style="padding: 10px 4px; width: 25%;
                                              border-right: 1px solid #1E293B;">
                        <div style="font-size: 0.68rem; color: #94A3B8; font-weight: 600;
                                    letter-spacing: 0.01em;">
                            1 HAFTA (5 GÜN)
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC;
                                    margin-top: 4px;">
                            %{prob_5d:.1f}
                        </div>
                    </td>
                    <td align="center" style="padding: 10px 4px; width: 25%;
                                              border-right: 1px solid #1E293B;">
                        <div style="font-size: 0.68rem; color: #94A3B8; font-weight: 600;
                                    letter-spacing: 0.01em;">
                            1 AY (20 GÜN)
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC;
                                    margin-top: 4px;">
                            %{prob_20d:.1f}
                        </div>
                    </td>
                    <td align="center" style="padding: 10px 4px; width: 25%;">
                        <div style="font-size: 0.68rem; color: #94A3B8; font-weight: 600;
                                    letter-spacing: 0.01em;">
                            Kelly Payı
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 800; color: #F59E0B;
                                    margin-top: 4px;">
                            {alloc_badge}
                        </div>
                    </td>
                </tr>
            </table>

            {extra_info}
        </div>
        """

    summary_note = market_summary or (
        "BIST 30 seans kapanışı sonrasında yapay zeka topluluk motoru (XGBoost, CatBoost, "
        "LightGBM) ve çoklu vade trend modelleri çalıştırılmış; en güçlü alım konfluansı veren "
        "hisseler filtrelenmiştir."
    )

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BIST Trend Forecaster · Günlük Seans Kapanış Bülteni</title>
</head>
<body style="background-color: #07090E; font-family: -apple-system, BlinkMacSystemFont,
             'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 24px 10px;
             color: #E2E8F0;">
    <div style="max-width: 660px; margin: 0 auto; background-color: #0B0E17; border-radius: 14px;
                border: 1px solid #1E293B; overflow: hidden;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5);">

        <!-- Header -->
        <div style="background: linear-gradient(135deg, #091326 0%, #0F172A 50%, #172554 100%);
                    padding: 26px 20px; text-align: center; border-bottom: 1px solid #1E293B;">
            <div style="margin-bottom: 10px;">
                {logo_html}
            </div>
            <h1 style="margin: 0; font-size: 1.45rem; color: #F8FAFC; font-weight: 800;
                       letter-spacing: -0.01em;">
                BIST Trend Forecaster
            </h1>
            <p style="margin: 6px 0 0 0; color: #38BDF8; font-size: 0.88rem; font-weight: 600;">
                🏛️ Günlük Seans Kapanış ve Çoklu Vade Trend Bülteni · {report_date}
            </p>
        </div>

        <!-- Body -->
        <div style="padding: 24px 22px;">
            <p style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.6; margin-top: 0;">
                {summary_note}
            </p>

            <div style="margin: 18px 0 8px 0;">
                <h3 style="margin: 0 0 12px 0; font-size: 1.02rem; color: #F8FAFC;
                           font-weight: 700;">
                    🎯 Günün Öne Çıkan Fırsat & Vade Konsensüs Sinyalleri:
                </h3>
            </div>

            <!-- Cards -->
            {cards_html}

            <!-- Info Bar -->
            <div style="background-color: #0F131D; border-left: 4px solid #10B981;
                        padding: 12px 16px; border-radius: 6px; margin: 20px 0 24px 0;">
                <p style="margin: 0; font-size: 0.80rem; color: #94A3B8; line-height: 1.6;">
                    ⚙️ <strong>Model Mimarisi:</strong> {model_version} &nbsp;•&nbsp;
                    🎯 <strong>Test Doğruluğu:</strong> {accuracy_text} &nbsp;•&nbsp;
                    🛡️ <strong>Risk Modeli:</strong> Fraksiyonel Kelly (%25) & MPT
                </p>
            </div>

            <!-- CTA Button -->
            <div style="text-align: center; margin: 26px 0 10px 0;">
                <a href="{app_url}" style="background-color: #2563EB; color: #FFFFFF;
                   font-weight: 700; font-size: 0.92rem; padding: 13px 30px; border-radius: 8px;
                   text-decoration: none; display: inline-block;
                   box-shadow: 0 0 16px rgba(37, 99, 235, 0.4);">
                    📊 Tüm Detayları Canlı Uygulamada İncele ↗
                </a>
            </div>
        </div>

        <!-- Footer -->
        <div style="background-color: #07090E; border-top: 1px solid #1E293B; padding: 18px 20px;
                    text-align: center; font-size: 0.74rem; color: #64748B; line-height: 1.5;">
            Bu bülten otomatik yapay zeka analiz robotu tarafından üretilmiştir.
            Yatırım tavsiyesi niteliği taşımaz.<br>
            <strong>BIST Trend Forecaster</strong> © 2026 · Açık Kaynak Karar Destek Platformu
        </div>
    </div>
</body>
</html>
"""


def send_email_alert(
    subject: str,
    html_content: str,
    recipient_email: str | None = None,
    smtp_server: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_password: str | None = None,
    from_email: str | None = None,
) -> bool:
    """Hazırlanan HTML e-posta bültenini SMTP üzerinden alıcıya ulaştırır."""
    to_email = (recipient_email or os.getenv("ALERT_RECIPIENT_EMAIL", "")).strip()
    raw_server = (smtp_server or os.getenv("SMTP_SERVER", "smtp.resend.com")).strip()

    # URL formatında girilmişse protokol ve path temizlenir
    # (örn. https://smtp.resend.com/ -> smtp.resend.com)
    if "://" in raw_server:
        raw_server = raw_server.split("://")[-1]
    server_host = raw_server.split("/")[0].split(":")[0].strip()
    if server_host.lower() == "resend.com":
        server_host = "smtp.resend.com"

    port_val = str(smtp_port or os.getenv("SMTP_PORT", "465")).strip()
    try:
        server_port = int(port_val)
    except (ValueError, TypeError):
        server_port = 465

    user = (smtp_user or os.getenv("SMTP_USER", "")).strip("'\" \t\r\n")
    password = (smtp_password or os.getenv("SMTP_PASSWORD", "")).strip("'\" \t\r\n")
    sender_addr = (from_email or os.getenv("SMTP_FROM_EMAIL", "")).strip("'\" \t\r\n")

    # Resend veya Gmail sağlayıcıları için sunucu adı otomatik düzeltilir
    if user.lower() == "resend" or "resend" in server_host.lower():
        server_host = "smtp.resend.com"
    elif "gmail" in server_host.lower() or user.lower().endswith("@gmail.com"):
        server_host = "smtp.gmail.com"

    if not to_email or not user or not password:
        logger.warning(
            "E-posta gönderimi için SMTP bilgileri eksik "
            "(SMTP_USER, SMTP_PASSWORD, ALERT_RECIPIENT_EMAIL)."
        )
        return False

    if not sender_addr:
        if user.lower() == "resend":
            sender_addr = "onboarding@resend.dev"
        elif "@" in user:
            sender_addr = user
        else:
            sender_addr = "onboarding@resend.dev"

    logger.info(
        "SMTP bağlantısı kuruluyor: host='%s', port=%d, sender='%s', recipient='%s'",
        server_host,
        server_port,
        sender_addr,
        to_email,
    )

    msg = email.message.EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"BIST Trend Forecaster <{sender_addr}>"
    msg["To"] = to_email
    msg.set_content(
        "Bu bülteni görüntülemek için lütfen HTML destekli bir e-posta istemcisi kullanın."
    )
    msg.add_alternative(html_content, subtype="html")

    def _connect_and_send(host: str, port: int) -> None:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=15) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(user, password)
                server.send_message(msg)

    try:
        _connect_and_send(server_host, server_port)
        logger.info("E-posta bülteni başarıyla gönderildi: %s", to_email)
        return True
    except Exception as exc:
        logger.warning(
            "SMTP ilk denemede hata (host='%s', port=%d): %s. Alternatif port deneniyor...",
            server_host,
            server_port,
            exc,
        )
        alt_port = 587 if server_port == 465 else 465
        try:
            _connect_and_send(server_host, alt_port)
            logger.info(
                "E-posta bülteni alternatif port (%d) üzerinden başarıyla gönderildi: %s",
                alt_port,
                to_email,
            )
            return True
        except Exception as alt_exc:
            logger.error(
                "E-posta her iki portta da gönderilemedi (asıl: %d, alternatif: %d). Hata: %s",
                server_port,
                alt_port,
                alt_exc,
            )
            return False


def send_test_email(recipient_email: str | None = None) -> bool:
    """Hızlı bağlantı testi için örnek BIST 30 verisiyle test bülteni gönderir."""
    target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sample_picks = [
        {
            "ticker": "THYAO",
            "price": 314.50,
            "prob_1d": 64.8,
            "prob_5d": 68.2,
            "prob_20d": 72.0,
            "verdict": "🏛️ GÜÇLÜ BOĞA",
            "confluence": "🏛️ GÜÇLÜ BOĞA",
            "corridor_1d": "302.00 TL — 328.00 TL",
            "allocation_pct": 14.0,
            "sector": "Ulaştırma",
            "adx_text": "ADX: 29.4 (Güçlü Trend)",
        },
        {
            "ticker": "ASELS",
            "price": 68.40,
            "prob_1d": 59.2,
            "prob_5d": 62.5,
            "prob_20d": 65.0,
            "verdict": "🚀 YÜKSELİŞ",
            "confluence": "🚀 YÜKSELİŞ",
            "corridor_1d": "66.00 TL — 71.50 TL",
            "allocation_pct": 10.0,
            "sector": "Savunma",
            "adx_text": "ADX: 24.1 (Yükseliş)",
        },
        {
            "ticker": "BIMAS",
            "price": 492.00,
            "prob_1d": 56.4,
            "prob_5d": 58.0,
            "prob_20d": 61.2,
            "verdict": "📈 ALIM YÖNLÜ",
            "confluence": "📈 ALIM YÖNLÜ",
            "corridor_1d": "480.00 TL — 505.00 TL",
            "allocation_pct": 6.0,
            "sector": "Perakende",
            "adx_text": "ADX: 21.0 (Ilımlı Trend)",
        },
    ]
    html = build_newsletter_html(
        top_picks=sample_picks,
        report_date=target_date,
        model_version="v2.1-ensemble (Test Modu)",
        model_accuracy=0.5213,
        market_summary=(
            "Bu bir test e-postasıdır. BIST Trend Forecaster e-posta bildirim servisinin ve "
            "SMTP yapılandırmasının sorunsuz çalıştığını doğrulamak amacıyla gönderilmiştir."
        ),
    )
    return send_email_alert(
        subject=f"🧪 [TEST] BIST Trend Forecaster Bülten Bağlantı Testi ({target_date})",
        html_content=html,
        recipient_email=recipient_email,
    )
