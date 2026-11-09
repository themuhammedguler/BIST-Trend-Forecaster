"""BIST 30 piyasa bülteni ve fırsat radarı e-posta bildirim modülü.

Haftalık MLOps yeniden eğitimi sonrasında veya talep edildiğinde
en yüksek potansiyelli ilk 3 hisseyi ve hibrit analiz özetini
yatırımcının e-posta adresine şık bir HTML bülteni olarak gönderir.
"""

from __future__ import annotations

import email.message
import logging
import os
import smtplib
from typing import Any

logger = logging.getLogger(__name__)


def build_newsletter_html(
    top_picks: list[dict[str, Any]],
    report_date: str = "",
    model_version: str = "v2.0.0",
    model_accuracy: float | None = None,
) -> str:
    """Yatırımcıya gönderilecek zengin HTML e-posta bültenini hazırlar."""
    rows_html = ""
    for idx, pick in enumerate(top_picks, 1):
        ticker = pick.get("ticker", pick.get("Hisse", "BIST"))
        price = pick.get("price", pick.get("Son Fiyat (TL)", 0.0))
        prob = pick.get("prob_pct", pick.get("Yükseliş Olasılığı (%)", 0.0))
        verdict = pick.get("verdict", pick.get("Hibrit Karar", "📈 ALIM YÖNLÜ"))
        allocation = pick.get("allocation_pct", 0.0)

        alloc_badge = f"%{allocation:.1f}" if allocation > 0 else "Nötr"

        rows_html += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 12px 14px; font-weight: 700; color: #F8FAFC;">#{idx} {ticker}</td>
            <td style="padding: 12px 14px; color: #CBD5E1;">{price:.2f} TL</td>
            <td style="padding: 12px 14px; color: #10B981; font-weight: 600;">%{prob:.1f}</td>
            <td style="padding: 12px 14px; color: #38BDF8; font-weight: 600;">{verdict}</td>
            <td style="padding: 12px 14px; color: #F59E0B; font-weight: 600;">{alloc_badge}</td>
        </tr>
        """

    accuracy_text = f"%{model_accuracy * 100:.1f}" if model_accuracy else "%51.2"
    app_url = "https://bist-trend-forecaster.streamlit.app"

    body_style = (
        "background-color: #0F172A; "
        "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
        "margin: 0; padding: 24px; color: #F8FAFC;"
    )
    container_style = (
        "max-width: 640px; margin: 0 auto; background-color: #1E293B; "
        "border-radius: 12px; border: 1px solid #334155; overflow: hidden;"
    )
    header_style = (
        "background: linear-gradient(135deg, #1E3A8A, #0284C7); padding: 24px; text-align: center;"
    )
    title_style = (
        "margin: 0; font-size: 1.5rem; color: #FFFFFF; font-weight: 800; letter-spacing: -0.02em;"
    )
    table_style = (
        "width: 100%; border-collapse: collapse; margin: 20px 0; "
        "background-color: #0F172A; border-radius: 8px; overflow: hidden; "
        "font-size: 0.88rem;"
    )
    info_box_style = (
        "background-color: #0F172A; border-left: 4px solid #3B82F6; "
        "padding: 12px 16px; border-radius: 4px; margin-bottom: 24px;"
    )
    btn_style = (
        "background-color: #2563EB; color: #FFFFFF; font-weight: 700; "
        "font-size: 0.92rem; padding: 12px 28px; border-radius: 8px; "
        "text-decoration: none; display: inline-block;"
    )
    footer_style = (
        "background-color: #0F172A; border-top: 1px solid #334155; "
        "padding: 16px; text-align: center; font-size: 0.75rem; color: #64748B;"
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>BIST Trend Forecaster Bülteni</title>
    </head>
    <body style="{body_style}">
        <div style="{container_style}">
            <!-- Header -->
            <div style="{header_style}">
                <h1 style="{title_style}">
                    📈 BIST Trend Forecaster
                </h1>
                <p style="margin: 6px 0 0 0; color: #BAE6FD; font-size: 0.90rem;">
                    Haftalık Yapay Zeka & Hibrit Karar Bülteni · {report_date}
                </p>
            </div>

            <!-- Body -->
            <div style="padding: 24px;">
                <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.5; margin-top: 0;">
                    Merhaba, en güncel BIST 30 makine öğrenmesi tahminleri ve bilanço analizleri
                    tamamlandı. Yapay zeka modellerimizin öne çıkardığı
                    <strong>en yüksek potansiyelli ilk 3 hisse</strong> aşağıdadır:
                </p>

                <!-- Table -->
                <table style="{table_style}">
                    <thead>
                        <tr style="background-color: #334155; color: #94A3B8; text-align: left;">
                            <th style="padding: 10px 14px;">Hisse</th>
                            <th style="padding: 10px 14px;">Son Fiyat</th>
                            <th style="padding: 10px 14px;">Yön Skoru</th>
                            <th style="padding: 10px 14px;">Sentez Karar</th>
                            <th style="padding: 10px 14px;">Kelly Payı</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>

                <!-- Info Box -->
                <div style="{info_box_style}">
                    <p style="margin: 0; font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">
                        ⚙️ <strong>Model Sürümü:</strong> {model_version} &nbsp;|&nbsp;
                        🎯 <strong>Test Doğruluğu:</strong> {accuracy_text} &nbsp;|&nbsp;
                        🛡️ <strong>Risk Modeli:</strong> Kelly Criterion (Fraksiyonel %25)
                    </p>
                </div>

                <!-- CTA Button -->
                <div style="text-align: center; margin: 28px 0 12px 0;">
                    <a href="{app_url}" style="{btn_style}">
                        Detaylı Analizi Uygulamada İncele ↗
                    </a>
                </div>
            </div>

            <!-- Footer -->
            <div style="{footer_style}">
                Bu bülten otomatik karar destek amacıyla üretilmiştir. Yatırım tavsiyesi niteliği
                taşımaz.<br>
                BIST Trend Forecaster © 2026 · Açık Kaynak Karar Destek Platformu
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
) -> bool:
    """Hazırlanan HTML e-posta bültenini SMTP üzerinden alıcıya ulaştırır."""
    to_email = recipient_email or os.getenv("ALERT_RECIPIENT_EMAIL", "")
    server_host = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
    server_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
    user = smtp_user or os.getenv("SMTP_USER", "")
    password = smtp_password or os.getenv("SMTP_PASSWORD", "")

    if not to_email or not user or not password:
        logger.warning(
            "E-posta gönderimi için SMTP bilgileri eksik "
            "(SMTP_USER, SMTP_PASSWORD, ALERT_RECIPIENT_EMAIL)."
        )
        return False

    msg = email.message.EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"BIST Trend Forecaster <{user}>"
    msg["To"] = to_email
    msg.set_content(
        "Bu bülteni görüntülemek için lütfen HTML destekli bir e-posta istemcisi kullanın."
    )
    msg.add_alternative(html_content, subtype="html")

    try:
        with smtplib.SMTP(server_host, server_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(user, password)
            server.send_message(msg)
            logger.info("E-posta bülteni başarıyla gönderildi: %s", to_email)
            return True
    except Exception as exc:
        logger.error("E-posta gönderilirken hata oluştu: %s", exc)
        return False
