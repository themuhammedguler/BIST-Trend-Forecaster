"""Otonom Günlük Tahmin & Ledger Güncelleme Modülü.

Her iş günü BIST seansı kapandıktan sonra (18:30 TRT / 15:30 UTC):
1. En güncel seans verilerini toplu olarak çeker.
2. Tüm BIST 30 hisseleri için yapay zeka modelinden yön tahminlerini üretir.
3. Tahminleri, giriş fiyatlarını ve sinyalleri prediction_audit.json defterine işler.
4. Yapılandırılmışsa en güçlü fırsatları e-posta bülteni olarak iletir.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Any

import xgboost as xgb

from src import config, email_alerts, network, scanner, tracker

logger = logging.getLogger(__name__)


def run_daily_forecast(
    dry_run: bool = False,
    send_email: bool = True,
    audit_log_path: str | None = None,
) -> dict[str, Any]:
    """Günlük seans kapanış tahminlerini üretir ve denetim defterine kaydeder.

    Returns:
        dict: İşlem özeti (tarih, kaydedilen_adet, basarisizlar, en_iyi_firsatlar).
    """
    logger.info("Günlük tahmin süreci başlatılıyor...")

    # 1. Modeli Yükle
    model = xgb.XGBClassifier()
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {config.MODEL_PATH}")
    model.load_model(config.MODEL_PATH)

    # 2. Piyasa Taraması (BIST 30 + Makro Göstergeler)
    try:
        df_scan, failed = scanner.scan_market(config.TICKERS, model, include_macro=True)
    except network.MarketDataUnavailableError as exc:
        logger.error("Piyasa verisi çekilemedi: %s", exc)
        return {"success": False, "error": str(exc), "recorded": 0, "failed": config.TICKERS}

    if df_scan.empty:
        logger.warning("Piyasa taramasından hiçbir hisse verisi üretilemedi.")
        return {"success": False, "recorded": 0, "failed": failed}

    # 3. Seans Tarihini Belirle
    target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 4. Tahminleri Kaydet
    recorded_entries = []
    top_picks = []

    for _, row in df_scan.iterrows():
        raw_ticker = str(row.get("ticker_code", ""))
        ticker = raw_ticker.replace(".IS", "")
        prob = float(row.get("_prob", 0.5))
        signal = 1 if prob >= 0.5 else 0
        entry_price = float(row.get("Son Fiyat (TL)", 0.0))

        if not dry_run:
            entry = tracker.record_prediction(
                date=target_date,
                ticker=ticker,
                signal=signal,
                prob=prob,
                entry_price=entry_price,
                log_path=audit_log_path,
            )
            recorded_entries.append(entry)
        else:
            recorded_entries.append(
                {
                    "date": target_date,
                    "ticker": ticker,
                    "signal": signal,
                    "prob": prob,
                    "entry_price": entry_price,
                }
            )

        if prob >= 0.52:
            top_picks.append(
                {
                    "ticker": ticker,
                    "price": entry_price,
                    "prob_pct": round(prob * 100, 1),
                    "verdict": "YÜKSELİŞ 🚀",
                    "allocation_pct": 10.0,
                }
            )

    top_picks.sort(key=lambda x: x["prob_pct"], reverse=True)
    top_3 = top_picks[:3]

    # 5. E-posta Bülteni Gönderimi (Opsiyonel)
    email_sent = False
    if send_email and top_3 and not dry_run:
        recipient = os.getenv("ALERT_RECIPIENT_EMAIL", "")
        if recipient and os.getenv("SMTP_USER"):
            try:
                html = email_alerts.build_newsletter_html(
                    top_picks=top_3,
                    report_date=target_date,
                    model_version="v2.1-ensemble",
                    model_accuracy=0.5213,
                )
                email_sent = email_alerts.send_email_alert(
                    subject=f"📈 BIST 30 Günlük Fırsat Raporu ({target_date})",
                    html_content=html,
                )
            except Exception as exc:
                logger.warning("E-posta gönderimi sırasında hata oluştu: %s", exc)

    summary = {
        "success": True,
        "date": target_date,
        "recorded_count": len(recorded_entries),
        "failed_tickers": failed,
        "top_picks": top_3,
        "email_sent": email_sent,
        "dry_run": dry_run,
    }
    return summary


def main() -> None:
    """CLI giriş noktası."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="BIST Trend Forecaster - Günlük Otonom Tahmin & Ledger Güncelleyici"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tahminleri hesapla fakat dosyaya veya veritabanına kaydetme",
    )
    parser.add_argument(
        "--no-email",
        action="store_true",
        help="SMTP ayarları olsa dahi e-posta bülteni gönderimini atla",
    )
    args = parser.parse_args()

    result = run_daily_forecast(
        dry_run=args.dry_run,
        send_email=not args.no_email,
    )

    print("\n" + "=" * 50)
    print("📊 GÜNLÜK TAHMİN & LEDGER GÜNCELLEME RAPORU")
    print("=" * 50)
    print(f"Tarih              : {result.get('date', '-')}")
    print(f"Kaydedilen Tahmin  : {result.get('recorded_count', 0)} hisse")
    print(f"Alınamayan Hisseler: {len(result.get('failed_tickers', []))} adet")
    email_status = "Gönderildi ✅" if result.get("email_sent") else "Atlandı / Yapılandırılmamış"
    print(f"E-posta Bildirimi  : {email_status}")

    top_picks = result.get("top_picks", [])
    if top_picks:
        print("\n🎯 Günün En Yüksek Potansiyelli Sinyalleri:")
        for pick in top_picks:
            print(
                f"  - {pick['ticker']:<8}: %{pick['prob_pct']:.1f} Yükseliş "
                f"({pick['price']:.2f} TL)"
            )
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
