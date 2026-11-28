# tests/test_tracker.py
"""Sinyal Takip ve Tahmin Gerçekleşme Defteri birim testleri (#53)."""

import pandas as pd
import pytest

from src import tracker


def test_record_and_load_prediction_lifecycle(tmp_path):
    """Tahminlerin JSON kütüğüne kaydedilmesi, okunması ve mükerrer güncellenmesi."""
    log_file = str(tmp_path / "test_audit.json")

    # İlk kayıt
    entry = tracker.record_prediction(
        date="2025-03-01",
        ticker="THYAO",
        signal=1,
        prob=0.68,
        entry_price=300.0,
        log_path=log_file,
    )
    assert entry["ticker"] == "THYAO"
    assert entry["direction"] == "YÜKSELİŞ"

    # Okuma
    loaded = tracker.load_predictions(log_file)
    assert len(loaded) == 1
    assert loaded[0]["prob"] == 0.68

    # Aynı gün aynı hisseye güncelleme (mükerrer kayıt olmamalı)
    tracker.record_prediction(
        date="2025-03-01",
        ticker="THYAO",
        signal=1,
        prob=0.72,
        entry_price=302.0,
        log_path=log_file,
    )
    reloaded = tracker.load_predictions(log_file)
    assert len(reloaded) == 1
    assert reloaded[0]["prob"] == 0.72
    assert reloaded[0]["entry_price"] == 302.0


def test_load_predictions_missing_and_corrupt_files(tmp_path):
    """Mevcut olmayan veya bozuk dosyalarda boş liste dönmelidir."""
    missing = str(tmp_path / "non_existent.json")
    assert tracker.load_predictions(missing) == []

    corrupt = str(tmp_path / "corrupt.json")
    with open(corrupt, "w", encoding="utf-8") as f:
        f.write("{invalid json content...")
    assert tracker.load_predictions(corrupt) == []


def test_evaluate_realization_and_horizons():
    """T+1, T+3 ve T+5 gerçekleşen fiyat ve 100k TL bakiye hesaplamaları doğrulanmalıdır."""
    dates = pd.bdate_range("2025-01-01", periods=10).strftime("%Y-%m-%d")
    # Fiyatlar: 100'den başlayıp her gün 5 TL artıyor
    prices = [100.0 + i * 5.0 for i in range(10)]
    price_df = pd.DataFrame({"Date": dates, "close": prices})

    predictions = [
        # 0. gün (T=0, P0=100 TL): Yükseliş tahmini
        {"date": dates[0], "ticker": "AKBNK", "signal": 1, "prob": 0.65, "entry_price": 100.0},
        # 8. gün (T=8, P8=140 TL): Sadece T+1 gün gerçekleşebilir, T+3 ve T+5 henüz gelmedi
        {"date": dates[8], "ticker": "AKBNK", "signal": 0, "prob": 0.40, "entry_price": 140.0},
    ]

    res_df = tracker.evaluate_realization(predictions, price_df, horizons=(1, 3, 5))
    assert len(res_df) == 2

    row0 = res_df.iloc[0]
    # T+1 fiyat 105 TL (+%5)
    assert row0["1G Fiyat (TL)"] == 105.0
    assert row0["1G Getiri (%)"] == 5.0
    assert row0["1G İsabet"] == "✅ Başarılı"
    assert row0["1G 100k TL Bakiye"] == 105_000.0

    # T+5 (1 hafta) fiyat 125 TL (+%25)
    assert row0["1H (5G) Fiyat (TL)"] == 125.0
    assert row0["1H (5G) Getiri (%)"] == 25.0
    assert row0["1H (5G) 100k TL Bakiye"] == 125_000.0

    # row1 (8. gün): 1G gerçekleşti (P9=145 TL, ret>0 ama tahmin düşüştü -> yanılgı)
    row1 = res_df.iloc[1]
    assert row1["1G İsabet"] == "❌ Yanılgı"
    assert row1["3G İsabet"] == "⏳ Bekleniyor"
    assert row1["1H (5G) İsabet"] == "⏳ Bekleniyor"
    assert pd.isna(row1["1H (5G) Fiyat (TL)"])


def test_summarize_performance():
    """Özet başarı oranı ve getiri metrikleri doğru hesaplanmalıdır."""
    evaluated_data = pd.DataFrame(
        {
            "1G İsabet": ["✅ Başarılı", "✅ Başarılı", "❌ Yanılgı", "⏳ Bekleniyor"],
            "1G Getiri (%)": [4.0, 2.0, -1.0, 0.0],
            "1H (5G) İsabet": ["✅ Başarılı", "❌ Yanılgı", "⏳ Bekleniyor", "⏳ Bekleniyor"],
            "1H (5G) Getiri (%)": [10.0, -5.0, 0.0, 0.0],
        }
    )

    summary = tracker.summarize_performance(evaluated_data, horizons=(1, 5))
    assert summary["total_evaluated"] == 4

    h1 = summary["horizons"]["1G"]
    assert h1["completed_count"] == 3
    # 2 başarılı / 3 tamamlanan = %66.67
    assert h1["hit_rate_pct"] == pytest.approx(66.67, abs=0.01)
    # (4 + 2 - 1) / 3 = 1.67%
    assert h1["avg_return_pct"] == pytest.approx(1.67, abs=0.01)

    h5 = summary["horizons"]["1H (5G)"]
    assert h5["completed_count"] == 2
    assert h5["hit_rate_pct"] == 50.0


def test_get_ticker_audit_snapshot(tmp_path):
    log_file = str(tmp_path / "snap_audit.json")
    dates = pd.bdate_range("2025-01-01", periods=5).strftime("%Y-%m-%d")
    prices = [100.0, 104.0, 108.0, 110.0, 112.0]
    price_df = pd.DataFrame({"Date": dates, "close": prices})

    tracker.record_prediction(
        date=dates[0],
        ticker="THYAO",
        signal=1,
        prob=0.62,
        entry_price=100.0,
        log_path=log_file,
    )
    snap = tracker.get_ticker_audit_snapshot("THYAO.IS", price_frame=price_df, log_path=log_file)
    assert snap["ticker"] == "THYAO"
    assert snap["total_records"] == 1
    assert snap["completed_1d"] == 1
    assert snap["hit_rate_1d_pct"] == 100.0
    assert snap["avg_return_1d_pct"] == 4.0
