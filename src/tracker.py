# src/tracker.py
"""İleriye Dönük Sinyal Takip ve Tahmin Gerçekleşme Defteri (#53).

Modelin ürettiği her günün tahminlerini (tarih, hisse, sinyal, olasılık, giriş fiyatı)
yerel bir JSON günlüğüne kaydeder ve 1 gün, 3 gün ve 1 hafta (5 işlem günü) sonraki
gerçekleşen piyasa fiyatlarıyla eşleştirerek başarı karnesini ve somut getirisini hesaplar.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

DEFAULT_AUDIT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "prediction_audit.json",
)


def record_prediction(
    date: str,
    ticker: str,
    signal: int,
    prob: float,
    entry_price: float,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Modelin ürettiği günlük tahmini denetim günlüğüne (audit log) kaydeder."""
    target_path = log_path or DEFAULT_AUDIT_LOG_PATH
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    records = load_predictions(target_path)

    clean_date = str(date).split()[0]
    record_id = f"{ticker}_{clean_date}"

    # Aynı gün ve hisse için mükerrer kaydı önle (en günceli güncelle)
    direction_label = "YÜKSELİŞ" if signal == 1 else "DÜŞÜŞ"
    new_entry = {
        "id": record_id,
        "date": clean_date,
        "ticker": ticker,
        "signal": int(signal),
        "direction": direction_label,
        "prob": round(float(prob), 4),
        "entry_price": round(float(entry_price), 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    records = [r for r in records if r.get("id") != record_id]
    records.append(new_entry)
    records.sort(key=lambda r: (r.get("date", ""), r.get("ticker", "")))

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return new_entry


def load_predictions(log_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Kayıtlı tahmin günlüğünü okur. Dosya yoksa veya bozuksa boş liste döner."""
    target_path = log_path or DEFAULT_AUDIT_LOG_PATH
    if not os.path.exists(target_path):
        return []
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def evaluate_realization(
    predictions: Sequence[Dict[str, Any]],
    price_frame: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 5),
    hypothetical_capital: float = 100_000.0,
) -> pd.DataFrame:
    """Kayıtlı tahminlerin 1 gün, 3 gün ve 1 hafta (5 seans) sonraki gerçekleşmelerini hesaplar."""
    if not predictions or price_frame is None or price_frame.empty:
        return pd.DataFrame()

    df_prices = price_frame.copy()
    if "Date" in df_prices.columns:
        df_prices["Date"] = pd.to_datetime(df_prices["Date"]).dt.strftime("%Y-%m-%d")
        df_prices = df_prices.sort_values(by="Date").reset_index(drop=True)
    else:
        return pd.DataFrame()

    date_to_idx = {d: i for i, d in enumerate(df_prices["Date"])}
    n_prices = len(df_prices)

    results = []
    for pred in predictions:
        pred_date = str(pred.get("date", "")).split()[0]
        ticker = pred.get("ticker", "")
        signal = int(pred.get("signal", 1))
        p0 = float(pred.get("entry_price", 0.0))

        if pred_date not in date_to_idx or p0 <= 0:
            continue

        idx = date_to_idx[pred_date]
        row_data: Dict[str, Any] = {
            "Tarih": pred_date,
            "Hisse": ticker,
            "Tahmin": pred.get("direction", "YÜKSELİŞ" if signal == 1 else "DÜŞÜŞ"),
            "Olasılık": pred.get("prob", 0.5),
            "Giriş Fiyatı (TL)": p0,
        }

        for h in horizons:
            target_idx = idx + h
            h_label = f"{h}G" if h != 5 else "1H (5G)"
            if target_idx < n_prices:
                ph = float(df_prices.loc[target_idx, "close"])
                ret = (ph - p0) / p0
                is_correct = (ret > 0) if signal == 1 else (ret < 0)
                val_100k = hypothetical_capital * (1.0 + ret)

                row_data[f"{h_label} Fiyat (TL)"] = round(ph, 2)
                row_data[f"{h_label} Getiri (%)"] = round(ret * 100.0, 2)
                row_data[f"{h_label} İsabet"] = "✅ Başarılı" if is_correct else "❌ Yanılgı"
                row_data[f"{h_label} 100k TL Bakiye"] = round(val_100k, 2)
            else:
                row_data[f"{h_label} Fiyat (TL)"] = None
                row_data[f"{h_label} Getiri (%)"] = None
                row_data[f"{h_label} İsabet"] = "⏳ Bekleniyor"
                row_data[f"{h_label} 100k TL Bakiye"] = None

        results.append(row_data)

    return pd.DataFrame(results)


def summarize_performance(
    evaluated_df: pd.DataFrame, horizons: Sequence[int] = (1, 3, 5)
) -> Dict[str, Any]:
    """Gerçekleşme tablosundan vadelere göre isabet oranları ve getiri özetini üretir."""
    if evaluated_df is None or evaluated_df.empty:
        return {
            "total_evaluated": 0,
            "horizons": {},
        }

    total = len(evaluated_df)
    summary: Dict[str, Any] = {
        "total_evaluated": total,
        "horizons": {},
    }

    for h in horizons:
        h_label = f"{h}G" if h != 5 else "1H (5G)"
        col_hit = f"{h_label} İsabet"
        col_ret = f"{h_label} Getiri (%)"

        if col_hit in evaluated_df.columns:
            completed = evaluated_df[evaluated_df[col_hit].isin(["✅ Başarılı", "❌ Yanılgı"])]
            completed_count = len(completed)
            if completed_count > 0:
                hits = (completed[col_hit] == "✅ Başarılı").sum()
                hit_rate = float(hits / completed_count) * 100.0
                avg_ret = float(completed[col_ret].mean())
            else:
                hit_rate = 0.0
                avg_ret = 0.0

            summary["horizons"][h_label] = {
                "completed_count": completed_count,
                "hit_rate_pct": round(hit_rate, 2),
                "avg_return_pct": round(avg_ret, 2),
            }

    return summary


def get_ticker_audit_snapshot(
    ticker: str,
    price_frame: Optional[pd.DataFrame] = None,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Seçili hisse için denetim defterindeki geçmiş tahmin sayısını ve isabet özetini döner."""
    clean = ticker.upper().replace(".IS", "").strip()
    all_preds = load_predictions(log_path)
    ticker_preds = [p for p in all_preds if str(p.get("ticker", "")).upper() == clean]

    snapshot: Dict[str, Any] = {
        "ticker": clean,
        "total_records": len(ticker_preds),
        "completed_1d": 0,
        "hit_rate_1d_pct": None,
        "avg_return_1d_pct": None,
    }
    if not ticker_preds or price_frame is None or price_frame.empty:
        return snapshot

    eval_df = evaluate_realization(ticker_preds, price_frame, horizons=(1, 3, 5))
    if eval_df.empty:
        return snapshot

    perf = summarize_performance(eval_df, horizons=(1, 3, 5))
    h1 = perf.get("horizons", {}).get("1G", {})
    completed = int(h1.get("completed_count", 0))
    snapshot["completed_1d"] = completed
    if completed > 0:
        snapshot["hit_rate_1d_pct"] = float(h1.get("hit_rate_pct", 0.0))
        snapshot["avg_return_1d_pct"] = float(h1.get("avg_return_pct", 0.0))
    return snapshot
