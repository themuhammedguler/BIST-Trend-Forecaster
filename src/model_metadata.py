# src/model_metadata.py
"""Model meta-veri kayıt ve okuma mekanizması (#13).

Modelin eğitim tarihi, veri kesim noktası, satır sayıları, test metrikleri,
hiperparametreler ve öznitelik listesini JSON formatında saklar ve yükler.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def build_metadata(
    cutoff_date: str,
    train_rows: int,
    test_rows: int,
    test_accuracy: float,
    metrics: Optional[Dict[str, float]] = None,
    features: Optional[list] = None,
    best_params: Optional[Dict[str, Any]] = None,
    version: str = "1.0",
    trained_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Model metaveri sözlüğünü standart şemaya uygun olarak oluşturur."""
    if trained_at is None:
        trained_at = datetime.now(timezone.utc).isoformat()

    return {
        "version": version,
        "trained_at": trained_at,
        "data_cutoff_date": str(cutoff_date),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "test_accuracy": round(float(test_accuracy), 4),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()} if metrics else {},
        "features": list(features) if features else [],
        "best_params": best_params or {},
    }


def save_metadata(metadata: Dict[str, Any], path: str) -> str:
    """Metaveri sözlüğünü JSON dosyası olarak diske yazar."""
    target_path = path
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return target_path


def load_metadata(path: str) -> Optional[Dict[str, Any]]:
    """Verilen yoldan model metaverisini yükler.
    Dosya yoksa veya bozuksa None döner."""
    target_path = path
    if not os.path.exists(target_path):
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def format_metadata_summary(metadata: Optional[Dict[str, Any]]) -> str:
    """Arayüzde gösterilmek üzere okunabilir özet üretir."""
    if not metadata:
        return "Model Sürümü: v1.0 (Meta-veri bulunamadı)"
    ver = metadata.get("version", "1.0")
    acc = metadata.get("test_accuracy")
    cutoff = metadata.get("data_cutoff_date", "-")
    acc_str = f"%{acc * 100:.1f}" if acc is not None else "-"
    return f"Model v{ver} | Son Eğitim Kesimi: {cutoff} | Test Doğruluğu: {acc_str}"
