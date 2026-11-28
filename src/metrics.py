# src/metrics.py
# Sınıflandırma değerlendirme metrikleri (eğitim ve Optuna için ortak)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y_true, proba, threshold=0.5):
    """
    Yükseliş olasılıklarından doğruluk dışındaki metrikleri de hesaplar.

    Doğruluk tek başına yanıltıcıdır: çoğunluk sınıfını tahmin eden bir model
    yüksek doğruluk alır ama dengeli doğruluğu ve ROC-AUC'si 0.5'te kalır.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    preds = (proba >= threshold).astype(int)

    # Tek sınıflı sette AUC tanımsızdır
    roc_auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "roc_auc": roc_auc,
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "precision": precision_score(y_true, preds, zero_division=0),
    }


def evaluate_extended(y_true, proba, threshold=0.5):
    """Temel 5 metriğe ek olarak Recall (Duyarlılık), F1-Score ve Macro-F1 hesaplar."""
    base = evaluate(y_true, proba, threshold=threshold)
    y_arr = np.asarray(y_true)
    preds = (np.asarray(proba, dtype=float) >= threshold).astype(int)
    base["recall"] = float(recall_score(y_arr, preds, zero_division=0))
    base["f1"] = float(f1_score(y_arr, preds, zero_division=0))
    base["f1_macro"] = float(f1_score(y_arr, preds, average="macro", zero_division=0))
    return base


def class_distribution(y):
    """Düşüş (0) ve yükseliş (1) sınıflarının oranları."""
    y = np.asarray(y)
    if len(y) == 0:
        return {0: 0.0, 1: 0.0}
    return {label: float(np.mean(y == label)) for label in (0, 1)}


def scale_pos_weight(y):
    """
    XGBoost'un scale_pos_weight değeri: düşüş günü sayısı / yükseliş günü sayısı.
    Tek sınıf varsa ağırlıklandırma anlamsızdır, 1.0 döner.
    """
    y = np.asarray(y)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return n_neg / n_pos
