"""Çoklu vade trend tahmin motoru (1 Gün, 1 Hafta, 1 Ay) ve trend konfluans analizi.

Farklı yatırım vadeleri (günlük swing, haftalık dalga ve aylık ana trend) için
yön tahminleri üretir ve vadelerin birbirini doğrulama derecesini (confluence) puanlar.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import ta
import xgboost as xgb

from src import config, features, signals


@dataclass
class HorizonForecast:
    """Tek bir vade için tahmin çıktısını temsil eder."""

    name: str  # "1 Gün", "1 Hafta", "1 Ay"
    days: int  # 1, 5, 20
    probability: float  # 0.0 - 1.0 arası yükseliş olasılığı
    signal: int  # 1 (Al/Yükseliş), 0 (Nötr), -1 (Sat/Düşüş)
    label: str  # "YÜKSELİŞ", "NÖTR", "DÜŞÜŞ"


@dataclass
class MultiHorizonResult:
    """Tüm vadeler için toplu tahmin ve trend uyumu (confluence) özeti."""

    forecasts: Dict[str, HorizonForecast]
    confluence_score: float  # -1.0 (Tam Ayı) ile +1.0 (Tam Boğa) arası
    confluence_label: str  # "GÜÇLÜ BOĞA", "DÜZELTME", "TEPKİ", "AYI", "KARARSIZ"
    summary: str  # Detaylı açıklama metni


HORIZON_CONFIGS = {
    "1d": {"name": "1 Gün (Kısa Vade)", "days": 1},
    "5d": {"name": "1 Hafta (Orta Vade)", "days": 5},
    "20d": {"name": "1 Ay (Uzun Vade)", "days": 20},
}


def add_multi_horizon_targets(
    df: pd.DataFrame,
    close_col: str = "close",
) -> pd.DataFrame:
    """Veri çerçevesine 1 günlük, 5 günlük (1 hafta) ve 20 günlük (1 ay) hedef yönleri ekler.

    target = 1 if close[t + days] > close[t] else 0.
    Gelecek vadeler veri sınırını aştığında NaN olarak işaretlenir.
    """
    if close_col not in df.columns:
        raise ValueError(f"'{close_col}' sütunu veri çerçevesinde bulunamadı.")

    df_out = df.copy()
    close_series = df_out[close_col]

    for key, cfg in HORIZON_CONFIGS.items():
        n = cfg["days"]
        future_close = close_series.shift(-n)
        target_series = (future_close > close_series).astype(float)
        # Sonda kalan n satır için shift NaN üretir; float kalmasını sağlayalım
        target_series[future_close.isna()] = float("nan")
        df_out[f"target_{key}"] = target_series

    return df_out


def compute_confluence(forecasts: Dict[str, HorizonForecast]) -> Tuple[float, str, str]:
    """Farklı vadelerin sinyallerini ve olasılıklarını sentezleyerek konfluans puanı üretir.

    Ağırlıklar:
    - 1 Gün: 0.25 (Taktiksel zamanlama)
    - 1 Hafta: 0.45 (Ana swing dalgası)
    - 1 Ay: 0.30 (Makro trend zemini)

    Döndürülenler:
        (confluence_score, confluence_label, description)
    """
    weights = {"1d": 0.25, "5d": 0.45, "20d": 0.30}

    def _to_num(sig: Any) -> int:
        if isinstance(sig, (int, float)):
            return int(sig)
        if sig == signals.UP:
            return 1
        if sig == signals.DOWN:
            return -1
        return 0

    weighted_score = 0.0
    for key, weight in weights.items():
        if key in forecasts:
            weighted_score += _to_num(forecasts[key].signal) * weight

    sig_1d = _to_num(forecasts.get("1d", HorizonForecast("1d", 1, 0.5, 0, "NÖTR")).signal)
    sig_5d = _to_num(forecasts.get("5d", HorizonForecast("5d", 5, 0.5, 0, "NÖTR")).signal)
    sig_20d = _to_num(forecasts.get("20d", HorizonForecast("20d", 20, 0.5, 0, "NÖTR")).signal)

    # Yorumlama Mantığı
    if sig_1d == 1 and sig_5d == 1 and sig_20d == 1:
        label = "GÜÇLÜ BOĞA (TREND ONAYI)"
        summary = (
            "Kısa, orta ve uzun vadeli tüm modeller eş zamanlı yükseliş teyidi veriyor. "
            "Trend uyumu maksimum seviyede."
        )
    elif sig_1d == -1 and sig_5d == -1 and sig_20d == -1:
        label = "GÜÇLÜ AYI (DÜŞÜŞ ONAYI)"
        summary = (
            "Tüm vadelerde düşüş trendi baskın. Yeni pozisyon açılması yüksek risk "
            "taşımaktadır; nakit korunmalı."
        )
    elif sig_1d == -1 and sig_5d == 1 and sig_20d == 1:
        label = "BOĞA İÇİ SAĞLIKLI DÜZELTME"
        summary = (
            "Orta ve uzun vadeli ana trend yukarı yönlü; 1 günlük düşüş beklentisi "
            "trend içi kâr satışı / düzeltme fırsatı niteliğinde."
        )
    elif sig_1d == 1 and sig_5d == -1 and sig_20d == -1:
        label = "AYI İÇİ TEPKİ YÜKSELİŞİ"
        summary = (
            "Ana trend zayıf ve düşüş yönlü; günlük yükseliş beklentisi geçici tepki "
            "alımı niteliğinde olabilir, stop-loss titizlikle izlenmelidir."
        )
    elif weighted_score > 0.20:
        label = "ILIMLI BOĞA EĞİLİMİ"
        summary = (
            "Modeller genel olarak yukarı yönlü meyilli, ancak vadeler arası kısmi ayrışma var."
        )
    elif weighted_score < -0.20:
        label = "ILIMLI AYI EĞİLİMİ"
        summary = "Modeller genel olarak aşağı yönlü baskı altında, savunmacı strateji önerilir."
    else:
        label = "KARARSIZ / NÖTR TREND"
        summary = "Vadeler birbirini nötrlemekte; belirgin bir yön teyidi bulunmuyor."

    return round(weighted_score, 3), label, summary


def predict_horizons(
    models: Dict[str, Any],
    X: pd.DataFrame,
    prob_low: float = 0.47,
    prob_high: float = 0.53,
    fallback_prob: Optional[float] = None,
) -> MultiHorizonResult:
    """Farklı vadelerdeki modelleri kullanarak çoklu vade tahmin paketini oluşturur.

    Args:
        models: {"1d": model_1d, "5d": model_5d, "20d": model_20d}
        X: Tek satırlık öznitelik DataFrame'i (veya ilk satırı kullanılır)
        prob_low: Nötr / Düşüş eşiği
        prob_high: Nötr / Yükseliş eşiği
        fallback_prob: Model bulunmadığında kullanılacak varsayılan olasılık
    """
    if X.empty:
        raise ValueError("Tahmin yapılacak öznitelik verisi (X) boş olamaz.")

    X_row = X.iloc[[0]]
    forecasts: Dict[str, HorizonForecast] = {}

    for key, cfg in HORIZON_CONFIGS.items():
        model = models.get(key)
        if model is not None:
            if hasattr(model, "predict_proba"):
                aligned_X = features.align_model_features(model, X_row)
                prob = float(model.predict_proba(aligned_X)[0][1])
            elif callable(model):
                prob = float(model(X_row))
            else:
                prob = 0.50
        elif fallback_prob is not None:
            prob = fallback_prob
        else:
            # Model yoksa nötr kabul edilir
            prob = 0.50

        sig_str = signals.classify_signal(prob, prob_low, prob_high)
        sig_val = 1 if sig_str == signals.UP else (-1 if sig_str == signals.DOWN else 0)
        label = {signals.UP: "YÜKSELİŞ", signals.NEUTRAL: "NÖTR", signals.DOWN: "DÜŞÜŞ"}[sig_str]

        forecasts[key] = HorizonForecast(
            name=cfg["name"],
            days=cfg["days"],
            probability=round(prob, 4),
            signal=sig_val,
            label=label,
        )

    confluence_score, confluence_label, summary = compute_confluence(forecasts)

    return MultiHorizonResult(
        forecasts=forecasts,
        confluence_score=confluence_score,
        confluence_label=confluence_label,
        summary=summary,
    )


def calculate_pivot_levels(df: pd.DataFrame) -> Dict[str, float]:
    """Klasik ve Fibonacci pivot destek/direnç seviyelerini hesaplar.

    Formüller:
        Pivot (P) = (High + Low + Close) / 3
        R1 = 2P - Low, R2 = P + (High - Low), R3 = High + 2(P - Low)
        S1 = 2P - High, S2 = P - (High - Low), S3 = Low - 2(High - P)
        Fib %38.2, %50, %61.8
    """
    if df.empty:
        return {}

    # Sütun adları büyük veya küçük harf olabilir
    col_map = {c.lower(): c for c in df.columns}
    h_col = col_map.get("high", "high")
    l_col = col_map.get("low", "low")
    c_col = col_map.get("close", "close")

    last = df.iloc[-1]
    h_val = float(last[h_col])
    l_val = float(last[l_col])
    c_val = float(last[c_col])

    p = (h_val + l_val + c_val) / 3.0
    r1 = 2.0 * p - l_val
    s1 = 2.0 * p - h_val
    r2 = p + (h_val - l_val)
    s2 = p - (h_val - l_val)
    r3 = h_val + 2.0 * (p - l_val)
    s3 = l_val - 2.0 * (h_val - p)

    diff = h_val - l_val if (h_val - l_val) > 0 else 0.01
    fib_382 = h_val - diff * 0.382
    fib_500 = h_val - diff * 0.500
    fib_618 = h_val - diff * 0.618

    return {
        "pivot": round(p, 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "r3": round(r3, 2),
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "s3": round(s3, 2),
        "fib_382": round(fib_382, 2),
        "fib_500": round(fib_500, 2),
        "fib_618": round(fib_618, 2),
    }


def calculate_trend_strength(df: pd.DataFrame) -> Dict[str, Any]:
    """ADX trend gücü ve 50/200 günlük hareketli ortalama kesişimlerini analiz eder."""
    if df.empty or len(df) < 15:
        return {
            "adx": 20.0,
            "trend_strength": "YATAY / BELİRSİZ",
            "sma_50": None,
            "sma_200": None,
            "golden_cross": False,
            "death_cross": False,
        }

    col_map = {c.lower(): c for c in df.columns}
    h_col = col_map.get("high", "high")
    l_col = col_map.get("low", "low")
    c_col = col_map.get("close", "close")

    # ADX hesaplama (en az 28 satır gerektirir)
    adx_val = 20.0
    if len(df) >= 28:
        try:
            adx_series = ta.trend.adx(df[h_col], df[l_col], df[c_col], window=14)
            if not adx_series.empty and not np.isnan(adx_series.iloc[-1]):
                adx_val = float(adx_series.iloc[-1])
        except Exception:
            adx_val = 20.0

    if adx_val >= 25.0:
        strength = "GÜÇLÜ TREND"
    elif adx_val >= 20.0:
        strength = "GELİŞEN TREND"
    else:
        strength = "ZAYIF / TESTERE (YATAY)"

    # Hareketli Ortalamalar
    close_series = df[c_col]
    sma_50 = float(close_series.rolling(50).mean().iloc[-1]) if len(df) >= 50 else None
    sma_200 = float(close_series.rolling(200).mean().iloc[-1]) if len(df) >= 200 else None

    golden_cross = False
    death_cross = False
    if sma_50 is not None and sma_200 is not None:
        golden_cross = sma_50 > sma_200
        death_cross = sma_50 < sma_200

    return {
        "adx": round(adx_val, 1),
        "trend_strength": strength,
        "sma_50": round(sma_50, 2) if sma_50 is not None else None,
        "sma_200": round(sma_200, 2) if sma_200 is not None else None,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
    }


def calculate_expected_price_corridor(
    current_price: float,
    daily_volatility: float,
) -> Dict[str, Dict[str, float]]:
    """Tarihsel oynaklık üzerinden 1 Gün, 1 Hafta (5G) ve 1 Ay (20G)
    %95 güven fiyat koridorunu hesaplar.
    """
    sigma_d = float(daily_volatility)
    # Eğer mutlak TL cinsinden fiyat standart sapması verilmişse normalize et
    if sigma_d > 0.5 and current_price > 0:
        sigma_d = sigma_d / current_price
    # Finansal seriler için gerçekçi günlük oynaklık sınırı (%0.5 ile %6.0 arası)
    sigma_d = float(np.clip(sigma_d, 0.005, 0.06))

    # 1.96 * sigma * sqrt(T)
    d1 = 1.96 * sigma_d * 1.0
    d5 = 1.96 * sigma_d * np.sqrt(5.0)
    d20 = 1.96 * sigma_d * np.sqrt(20.0)

    return {
        "1d": {
            "min": max(0.01, round(current_price * (1.0 - d1), 2)),
            "max": round(current_price * (1.0 + d1), 2),
            "pct": round(d1 * 100, 1),
        },
        "5d": {
            "min": max(0.01, round(current_price * (1.0 - d5), 2)),
            "max": round(current_price * (1.0 + d5), 2),
            "pct": round(d5 * 100, 1),
        },
        "20d": {
            "min": max(0.01, round(current_price * (1.0 - d20), 2)),
            "max": round(current_price * (1.0 + d20), 2),
            "pct": round(d20 * 100, 1),
        },
    }


def calculate_risk_reward(current_price: float, s1: float, r1: float) -> Dict[str, Any]:
    """Destek (S1) ve Direnç (R1) seviyelerine göre Risk / Ödül oranını hesaplar."""
    gain = max(0.01, r1 - current_price)
    risk = max(0.01, current_price - s1)
    ratio = round(gain / risk, 2)

    if ratio >= 1.5:
        label = f"CAZİP (1 : {ratio})"
        favorable = True
    elif ratio >= 1.0:
        label = f"DENGELİ (1 : {ratio})"
        favorable = False
    else:
        label = f"RİSKLİ / DÜŞÜK (1 : {ratio})"
        favorable = False

    return {
        "ratio": ratio,
        "potential_gain_tl": round(gain, 2),
        "potential_risk_tl": round(risk, 2),
        "label": label,
        "is_favorable": favorable,
    }


def _fallback_momentum_estimator(days: int = 5):
    """Eğitilmiş model bulunamadığında kullanılan kalibre edilmiş momentum tahmincisi."""

    def _estimator(X_row: pd.DataFrame) -> float:
        score = 0.50
        if "roc_5" in X_row.columns:
            score += float(np.tanh(X_row["roc_5"].iloc[0] * 8.0) * 0.15)
        if "sma_10_ratio" in X_row.columns:
            score += float(np.tanh(X_row["sma_10_ratio"].iloc[0] * 5.0) * 0.10)
        if "mfi" in X_row.columns:
            score += float((X_row["mfi"].iloc[0] - 50.0) / 100.0 * 0.10)
        return float(np.clip(score, 0.35, 0.65))

    return _estimator


def load_multi_horizon_models() -> Dict[str, Any]:
    """1 Gün, 1 Hafta (5G) ve 1 Ay (20G) yön tahmin modellerini yükler."""
    models: Dict[str, Any] = {}

    # 1 Gün Modeli: xgb_bist_model.json
    try:
        m1 = xgb.XGBClassifier()
        m1.load_model(config.MODEL_PATH)
        models["1d"] = m1
    except Exception:
        pass

    models_dir = os.path.join(config.BASE_DIR, "models")
    path_5d = os.path.join(models_dir, "xgb_5d_model.json")
    path_20d = os.path.join(models_dir, "xgb_20d_model.json")

    # 5 Gün Modeli
    if os.path.exists(path_5d):
        try:
            m5 = xgb.XGBClassifier()
            m5.load_model(path_5d)
            models["5d"] = m5
        except Exception:
            models["5d"] = _fallback_momentum_estimator(5)
    else:
        models["5d"] = _fallback_momentum_estimator(5)

    # 20 Gün Modeli
    if os.path.exists(path_20d):
        try:
            m20 = xgb.XGBClassifier()
            m20.load_model(path_20d)
            models["20d"] = m20
        except Exception:
            models["20d"] = _fallback_momentum_estimator(20)
    else:
        models["20d"] = _fallback_momentum_estimator(20)

    return models
