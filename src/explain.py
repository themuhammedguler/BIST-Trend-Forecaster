# src/explain.py - XGBoost TreeSHAP katkılarıyla tek bir tahminin yerel açıklaması
import numpy as np
import pandas as pd
import xgboost as xgb

# Arayüzde gösterilecek okunabilir öznitelik adları
FEATURE_LABELS = {
    "rsi": "RSI",
    "macd": "MACD",
    "sma_10": "10 Günlük Ortalama",
    "sma_50": "50 Günlük Ortalama",
    "bb_width": "Bollinger Bant Genişliği",
    "volatility": "Volatilite",
    "lag_1_ret": "Dünkü Getiri",
    "lag_2_ret": "Önceki Günün Getirisi",
    "vol_change": "Hacim Değişimi",
    "day_of_week": "Haftanın Günü",
    "month": "Ay",
    "xu100_ret": "BIST 100 Getirisi",
    "rel_strength_bist": "BIST 100 Göreceli Güç",
    "usdtry_change": "USD/TRY Değişimi",
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def explain_prediction(model, X_row):
    """Tek satırlık bir tahmini öznitelik katkılarına ayırır.

    XGBoost'un yerleşik TreeSHAP'i (pred_contribs) log-odds uzayında
    bias + sum(katkılar) == model çıktısı eşitliğini sağlar.

    Dönüş: (explanation, base_prob)
      explanation: |katkı|'ya göre azalan sıralı DataFrame
        - feature: öznitelik adı
        - value: öznitelik değeri
        - contribution: log-odds katkısı (SHAP değeri)
        - impact: olasılık uzayındaki etki; en büyük katkıdan başlayarak
          sırayla uygulanır, böylece base_prob + sum(impact) == tahmin olasılığı
      base_prob: hiçbir öznitelik bilinmediğinde modelin beklenen olasılığı
    """
    if len(X_row) != 1:
        raise ValueError(f"Tek bir satır bekleniyordu, {len(X_row)} satır verildi.")

    contribs = model.get_booster().predict(xgb.DMatrix(X_row), pred_contribs=True)[0]
    bias = contribs[-1]

    explanation = pd.DataFrame(
        {
            "feature": list(X_row.columns),
            "value": X_row.iloc[0].to_numpy(dtype=float),
            "contribution": contribs[:-1],
        }
    )
    order = explanation["contribution"].abs().sort_values(ascending=False, kind="stable").index
    explanation = explanation.loc[order].reset_index(drop=True)

    cumulative = bias + explanation["contribution"].cumsum().to_numpy()
    previous = np.concatenate([[bias], cumulative[:-1]])
    explanation["impact"] = _sigmoid(cumulative) - _sigmoid(previous)

    return explanation, float(_sigmoid(bias))


def top_drivers(explanation, n=3):
    """Olasılığı en çok yukarı ve aşağı iten ilk n özniteliği döndürür: (up, down)."""
    up = explanation[explanation["impact"] > 0].nlargest(n, "impact")
    down = explanation[explanation["impact"] < 0].nsmallest(n, "impact")
    return up, down


def _format_drivers(drivers):
    return ", ".join(
        f"{FEATURE_LABELS.get(f, f)} ({'+' if i > 0 else '-'}%{abs(i) * 100:.1f})"
        for f, i in zip(drivers["feature"], drivers["impact"])
    )


def summarize_drivers(explanation, ticker, n=3):
    """Tahmini en çok etkileyen faktörleri tek cümlelik bir özetle anlatır."""
    up, down = top_drivers(explanation, n)
    up_text = (
        f"yukarı taşıyan ana faktörler: {_format_drivers(up)}"
        if not up.empty
        else "yukarı taşıyan belirgin bir faktör yok"
    )
    down_text = (
        f"aşağı çeken: {_format_drivers(down)}"
        if not down.empty
        else "aşağı çeken belirgin bir faktör yok"
    )
    return f"{ticker} tahminini {up_text}; {down_text}."
