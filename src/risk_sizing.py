"""Güven skoru ve volatiliteye duyarlı pozisyon boyutlandırma (Kelly Criterion) modülü (#71).

Modelin yükseliş olasılığını (p) ve hissenin anlık oynaklığını (volatilite) dikkate alarak
Fractional Kelly ve Volatility Targeting algoritmalarıyla kaldıraçsız (%0 - %100)
dinamik sermaye tahsisi ve portföy risk dağılımı önerileri üretir.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

DEFAULT_TARGET_VOLATILITY = 0.25
DEFAULT_KELLY_FRACTION = 0.50  # Half-Kelly (daha dengeli sermaye riski)


@dataclass(frozen=True)
class SizingRecommendation:
    """Tek bir hisse için pozisyon boyutlandırma ve risk değerlendirme sonucu."""

    ticker: str
    prob: float
    volatility: float
    position_size: float  # [0.0, 1.0]
    cash_size: float  # 1.0 - position_size
    kelly_raw: float
    vol_multiplier: float
    risk_level: str
    summary_note: str


def calculate_kelly_fraction(
    prob: float,
    win_loss_ratio: float = 1.0,
    fraction: float = DEFAULT_KELLY_FRACTION,
) -> float:
    """Fractional Kelly Kriteri sermaye riski oranını hesaplar.

    Formül: f* = ((p * b) - (1 - p)) / b * fraction
    Burada:
      p: Yükseliş olasılığı
      b: Kâr/zarar oranı (win/loss ratio, varsayılan 1.0)
      fraction: Kelly kesri (örn: 0.50 Half-Kelly, 0.25 Quarter-Kelly)

    Sonuç [0.0, 1.0] aralığına sınırlandırılır (kaldıraçsız).
    """
    if prob <= 0.0 or win_loss_ratio <= 0.0 or fraction <= 0.0:
        return 0.0

    p = min(1.0, float(prob))
    b = float(win_loss_ratio)
    q = 1.0 - p

    # Kelly avantaj (edge) kontrolü
    raw_kelly = (p * b - q) / b
    if raw_kelly <= 0.0:
        return 0.0

    scaled_kelly = raw_kelly * float(fraction)
    return round(float(np.clip(scaled_kelly, 0.0, 1.0)), 4)


def calculate_volatility_multiplier(
    volatility: float,
    target_volatility: float = DEFAULT_TARGET_VOLATILITY,
    max_multiplier: float = 1.0,
) -> float:
    """Hisse oynaklığına göre pozisyon ölçekleme çarpanını hesaplar (Volatility Targeting).

    Formül: w_vol = min(max_multiplier, target_volatility / max(volatility, 0.05))
    Yüksek volatilite dönemlerinde pozisyonu küçültür, düşük volatilitede tavan değere çeker.
    """
    if volatility <= 0.0 or target_volatility <= 0.0:
        return float(max_multiplier)

    vol = max(float(volatility), 0.05)
    target = float(target_volatility)
    multiplier = min(float(max_multiplier), target / vol)
    return round(float(np.clip(multiplier, 0.0, max_multiplier)), 4)


def calculate_position_size(
    prob: float,
    volatility: float = DEFAULT_TARGET_VOLATILITY,
    min_threshold: float = config.PROB_THRESHOLD_HIGH,
    win_loss_ratio: float = 1.0,
    fraction: float = DEFAULT_KELLY_FRACTION,
    target_volatility: float = DEFAULT_TARGET_VOLATILITY,
) -> float:
    """Model olasılığı ve volatiliteye göre birleşik nihai pozisyon boyutunu ([0.0, 1.0]) döner."""
    if prob < min_threshold:
        return 0.0

    kelly = calculate_kelly_fraction(prob, win_loss_ratio=win_loss_ratio, fraction=fraction)
    vol_mult = calculate_volatility_multiplier(volatility, target_volatility=target_volatility)

    combined = kelly * vol_mult
    return round(float(np.clip(combined, 0.0, 1.0)), 4)


def recommend_position_size(
    ticker: str,
    prob: float,
    volatility: float = DEFAULT_TARGET_VOLATILITY,
    min_threshold: float = config.PROB_THRESHOLD_HIGH,
    win_loss_ratio: float = 1.0,
    fraction: float = DEFAULT_KELLY_FRACTION,
    target_volatility: float = DEFAULT_TARGET_VOLATILITY,
) -> SizingRecommendation:
    """Tek bir hisse için yapılandırılmış SizingRecommendation nesnesi üretir."""
    clean_ticker = ticker.replace(".IS", "")
    kelly_raw = calculate_kelly_fraction(prob, win_loss_ratio=win_loss_ratio, fraction=fraction)
    vol_mult = calculate_volatility_multiplier(volatility, target_volatility=target_volatility)
    pos_size = calculate_position_size(
        prob,
        volatility=volatility,
        min_threshold=min_threshold,
        win_loss_ratio=win_loss_ratio,
        fraction=fraction,
        target_volatility=target_volatility,
    )
    cash_size = round(1.0 - pos_size, 4)

    if pos_size >= 0.50:
        risk_level = "🟢 Yüksek Güven / Agresif"
        summary = (
            f"{clean_ticker} için güçlü artış olasılığı (%{prob * 100:.1f}) ve makul oynaklık "
            f"(%{volatility * 100:.1f}) tespit edildi. Sermayenin %{pos_size * 100:.1f}'i hisseye, "
            f"%{cash_size * 100:.1f}'i nakde tahsis edilebilir."
        )
    elif pos_size >= 0.20:
        risk_level = "🟡 Dengeli / Orta Düzey"
        summary = (
            f"{clean_ticker} için dengeli sinyal (%{prob * 100:.1f}). Oynaklık baskısı "
            f"göz önüne alınarak sermayenin %{pos_size * 100:.1f}'i hisseye, "
            f"%{cash_size * 100:.1f}'i nakde tahsis edilmesi önerilir."
        )
    elif pos_size > 0.0:
        risk_level = "🟠 Muhafazakar / Küçük Pozisyon"
        summary = (
            f"{clean_ticker} sinyali pozitif ancak risk marjı dar (%{prob * 100:.1f}). "
            f"Savunmacı yaklaşımla sermayenin %{pos_size * 100:.1f}'i hisseye ayrılmalıdır."
        )
    else:
        risk_level = "⚪ Nakit / Giriş Yok"
        summary = (
            f"{clean_ticker} için model olasılığı (%{prob * 100:.1f}) minimum güven eşiğinin "
            f"(%{min_threshold * 100:.0f}) altında olduğundan pozisyon alınmamalı, "
            "nakitte kalınmalıdır."
        )

    return SizingRecommendation(
        ticker=clean_ticker,
        prob=round(prob, 4),
        volatility=round(volatility, 4),
        position_size=pos_size,
        cash_size=cash_size,
        kelly_raw=kelly_raw,
        vol_multiplier=vol_mult,
        risk_level=risk_level,
        summary_note=summary,
    )


def calculate_portfolio_allocation(
    df_scan: pd.DataFrame,
    max_tickers: int = 5,
    min_threshold: float = config.PROB_THRESHOLD_HIGH,
    fraction: float = DEFAULT_KELLY_FRACTION,
    target_volatility: float = DEFAULT_TARGET_VOLATILITY,
) -> pd.DataFrame:
    """Tarayıcı sonuçlarından en yüksek fırsat vadeden hisseler arasında
    normalize sermaye dağılımı üretir.
    """
    if df_scan is None or df_scan.empty:
        return pd.DataFrame(
            columns=[
                "Hisse",
                "Sektör",
                "Son Fiyat (TL)",
                "Yükseliş Olasılığı (%)",
                "Önerilen Tahsis (%)",
                "Risk Profili",
            ]
        )

    prob_col = "_prob" if "_prob" in df_scan.columns else "Yükseliş Olasılığı (%)"
    filtered = df_scan.copy()
    if prob_col == "Yükseliş Olasılığı (%)":
        filtered["_p"] = filtered[prob_col] / 100.0
    else:
        filtered["_p"] = filtered[prob_col]

    bulls = filtered[filtered["_p"] >= min_threshold].sort_values(by="_p", ascending=False)
    if bulls.empty:
        return pd.DataFrame(
            columns=[
                "Hisse",
                "Sektör",
                "Son Fiyat (TL)",
                "Yükseliş Olasılığı (%)",
                "Önerilen Tahsis (%)",
                "Risk Profili",
            ]
        )

    top_assets = bulls.head(max_tickers).copy()
    raw_sizes = []
    risk_profiles = []

    for _, row in top_assets.iterrows():
        p = float(row["_p"])
        vol = float(row.get("volatility", row.get("_volatility", DEFAULT_TARGET_VOLATILITY)))
        rec = recommend_position_size(
            ticker=str(row["Hisse"]),
            prob=p,
            volatility=vol,
            min_threshold=min_threshold,
            fraction=fraction,
            target_volatility=target_volatility,
        )
        raw_sizes.append(rec.position_size)
        risk_profiles.append(rec.risk_level)

    total_raw = sum(raw_sizes)
    # Eğer toplam boyut %100'ü aşarsa oransal olarak normalize et
    if total_raw > 1.0:
        norm_sizes = [round((s / total_raw) * 100.0, 1) for s in raw_sizes]
    else:
        norm_sizes = [round(s * 100.0, 1) for s in raw_sizes]

    result = pd.DataFrame(
        {
            "Hisse": top_assets["Hisse"].values,
            "Sektör": top_assets["Sektör"].values if "Sektör" in top_assets.columns else "-",
            "Son Fiyat (TL)": (
                top_assets["Son Fiyat (TL)"].values
                if "Son Fiyat (TL)" in top_assets.columns
                else 0.0
            ),
            "Yükseliş Olasılığı (%)": [round(float(p) * 100.0, 1) for p in top_assets["_p"]],
            "Önerilen Tahsis (%)": norm_sizes,
            "Risk Profili": risk_profiles,
        }
    )
    return result


def format_allocation_as_markdown(df_alloc: pd.DataFrame) -> str:
    """Sermaye dağılım tablosunu harici kütüphane gerektirmeden Markdown tablosuna dönüştürür."""
    if df_alloc.empty:
        return "Dağılım için uygun hisse bulunamadı."
    headers = list(df_alloc.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df_alloc.iterrows():
        rows.append("| " + " | ".join(str(val) for val in row.values) + " |")
    return "\n".join([header_line, separator_line] + rows)
