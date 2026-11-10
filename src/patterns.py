"""BIST 30 Mum Formasyonları ve Dinamik Destek / Direnç Tespiti Modülü.

Bu modül, fiyat serilerindeki klasik Japon mum formasyonlarını (Çekiç, Yutan Mum,
Doji vb.) ve dinamik destek/direnç kırılımlarını tespit ederek makine öğrenimi
tahminleriyle teknik analizi birleştiren hibrit sinyal teyidi üretir.
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class CandlestickPattern:
    """Tespit edilen tek bir mum formasyonu."""

    name: str
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    label: str  # Türkçe açıklayıcı başlık
    description: str
    strength: float  # 0.0 - 1.0 arası sinyal gücü
    date: str


@dataclass(frozen=True)
class SupportResistance:
    """Dinamik pivot, destek ve direnç seviyeleri."""

    pivot: float
    resistance_1: float
    resistance_2: float
    support_1: float
    support_2: float
    current_price: float
    status: str  # BREAKOUT_BULLISH, BREAKDOWN_BEARISH, TESTING_RESISTANCE vb.
    status_label: str
    distance_to_resistance_pct: float
    distance_to_support_pct: float


def detect_candlestick_patterns(df: pd.DataFrame, lookback: int = 5) -> List[CandlestickPattern]:
    """Fiyat serisinin son `lookback` günündeki mum formasyonlarını tespit eder."""
    if df is None or len(df) < 3:
        return []

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        return []

    patterns: List[CandlestickPattern] = []
    recent = df.tail(lookback + 2).copy().reset_index(drop=True)
    n = len(recent)

    for i in range(1, n):
        curr = recent.iloc[i]
        prev = recent.iloc[i - 1]

        c_open, c_high, c_low, c_close = (
            float(curr["open"]),
            float(curr["high"]),
            float(curr["low"]),
            float(curr["close"]),
        )
        p_open, p_close = (
            float(prev["open"]),
            float(prev["close"]),
        )
        date_str = str(curr.get("Date", f"Bar-{i}"))[:10]

        candle_range = max(c_high - c_low, 1e-6)
        body = abs(c_close - c_open)
        body_pct = body / candle_range
        upper_shadow = c_high - max(c_close, c_open)
        lower_shadow = min(c_close, c_open) - c_low

        # 1. Doji (Kararsızlık Mumu)
        if body_pct <= 0.10:
            patterns.append(
                CandlestickPattern(
                    name="Doji",
                    direction="NEUTRAL",
                    label="⚪ Doji (Kararsızlık)",
                    description="Alıcılar ve satıcılar dengede; olası trend dönüşü.",
                    strength=0.50,
                    date=date_str,
                )
            )

        # 2. Çekiç (Hammer) - Düşüş sonrası boğa dönüşü
        if lower_shadow >= 2.0 * body and upper_shadow <= 0.20 * candle_range and body_pct >= 0.10:
            if c_close > c_open:
                patterns.append(
                    CandlestickPattern(
                        name="Hammer",
                        direction="BULLISH",
                        label="🔨 Çekiç (Boğa Dönüşü)",
                        description="Satıcılar bassa da güçlü alıcılar seansı yukarı taşıdı.",
                        strength=0.80,
                        date=date_str,
                    )
                )

        # 3. Kayan Yıldız / Ters Çekiç (Shooting Star / Inverted Hammer)
        if upper_shadow >= 2.0 * body and lower_shadow <= 0.20 * candle_range and body_pct >= 0.10:
            if c_close < c_open:
                patterns.append(
                    CandlestickPattern(
                        name="Shooting_Star",
                        direction="BEARISH",
                        label="🌠 Kayan Yıldız (Ayı Dönüşü)",
                        description="Günün zirvesinden sert kâr satışı geldi; satış uyarısı.",
                        strength=0.75,
                        date=date_str,
                    )
                )
            else:
                patterns.append(
                    CandlestickPattern(
                        name="Inverted_Hammer",
                        direction="BULLISH",
                        label="🔨 Ters Çekiç (Olası Dip)",
                        description="Dip seviyelerde alım tepkisi ve talep oluşuyor.",
                        strength=0.65,
                        date=date_str,
                    )
                )

        # 4. Boğa Yutan Mum (Bullish Engulfing)
        if (
            p_close < p_open
            and c_close > c_open
            and c_close >= p_open
            and c_open <= p_close
            and body > abs(p_close - p_open)
        ):
            patterns.append(
                CandlestickPattern(
                    name="Bullish_Engulfing",
                    direction="BULLISH",
                    label="🟢 Boğa Yutan Mum (Güçlü Alım)",
                    description="Yeşil mum gövdesi bir önceki kırmızı mumu yutarak talep gösterdi.",
                    strength=0.85,
                    date=date_str,
                )
            )

        # 5. Ayı Yutan Mum (Bearish Engulfing)
        if (
            p_close > p_open
            and c_close < c_open
            and c_close <= p_open
            and c_open >= p_close
            and body > abs(p_close - p_open)
        ):
            patterns.append(
                CandlestickPattern(
                    name="Bearish_Engulfing",
                    direction="BEARISH",
                    label="🔴 Ayı Yutan Mum (Güçlü Satış)",
                    description="Kırmızı mum gövdesi bir önceki yeşil mumu kaplayarak satış kurdu.",
                    strength=0.85,
                    date=date_str,
                )
            )

    return patterns


def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Optional[SupportResistance]:
    """Son dönem fiyat hareketlerine göre klasik Pivot ve Destek/Direnç seviyelerini hesaplar."""
    if df is None or len(df) < 5:
        return None

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(df.columns):
        return None

    recent = df.tail(window)
    high_val = float(recent["high"].max())
    low_val = float(recent["low"].min())
    close_val = float(df["close"].iloc[-1])

    # Standart Floor Pivot Hesaplaması
    pivot = (high_val + low_val + close_val) / 3.0
    r1 = (2.0 * pivot) - low_val
    s1 = (2.0 * pivot) - high_val
    r2 = pivot + (high_val - low_val)
    s2 = pivot - (high_val - low_val)

    # Direnç ve Destek Mesafeleri (%)
    dist_r1 = ((r1 - close_val) / close_val) * 100.0
    dist_s1 = ((close_val - s1) / close_val) * 100.0

    # Kırılım ve Durum Analizi
    if close_val >= r1:
        status = "BREAKOUT_BULLISH"
        status_label = "🚀 Direnç Üstü Kırılım (Bullish Breakout)"
    elif close_val <= s1:
        status = "BREAKDOWN_BEARISH"
        status_label = "⚠️ Destek Altı Kırılım (Bearish Breakdown)"
    elif abs(dist_r1) <= 1.0:
        status = "TESTING_RESISTANCE"
        status_label = "🎯 Direnç Test Ediliyor (Karar Aşaması)"
    elif abs(dist_s1) <= 1.0:
        status = "TESTING_SUPPORT"
        status_label = "🛡️ Destek Test Ediliyor (Tepki Beklentisi)"
    else:
        status = "IN_RANGE"
        status_label = "↔️ Kanal İçi Hareket (Konsolidasyon)"

    return SupportResistance(
        pivot=round(pivot, 2),
        resistance_1=round(r1, 2),
        resistance_2=round(r2, 2),
        support_1=round(s1, 2),
        support_2=round(s2, 2),
        current_price=round(close_val, 2),
        status=status,
        status_label=status_label,
        distance_to_resistance_pct=round(dist_r1, 2),
        distance_to_support_pct=round(dist_s1, 2),
    )


def calculate_relative_strength(
    ticker_df: pd.DataFrame, index_df: pd.DataFrame, window: int = 20
) -> Optional[pd.DataFrame]:
    """Hisse fiyatının BIST 100 (XU100) endeksine karşı Göreceli Gücünü (RS) hesaplar."""
    if ticker_df is None or index_df is None or len(ticker_df) < window or len(index_df) < window:
        return None

    # Tarihe göre birleştir
    t_sub = ticker_df[["Date", "close"]].rename(columns={"close": "ticker_close"})
    i_sub = index_df[["Date", "close"]].rename(columns={"close": "index_close"})

    merged = pd.merge(t_sub, i_sub, on="Date").dropna()
    if len(merged) < window:
        return None

    merged["rs_ratio"] = merged["ticker_close"] / merged["index_close"]
    # 20 günlük Göreceli Güç Momentum Çizgisi (Base 100)
    merged["rs_base100"] = (merged["rs_ratio"] / merged["rs_ratio"].iloc[0]) * 100.0
    merged["rs_sma"] = merged["rs_ratio"].rolling(window=min(window, 10)).mean()
    merged["outperforming"] = merged["rs_ratio"] >= merged["rs_sma"]

    return merged
