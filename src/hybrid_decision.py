"""Teknik ML ve RAG temel analiz modellerini birleştiren Hibrit Karar Motoru.

XGBoost yön tahmin olasılığı ile LLM/RAG tabanlı KAP bilanço duygu skorunu
sentezleyerek yatırımcıya tek ve net bir kurumsal karar rozeti sunar.
"""

from __future__ import annotations

from dataclasses import dataclass

from src import config


@dataclass(frozen=True)
class HybridDecision:
    """Teknik ve temel analiz sentezi karar çıktısı."""

    ticker: str
    verdict: str
    badge: str
    composite_score: float  # 0.0 - 100.0
    summary: str
    alignment: str
    technical_prob: float
    fundamental_score: float | None

    @property
    def is_high_conviction(self) -> bool:
        """Kararın yüksek güvenilirlikli (Strong Buy / Strong Sell) olup olmadığını belirtir."""
        return self.verdict in ("STRONG_BUY", "STRONG_SELL")


def evaluate_hybrid_decision(
    ticker: str,
    technical_prob: float,
    fundamental_score: float | None,
    prob_high: float = config.PROB_THRESHOLD_HIGH,
    prob_low: float = config.PROB_THRESHOLD_LOW,
) -> HybridDecision:
    """Teknik olasılık ve temel analiz duygu skorunu sentezler."""
    clean_ticker = ticker.upper().replace(".IS", "").strip()

    # 1. Senaryo: Temel analiz raporu henüz üretilmemiş veya yok
    if fundamental_score is None:
        raw_score = round(float(technical_prob) * 100, 1)
        if technical_prob >= prob_high:
            return HybridDecision(
                ticker=clean_ticker,
                verdict="TECHNICAL_UP",
                badge="📈 TEKNİK YÜKSELİŞ (Temel Rapor Yok)",
                composite_score=raw_score,
                summary=(
                    "Model teknik göstergelere göre yükseliş bekliyor; ancak RAG "
                    "temel analiz raporu henüz bulunmuyor."
                ),
                alignment="Yalnızca Teknik",
                technical_prob=technical_prob,
                fundamental_score=None,
            )
        if technical_prob <= prob_low:
            return HybridDecision(
                ticker=clean_ticker,
                verdict="TECHNICAL_DOWN",
                badge="📉 TEKNİK DÜŞÜŞ (Temel Rapor Yok)",
                composite_score=raw_score,
                summary=(
                    "Model teknik göstergelere göre zayıf / düşüş trendi bekliyor; "
                    "temel analiz raporu yok."
                ),
                alignment="Yalnızca Teknik",
                technical_prob=technical_prob,
                fundamental_score=None,
            )
        return HybridDecision(
            ticker=clean_ticker,
            verdict="NEUTRAL",
            badge="⏸️ NÖTR TEKNİK",
            composite_score=raw_score,
            summary="Teknik göstergeler yön konusunda nötr aralıkta seyrediyor.",
            alignment="Yalnızca Teknik",
            technical_prob=technical_prob,
            fundamental_score=None,
        )

    # 2. Senaryo: Hem teknik hem de temel analiz mevcut
    # fundamental_score: -1.0 ile +1.0 arasında -> 0.0 ile 1.0 arasına normalize edilir
    f_norm = max(0.0, min(1.0, (float(fundamental_score) + 1.0) / 2.0))
    # Ağırlıklı kompozit skor (%55 teknik, %45 temel analiz)
    composite = round((0.55 * float(technical_prob) + 0.45 * f_norm) * 100, 1)

    # Kural 1: Güçlü Al (Yüksek Uyum)
    if technical_prob >= prob_high and fundamental_score >= 0.20:
        return HybridDecision(
            ticker=clean_ticker,
            verdict="STRONG_BUY",
            badge="🚀 GÜÇLÜ AL / YÜKSEK UYUM",
            composite_score=composite,
            summary=(
                "Hem makine öğrenmesi teknik göstergeleri yükseliş sinyali veriyor "
                "hem de KAP bilanço analizi güçlü kârlılık teyidi sağlıyor."
            ),
            alignment="Yüksek Uyum",
            technical_prob=technical_prob,
            fundamental_score=fundamental_score,
        )

    # Kural 2: Fırsat / Değerleme Diverjansı (Temel çok güçlü, teknik düzeltmede)
    if technical_prob <= prob_low and fundamental_score >= 0.40:
        return HybridDecision(
            ticker=clean_ticker,
            verdict="VALUE_DIP",
            badge="⚖️ FIRSAT / DİVERJANS",
            composite_score=composite,
            summary=(
                "Şirketin bilançosu ve kârlılığı oldukça pozitif; ancak hisse fiyatta "
                "teknik düzeltmede veya iskonto bölgesinde."
            ),
            alignment="Diverjans (Aykırılık)",
            technical_prob=technical_prob,
            fundamental_score=fundamental_score,
        )

    # Kural 3: Spekülatif Momentum (Grafik yükseliyor ama bilanço zayıf)
    if technical_prob >= 0.55 and fundamental_score <= -0.20:
        return HybridDecision(
            ticker=clean_ticker,
            verdict="SPECULATIVE_MOMENTUM",
            badge="⚡ SPEKÜLATİF MOMENTUM",
            composite_score=composite,
            summary=(
                "Kısa vadeli teknik grafik yukarı yönlü hareket etse de, KAP ve bilanço "
                "göstergeleri negatif. Stop-loss kullanımı kritiktir."
            ),
            alignment="Zayıf Bilanço Riski",
            technical_prob=technical_prob,
            fundamental_score=fundamental_score,
        )

    # Kural 4: Güçlü Sat (Çifte Risk)
    if technical_prob <= prob_low and fundamental_score <= -0.20:
        return HybridDecision(
            ticker=clean_ticker,
            verdict="STRONG_SELL",
            badge="🔻 GÜÇLÜ SAT / ÇİFTE RİSK",
            composite_score=composite,
            summary=(
                "Hem teknik analiz düşüş eğiliminde hem de şirket temel göstergeleri "
                "zayıflıyor. Sermaye koruması için nakit tercih edilmelidir."
            ),
            alignment="Çifte Risk",
            technical_prob=technical_prob,
            fundamental_score=fundamental_score,
        )

    # Kural 5: Nötr / Dengeli
    return HybridDecision(
        ticker=clean_ticker,
        verdict="NEUTRAL",
        badge="⏸️ NÖTR / DENGELİ",
        composite_score=composite,
        summary=(
            "Teknik ve temel göstergeler arasında belirgin bir tek yönlü uzlaşma yok; "
            "piyasa yatay veya dengeli seyrediyor."
        ),
        alignment="Dengeli",
        technical_prob=technical_prob,
        fundamental_score=fundamental_score,
    )
