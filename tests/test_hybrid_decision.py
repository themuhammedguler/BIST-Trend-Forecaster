"""Hibrit Karar Motoru (Teknik ML + RAG Temel Analiz) birim testleri."""

import pytest

from src import hybrid_decision


def test_hybrid_decision_strong_buy():
    dec = hybrid_decision.evaluate_hybrid_decision(
        ticker="THYAO.IS",
        technical_prob=0.62,
        fundamental_score=0.75,
        prob_high=0.53,
        prob_low=0.47,
    )
    assert dec.ticker == "THYAO"
    assert dec.verdict == "STRONG_BUY"
    assert "GÜÇLÜ AL" in dec.badge
    assert dec.is_high_conviction is True
    assert dec.composite_score > 60.0
    assert dec.alignment == "Yüksek Uyum"
    assert dec.fundamental_score == 0.75


def test_hybrid_decision_value_dip():
    dec = hybrid_decision.evaluate_hybrid_decision(
        ticker="EREGL",
        technical_prob=0.42,
        fundamental_score=0.60,
        prob_high=0.53,
        prob_low=0.47,
    )
    assert dec.ticker == "EREGL"
    assert dec.verdict == "VALUE_DIP"
    assert "DİVERJANS" in dec.badge
    assert dec.is_high_conviction is False
    assert "iskonto" in dec.summary.lower()


def test_hybrid_decision_speculative_momentum():
    dec = hybrid_decision.evaluate_hybrid_decision(
        ticker="SASA",
        technical_prob=0.58,
        fundamental_score=-0.40,
        prob_high=0.53,
        prob_low=0.47,
    )
    assert dec.ticker == "SASA"
    assert dec.verdict == "SPECULATIVE_MOMENTUM"
    assert "SPEKÜLATİF" in dec.badge
    assert "stop-loss" in dec.summary.lower()


def test_hybrid_decision_strong_sell():
    dec = hybrid_decision.evaluate_hybrid_decision(
        ticker="HEKTS.IS",
        technical_prob=0.38,
        fundamental_score=-0.50,
        prob_high=0.53,
        prob_low=0.47,
    )
    assert dec.ticker == "HEKTS"
    assert dec.verdict == "STRONG_SELL"
    assert "GÜÇLÜ SAT" in dec.badge
    assert dec.is_high_conviction is True


def test_hybrid_decision_neutral():
    dec = hybrid_decision.evaluate_hybrid_decision(
        ticker="BIMAS",
        technical_prob=0.51,
        fundamental_score=0.05,
        prob_high=0.53,
        prob_low=0.47,
    )
    assert dec.ticker == "BIMAS"
    assert dec.verdict == "NEUTRAL"
    assert "DENGELİ" in dec.badge or "NÖTR" in dec.badge


@pytest.mark.parametrize(
    "prob, expected_verdict",
    [
        (0.60, "TECHNICAL_UP"),
        (0.40, "TECHNICAL_DOWN"),
        (0.50, "NEUTRAL"),
    ],
)
def test_hybrid_decision_without_fundamental_report(prob, expected_verdict):
    dec = hybrid_decision.evaluate_hybrid_decision(
        ticker="KCHOL.IS",
        technical_prob=prob,
        fundamental_score=None,
        prob_high=0.53,
        prob_low=0.47,
    )
    assert dec.ticker == "KCHOL"
    assert dec.verdict == expected_verdict
    assert dec.fundamental_score is None
    assert dec.composite_score == round(prob * 100, 1)
