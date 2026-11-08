"""tests/test_risk_sizing.py - Kelly Kriteri ve pozisyon boyutlandırma testleri (#71)."""

import pandas as pd
import pytest

from src import backtest, risk_sizing


def test_calculate_kelly_fraction_basic():
    # p=0.50 -> raw kelly = 0 -> 0.0
    assert risk_sizing.calculate_kelly_fraction(0.50, win_loss_ratio=1.0, fraction=0.5) == 0.0
    # p=0.60 -> raw kelly = (0.60 - 0.40) = 0.20 -> half kelly = 0.10
    assert risk_sizing.calculate_kelly_fraction(0.60, win_loss_ratio=1.0, fraction=0.5) == 0.10
    # p=0.70 -> raw kelly = (0.70 - 0.30) = 0.40 -> half kelly = 0.20
    assert risk_sizing.calculate_kelly_fraction(0.70, win_loss_ratio=1.0, fraction=0.5) == 0.20
    # p=0.40 -> negative edge -> 0.0
    assert risk_sizing.calculate_kelly_fraction(0.40, win_loss_ratio=1.0, fraction=0.5) == 0.0


def test_calculate_kelly_fraction_boundaries():
    assert risk_sizing.calculate_kelly_fraction(0.0) == 0.0
    assert risk_sizing.calculate_kelly_fraction(-0.1) == 0.0
    assert risk_sizing.calculate_kelly_fraction(0.6, fraction=0.0) == 0.0
    assert risk_sizing.calculate_kelly_fraction(1.0, win_loss_ratio=1.0, fraction=1.0) == 1.0


def test_calculate_volatility_multiplier():
    # Hedef 0.25 iken hisse 0.50 ise çarpan = 0.50
    assert risk_sizing.calculate_volatility_multiplier(0.50, target_volatility=0.25) == 0.50
    # Hedef 0.25 iken hisse 0.25 ise çarpan = 1.0
    assert risk_sizing.calculate_volatility_multiplier(0.25, target_volatility=0.25) == 1.0
    # Hedef 0.25 iken hisse 0.10 ise max_multiplier 1.0'a kırpılır (kaldıraçsız)
    assert risk_sizing.calculate_volatility_multiplier(0.10, target_volatility=0.25) == 1.0
    # Negatif veya sıfır volatilite durumunda tavan değer döner
    assert risk_sizing.calculate_volatility_multiplier(0.0, target_volatility=0.25) == 1.0


def test_calculate_position_size():
    # Eşik altı -> 0.0
    assert risk_sizing.calculate_position_size(0.51, min_threshold=0.53) == 0.0
    # Eşik üstü: p=0.65, vol=0.25 -> kelly=(0.65-0.35)*0.5 = 0.15, vol_mult=1.0 -> 0.15
    size = risk_sizing.calculate_position_size(
        0.65,
        volatility=0.25,
        min_threshold=0.53,
        win_loss_ratio=1.0,
        fraction=0.5,
        target_volatility=0.25,
    )
    assert size == pytest.approx(0.15, abs=0.01)


def test_recommend_position_size_structure():
    rec = risk_sizing.recommend_position_size(
        "THYAO.IS",
        prob=0.75,
        volatility=0.30,
        min_threshold=0.53,
    )
    assert rec.ticker == "THYAO"
    assert rec.prob == 0.75
    assert 0.0 <= rec.position_size <= 1.0
    assert rec.cash_size == pytest.approx(1.0 - rec.position_size)
    assert "THYAO" in rec.summary_note
    assert rec.risk_level in [
        "🟢 Yüksek Güven / Agresif",
        "🟡 Dengeli / Orta Düzey",
        "🟠 Muhafazakar / Küçük Pozisyon",
        "⚪ Nakit / Giriş Yok",
    ]


def test_calculate_portfolio_allocation_normalizes_properly():
    sample_scan = pd.DataFrame(
        [
            {
                "Hisse": "THYAO",
                "Sektör": "Havacılık",
                "Son Fiyat (TL)": 300.0,
                "_prob": 0.85,
                "volatility": 0.25,
            },
            {
                "Hisse": "GARAN",
                "Sektör": "Bankacılık",
                "Son Fiyat (TL)": 110.0,
                "_prob": 0.80,
                "volatility": 0.20,
            },
            {
                "Hisse": "ASELS",
                "Sektör": "Savunma",
                "Son Fiyat (TL)": 60.0,
                "_prob": 0.75,
                "volatility": 0.25,
            },
            {
                "Hisse": "EREGL",
                "Sektör": "Demir-Çelik",
                "Son Fiyat (TL)": 50.0,
                "_prob": 0.45,
                "volatility": 0.30,
            },
        ]
    )
    alloc = risk_sizing.calculate_portfolio_allocation(
        sample_scan, max_tickers=3, min_threshold=0.53
    )
    assert len(alloc) == 3
    assert "EREGL" not in alloc["Hisse"].values  # 0.45 < 0.53
    total_alloc = alloc["Önerilen Tahsis (%)"].sum()
    assert total_alloc <= 100.0


def test_calculate_portfolio_allocation_empty():
    empty_df = pd.DataFrame()
    res = risk_sizing.calculate_portfolio_allocation(empty_df)
    assert res.empty
    assert "Önerilen Tahsis (%)" in res.columns


def test_format_allocation_as_markdown():
    sample_alloc = pd.DataFrame(
        {
            "Hisse": ["THYAO", "GARAN"],
            "Önerilen Tahsis (%)": [60.0, 40.0],
        }
    )
    md = risk_sizing.format_allocation_as_markdown(sample_alloc)
    assert "| Hisse | Önerilen Tahsis (%) |" in md
    assert "| THYAO | 60.0 |" in md


def test_backtest_with_kelly_sizing():
    dates = pd.bdate_range("2026-01-05", periods=5)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "close": [100.0, 105.0, 110.0, 108.0, 112.0],
            "volatility": [0.25, 0.25, 0.25, 0.25, 0.25],
        }
    )
    probs = [0.65, 0.70, 0.60, 0.40, 0.65]

    res_binary = backtest.run_backtest(frame, probs, threshold=0.53, sizing="binary")
    res_kelly = backtest.run_backtest(frame, probs, threshold=0.53, sizing="kelly")

    assert res_binary.sizing == "binary"
    assert res_kelly.sizing == "kelly"
    assert 0.0 <= res_kelly.mean_position_size <= 1.0
    assert len(res_kelly.returns) == 4
    assert len(res_kelly.curve) == 5
