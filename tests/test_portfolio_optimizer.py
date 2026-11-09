"""Sektörel Treemap & Markowitz Portföy Optimizasyonu birim testleri."""

import numpy as np
import pandas as pd
import pytest

from src import portfolio_optimizer


@pytest.fixture
def sample_scan_df():
    """Örnek piyasa tarama sonuç tablosu."""
    return pd.DataFrame(
        [
            {
                "Hisse": "THYAO",
                "Sektör": "Ulaştırma & Havacılık",
                "Son Fiyat (TL)": 310.50,
                "_prob": 0.65,
                "Yükseliş Olasılığı (%)": 65.0,
                "Günlük Değişim (%)": 2.4,
                "Tahmin": "YUKARI 🚀",
            },
            {
                "Hisse": "GARAN",
                "Sektör": "Bankacılık",
                "Son Fiyat (TL)": 125.00,
                "_prob": 0.58,
                "Yükseliş Olasılığı (%)": 58.0,
                "Günlük Değişim (%)": 1.1,
                "Tahmin": "YUKARI 🚀",
            },
            {
                "Hisse": "ASELS",
                "Sektör": "Savunma & Teknoloji",
                "Son Fiyat (TL)": 65.20,
                "_prob": 0.52,
                "Yükseliş Olasılığı (%)": 52.0,
                "Günlük Değişim (%)": -0.5,
                "Tahmin": "YUKARI 🚀",
            },
            {
                "Hisse": "EREGL",
                "Sektör": "Demir Çelik & Metal",
                "Son Fiyat (TL)": 48.10,
                "_prob": 0.44,
                "Yükseliş Olasılığı (%)": 44.0,
                "Günlük Değişim (%)": -1.8,
                "Tahmin": "DÜŞÜŞ 🔻",
            },
        ]
    )


def test_optimize_sharpe_portfolio_empty():
    res = portfolio_optimizer.optimize_sharpe_portfolio(
        tickers=[], expected_returns=np.array([]), cov_matrix=np.array([[]])
    )
    assert res.sharpe_ratio == 0.0
    assert res.allocation_table.empty


def test_optimize_sharpe_portfolio_weights_sum_to_one():
    tickers = ["THYAO", "GARAN", "ASELS"]
    mu = np.array([0.55, 0.48, 0.42])
    cov = np.array(
        [
            [0.08, 0.02, 0.01],
            [0.02, 0.06, 0.01],
            [0.01, 0.01, 0.05],
        ]
    )
    res = portfolio_optimizer.optimize_sharpe_portfolio(
        tickers=tickers,
        expected_returns=mu,
        cov_matrix=cov,
        risk_free_rate=0.40,
        max_weight=0.50,
        total_capital=100_000.0,
    )

    weights = list(res.weights.values())
    assert sum(weights) == pytest.approx(1.0, abs=1e-2)
    assert all(w >= 0.0 for w in weights)
    assert all(w <= 0.51 for w in weights)
    assert res.expected_annual_return > 0.40
    assert res.annual_volatility > 0.0
    assert not res.allocation_table.empty
    assert "Tutar (TL)" in res.allocation_table.columns


def test_get_historical_covariance_matrix():
    tickers = ["THYAO", "GARAN"]
    cov_df = portfolio_optimizer.get_historical_covariance_matrix(tickers)
    assert isinstance(cov_df, pd.DataFrame)
    assert cov_df.shape[0] == cov_df.shape[1]
    assert cov_df.shape[0] >= 2


def test_compute_markowitz_allocation(sample_scan_df):
    res = portfolio_optimizer.compute_markowitz_allocation(
        sample_scan_df,
        max_tickers=3,
        total_capital=100_000.0,
        max_weight=0.40,
    )

    assert not res.allocation_table.empty
    assert len(res.allocation_table) <= 3
    # Toplam tutar yaklaşık 100,000 TL olmalıdır
    total_allocated = res.allocation_table["Tutar (TL)"].sum()
    assert total_allocated == pytest.approx(100_000.0, rel=1e-2)


def test_build_sector_treemap_figure(sample_scan_df):
    # AI Olasılığına göre renklendirme
    fig_prob = portfolio_optimizer.build_sector_treemap_figure(sample_scan_df, color_by="ai_prob")
    assert len(fig_prob.data) == 1
    assert fig_prob.data[0].type == "treemap"

    # Günlük değişime göre renklendirme
    fig_change = portfolio_optimizer.build_sector_treemap_figure(
        sample_scan_df, color_by="daily_change"
    )
    assert len(fig_change.data) == 1
    assert fig_change.data[0].type == "treemap"


def test_build_portfolio_allocation_figure(sample_scan_df):
    res = portfolio_optimizer.compute_markowitz_allocation(sample_scan_df)
    fig_donut = portfolio_optimizer.build_portfolio_allocation_figure(res.allocation_table)
    assert len(fig_donut.data) == 1
    assert fig_donut.data[0].type == "pie"
    assert fig_donut.data[0].hole == 0.45


def test_format_markowitz_as_markdown(sample_scan_df):
    assert portfolio_optimizer.format_markowitz_as_markdown(pd.DataFrame()) == ""

    res = portfolio_optimizer.compute_markowitz_allocation(sample_scan_df)
    md = portfolio_optimizer.format_markowitz_as_markdown(res.allocation_table)
    assert "| Hisse | Sektör | Portföy Payı | Alım Tutarı |" in md
    assert "THYAO" in md
