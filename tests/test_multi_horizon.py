"""tests/test_multi_horizon.py - Çoklu vade trend tahmin motoru birim testleri."""

import numpy as np
import pandas as pd
import pytest

from src.multi_horizon import (
    HorizonForecast,
    add_multi_horizon_targets,
    calculate_expected_price_corridor,
    calculate_pivot_levels,
    calculate_risk_reward,
    calculate_trend_strength,
    compute_confluence,
    load_multi_horizon_models,
    predict_horizons,
)


def test_add_multi_horizon_targets_creates_correct_columns_and_shifts():
    # 30 günlük yapay fiyat serisi
    prices = [10.0 + i for i in range(30)]  # Sürekli artan seri
    df = pd.DataFrame({"close": prices})

    result = add_multi_horizon_targets(df)

    assert "target_1d" in result.columns
    assert "target_5d" in result.columns
    assert "target_20d" in result.columns

    # Fiyat sürekli arttığı için tüm hedefler 1.0 olmalı (sondaki NaN'lar hariç)
    assert result["target_1d"].iloc[0] == 1.0
    assert result["target_5d"].iloc[0] == 1.0
    assert result["target_20d"].iloc[0] == 1.0

    # Sondaki satırlarda shift kaynaklı NaN bulunmalı
    assert np.isnan(result["target_1d"].iloc[-1])
    assert np.isnan(result["target_5d"].iloc[-5])
    assert np.isnan(result["target_20d"].iloc[-20])


def test_add_multi_horizon_targets_raises_on_missing_column():
    df = pd.DataFrame({"wrong_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="sütunu veri çerçevesinde bulunamadı"):
        add_multi_horizon_targets(df, close_col="close")


def test_compute_confluence_full_bull():
    forecasts = {
        "1d": HorizonForecast("1d", 1, 0.65, 1, "YÜKSELİŞ"),
        "5d": HorizonForecast("5d", 5, 0.70, 1, "YÜKSELİŞ"),
        "20d": HorizonForecast("20d", 20, 0.60, 1, "YÜKSELİŞ"),
    }
    score, label, summary = compute_confluence(forecasts)
    assert score == pytest.approx(1.0)
    assert "GÜÇLÜ BOĞA" in label
    assert "uyumu maksimum" in summary


def test_compute_confluence_full_bear():
    forecasts = {
        "1d": HorizonForecast("1d", 1, 0.35, -1, "DÜŞÜŞ"),
        "5d": HorizonForecast("5d", 5, 0.30, -1, "DÜŞÜŞ"),
        "20d": HorizonForecast("20d", 20, 0.40, -1, "DÜŞÜŞ"),
    }
    score, label, summary = compute_confluence(forecasts)
    assert score == pytest.approx(-1.0)
    assert "GÜÇLÜ AYI" in label


def test_compute_confluence_healthy_pullback():
    forecasts = {
        "1d": HorizonForecast("1d", 1, 0.40, -1, "DÜŞÜŞ"),
        "5d": HorizonForecast("5d", 5, 0.65, 1, "YÜKSELİŞ"),
        "20d": HorizonForecast("20d", 20, 0.60, 1, "YÜKSELİŞ"),
    }
    score, label, summary = compute_confluence(forecasts)
    # -1 * 0.25 + 1 * 0.45 + 1 * 0.30 = 0.50
    assert score == pytest.approx(0.50)
    assert "BOĞA İÇİ SAĞLIKLI DÜZELTME" in label


def test_compute_confluence_bear_market_rally():
    forecasts = {
        "1d": HorizonForecast("1d", 1, 0.60, 1, "YÜKSELİŞ"),
        "5d": HorizonForecast("5d", 5, 0.40, -1, "DÜŞÜŞ"),
        "20d": HorizonForecast("20d", 20, 0.35, -1, "DÜŞÜŞ"),
    }
    score, label, summary = compute_confluence(forecasts)
    # 1 * 0.25 - 1 * 0.45 - 1 * 0.30 = -0.50
    assert score == pytest.approx(-0.50)
    assert "AYI İÇİ TEPKİ YÜKSELİŞİ" in label


class MockModel:
    def __init__(self, prob):
        self.prob = prob

    def predict_proba(self, X):
        return np.array([[1.0 - self.prob, self.prob]])


def test_predict_horizons_with_models():
    models = {
        "1d": MockModel(0.60),
        "5d": MockModel(0.70),
        "20d": MockModel(0.50),
    }
    X = pd.DataFrame({"rsi": [55.0], "macd": [0.2]})

    result = predict_horizons(models, X, prob_low=0.47, prob_high=0.53)

    assert result.forecasts["1d"].signal == 1
    assert result.forecasts["5d"].signal == 1
    assert result.forecasts["20d"].signal == 0  # 0.50 nötr bandında
    assert result.confluence_score > 0.0


def test_predict_horizons_empty_df_raises():
    with pytest.raises(ValueError, match="boş olamaz"):
        predict_horizons({}, pd.DataFrame())


def test_calculate_pivot_levels():
    df = pd.DataFrame(
        {
            "high": [105.0],
            "low": [95.0],
            "close": [100.0],
        }
    )
    pivots = calculate_pivot_levels(df)
    assert pivots["pivot"] == pytest.approx(100.0)
    assert pivots["r1"] == pytest.approx(105.0)
    assert pivots["s1"] == pytest.approx(95.0)
    assert pivots["fib_500"] == pytest.approx(100.0)


def test_calculate_trend_strength():
    # En az 30 satırlık veri
    dates = pd.bdate_range("2026-01-01", periods=35)
    close = np.linspace(100.0, 150.0, 35)
    df = pd.DataFrame(
        {
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
        },
        index=dates,
    )
    trend = calculate_trend_strength(df)
    assert "adx" in trend
    assert trend["adx"] >= 0.0
    assert "trend_strength" in trend


def test_calculate_expected_price_corridor():
    corridor = calculate_expected_price_corridor(current_price=100.0, daily_volatility=0.02)
    assert "1d" in corridor
    assert "5d" in corridor
    assert "20d" in corridor
    assert corridor["1d"]["min"] < 100.0 < corridor["1d"]["max"]
    assert corridor["5d"]["min"] < corridor["1d"]["min"]
    assert corridor["20d"]["min"] < corridor["5d"]["min"]


def test_calculate_risk_reward():
    rr = calculate_risk_reward(current_price=100.0, s1=95.0, r1=110.0)
    # Gain: 10, Risk: 5 -> Ratio = 2.0
    assert rr["ratio"] == pytest.approx(2.0)
    assert rr["is_favorable"] is True
    assert "CAZİP" in rr["label"]


def test_load_multi_horizon_models():
    models = load_multi_horizon_models()
    assert "1d" in models
    assert "5d" in models
    assert "20d" in models
