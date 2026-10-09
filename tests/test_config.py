# tests/test_config.py - config modülü ve merkezi FEATURES sabiti birim testleri
import os

import pandas as pd
import pytest
import xgboost as xgb

import config
import explain
import features


def test_config_defines_macro_tickers():
    assert config.INDEX_TICKER == "XU100.IS"
    assert config.FX_TICKER == "USDTRY=X"
    assert config.MACRO_TICKERS == ["XU100.IS", "USDTRY=X"]


def test_config_defines_features_list():
    assert isinstance(config.FEATURES, list)
    assert len(config.FEATURES) == 14
    assert len(config.FEATURES) == len(set(config.FEATURES))
    expected = [
        "rsi",
        "macd",
        "sma_10",
        "sma_50",
        "bb_width",
        "volatility",
        "lag_1_ret",
        "lag_2_ret",
        "vol_change",
        "day_of_week",
        "month",
        "xu100_ret",
        "rel_strength_bist",
        "usdtry_change",
    ]
    assert config.FEATURES == expected


@pytest.mark.skipif(not os.path.exists(config.MODEL_PATH), reason="model dosyası yok")
def test_config_features_match_shipped_model_booster():
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    assert config.FEATURES == model.get_booster().feature_names


def test_all_config_features_have_human_readable_labels():
    for f in config.FEATURES:
        assert f in explain.FEATURE_LABELS
        assert isinstance(explain.FEATURE_LABELS[f], str)
        assert len(explain.FEATURE_LABELS[f]) > 0


def test_add_features_produces_all_config_features():
    dates = pd.bdate_range("2025-01-01", periods=60)
    raw = pd.DataFrame(
        {
            "Date": dates,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
            "ticker": "TEST",
        }
    )
    processed = features.add_features(raw, drop_incomplete_target=False)
    for col in config.FEATURES:
        assert col in processed.columns
