# tests/test_explain.py - TreeSHAP tabanlı yerel açıklanabilirlik testleri
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

import config
from explain import FEATURE_LABELS, explain_prediction, summarize_drivers, top_drivers


@pytest.fixture
def trained_model(synthetic_data):
    X, y, _ = synthetic_data
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3, random_state=0)
    model.fit(X, y)
    return model, X


def test_contributions_cover_every_feature(trained_model):
    model, X = trained_model
    explanation, _ = explain_prediction(model, X.iloc[[0]])

    assert sorted(explanation["feature"]) == sorted(X.columns)
    assert list(explanation.columns) == ["feature", "value", "contribution", "impact"]


def test_contributions_sum_to_model_margin(trained_model):
    """TreeSHAP yerel doğruluk: bias + katkılar == modelin log-odds çıktısı."""
    model, X = trained_model
    row = X.iloc[[5]]
    explanation, base_prob = explain_prediction(model, row)

    margin = model.predict(row, output_margin=True)[0]
    base_margin = np.log(base_prob / (1 - base_prob))
    assert base_margin + explanation["contribution"].sum() == pytest.approx(margin, abs=1e-5)


def test_probability_impacts_reconcile_to_predicted_probability(trained_model):
    model, X = trained_model
    row = X.iloc[[7]]
    explanation, base_prob = explain_prediction(model, row)

    prob = model.predict_proba(row)[0][1]
    assert base_prob + explanation["impact"].sum() == pytest.approx(prob, abs=1e-5)


def test_impact_sign_follows_contribution_sign(trained_model):
    model, X = trained_model
    explanation, _ = explain_prediction(model, X.iloc[[3]])

    nonzero = explanation[explanation["contribution"] != 0]
    assert (np.sign(nonzero["impact"]) == np.sign(nonzero["contribution"])).all()


def test_sorted_by_absolute_contribution(trained_model):
    model, X = trained_model
    explanation, _ = explain_prediction(model, X.iloc[[0]])

    magnitudes = explanation["contribution"].abs().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_dominant_feature_ranks_first(trained_model):
    """Sentetik hedef f0'a bağlı; uç bir f0 değeri en büyük katkıyı vermeli."""
    model, X = trained_model
    row = X.iloc[[X["f0"].abs().idxmax()]]
    explanation, _ = explain_prediction(model, row)

    assert explanation.iloc[0]["feature"] == "f0"


def test_reports_feature_values_of_the_explained_row(trained_model):
    model, X = trained_model
    row = X.iloc[[2]]
    explanation, _ = explain_prediction(model, row)

    values = explanation.set_index("feature")["value"]
    for col in X.columns:
        assert values[col] == pytest.approx(row[col].iloc[0])


def test_rejects_multiple_rows(trained_model):
    model, X = trained_model
    with pytest.raises(ValueError):
        explain_prediction(model, X.iloc[:2])


def _explanation(impacts):
    """Verilen etkilerden (feature -> impact) explain_prediction çıktısı biçiminde tablo üretir."""
    df = pd.DataFrame(
        {
            "feature": list(impacts),
            "value": 0.0,
            "contribution": list(impacts.values()),
            "impact": list(impacts.values()),
        }
    )
    order = df["contribution"].abs().sort_values(ascending=False).index
    return df.loc[order].reset_index(drop=True)


def test_top_drivers_splits_by_direction_and_ranks_by_impact():
    explanation = _explanation(
        {
            "macd": 0.072,
            "vol_change": 0.041,
            "rsi": 0.010,
            "month": 0.002,
            "volatility": -0.030,
            "lag_1_ret": -0.005,
        }
    )

    up, down = top_drivers(explanation, n=3)

    assert up["feature"].tolist() == ["macd", "vol_change", "rsi"]
    assert down["feature"].tolist() == ["volatility", "lag_1_ret"]


def test_top_drivers_ignores_zero_impact():
    up, down = top_drivers(_explanation({"rsi": 0.0, "macd": 0.01}), n=3)

    assert up["feature"].tolist() == ["macd"]
    assert down.empty


def test_every_model_feature_has_a_readable_label():
    assert set(config.FEATURES) <= set(FEATURE_LABELS)


def test_summary_names_upward_and_downward_drivers_with_percent_impact():
    explanation = _explanation({"macd": 0.072, "vol_change": 0.041, "volatility": -0.030})

    summary = summarize_drivers(explanation, "THYAO")

    assert summary == (
        "THYAO tahminini yukarı taşıyan ana faktörler: "
        f"{FEATURE_LABELS['macd']} (+%7.2), {FEATURE_LABELS['vol_change']} (+%4.1); "
        f"aşağı çeken: {FEATURE_LABELS['volatility']} (-%3.0)."
    )


def test_summary_handles_one_sided_explanations():
    summary = summarize_drivers(_explanation({"rsi": -0.02}), "AKBNK")

    assert "yukarı taşıyan belirgin bir faktör yok" in summary
    assert f"{FEATURE_LABELS['rsi']} (-%2.0)" in summary


def test_unknown_feature_falls_back_to_its_name():
    summary = summarize_drivers(_explanation({"f0": 0.05}), "X")
    assert "f0 (+%5.0)" in summary
