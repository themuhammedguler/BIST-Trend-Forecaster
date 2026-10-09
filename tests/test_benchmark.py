# tests/test_benchmark.py - Model benchmark altyapısı birim testleri (#24)
import numpy as np
import pandas as pd
import pytest

import benchmark


@pytest.fixture
def synthetic_benchmark_data():
    """30 günlük sentetik 2 hisseli veri seti."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    records = []
    for ticker in ["AKBNK", "GARAN"]:
        for i, d in enumerate(dates):
            records.append(
                {
                    "Date": d,
                    "ticker": ticker,
                    "close": 100.0 + i,
                    "f1": np.sin(i),
                    "f2": np.cos(i),
                    "lag_1_ret": 0.01 if i % 2 == 0 else -0.01,
                    "target": int(i % 2 == 0),
                }
            )
    df = pd.DataFrame(records).sort_values(by=["ticker", "Date"]).reset_index(drop=True)
    return df


def test_baseline_always_down_predicts_zero_probability():
    clf = benchmark.AlwaysDownBaseline()
    clf.fit(None, None)
    X = pd.DataFrame({"f1": [1.0, 2.0]})
    probs = clf.predict_proba(X)
    assert probs.shape == (2, 2)
    assert np.allclose(probs[:, 1], 0.0)
    assert np.allclose(probs[:, 0], 1.0)


def test_baseline_always_up_predicts_one_probability():
    clf = benchmark.AlwaysUpBaseline()
    clf.fit(None, None)
    X = pd.DataFrame({"f1": [1.0, 2.0]})
    probs = clf.predict_proba(X)
    assert probs.shape == (2, 2)
    assert np.allclose(probs[:, 1], 1.0)
    assert np.allclose(probs[:, 0], 0.0)


def test_baseline_repeat_yesterday_follows_lag_1():
    clf = benchmark.RepeatYesterdayBaseline()
    clf.fit(None, None)
    X = pd.DataFrame({"lag_1_ret": [0.05, -0.02, 0.01]})
    probs = clf.predict_proba(X)
    assert probs[0, 1] > 0.5
    assert probs[1, 1] < 0.5
    assert probs[2, 1] > 0.5


def test_evaluate_model_fold_computes_all_required_metrics(synthetic_benchmark_data):
    df = synthetic_benchmark_data
    X = df[["f1", "f2", "lag_1_ret"]]
    y = df["target"]

    model = benchmark.AlwaysDownBaseline()
    scores = benchmark.evaluate_model_fold(
        model,
        X.iloc[:20],
        y.iloc[:20],
        X.iloc[20:],
        y.iloc[20:],
        df_val=df.iloc[20:],
    )

    expected_keys = {
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "log_loss",
        "strategy_return",
        "sharpe",
        "fit_time_seconds",
    }
    assert expected_keys <= set(scores.keys())
    assert 0.0 <= scores["accuracy"] <= 1.0
    assert scores["fit_time_seconds"] >= 0.0


def test_run_benchmark_evaluates_on_walk_forward_folds(synthetic_benchmark_data):
    df = synthetic_benchmark_data
    feature_cols = ["f1", "f2", "lag_1_ret"]
    models = [
        benchmark.AlwaysDownBaseline(),
        benchmark.LogisticRegressionModel(),
        benchmark.HistGradientBoostingModel(),
    ]

    res = benchmark.run_benchmark(df, feature_cols, n_splits=2, models=models)

    assert len(res) == 3
    for name, data in res.items():
        assert len(data["folds"]) == 2
        assert "mean" in data
        assert "std" in data
        assert "accuracy" in data["mean"]
        assert 0.0 <= data["mean"]["accuracy"] <= 1.0


def test_format_benchmark_markdown_generates_valid_table():
    fake_results = {
        "Model A": {
            "mean": {
                "accuracy": 0.5234,
                "balanced_accuracy": 0.5123,
                "roc_auc": 0.5345,
                "log_loss": 0.6912,
                "strategy_return": 0.0456,
                "sharpe": 0.85,
                "fit_time_seconds": 0.12,
            },
            "std": {"accuracy": 0.012},
        }
    }
    report = benchmark.format_benchmark_markdown(fake_results, n_splits=3)
    assert "| **Model A** |" in report
    assert "0.5234" in report
    assert "Walk-Forward" in report
    assert "Karar ve Entegrasyon Stratejisi" in report


@pytest.mark.skipif(not benchmark.HAS_LIGHTGBM, reason="LightGBM yüklü değil")
def test_lightgbm_model_interface(synthetic_benchmark_data):
    df = synthetic_benchmark_data
    X = df[["f1", "f2"]]
    y = df["target"]

    clf = benchmark.LightGBMModel(n_estimators=5, max_depth=2)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


@pytest.mark.skipif(not benchmark.HAS_TORCH, reason="PyTorch yüklü değil")
def test_pytorch_gru_model_interface(synthetic_benchmark_data):
    df = synthetic_benchmark_data
    X = df[["f1", "f2"]]
    y = df["target"]

    clf = benchmark.PyTorchGRUModel(hidden_dim=8, num_layers=1, epochs=2, batch_size=16)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
