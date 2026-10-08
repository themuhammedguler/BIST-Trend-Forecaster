# tests/test_tune.py - tune.py birim testleri
import numpy as np
import optuna
import pandas as pd
import pytest
import xgboost as xgb

import metrics
import tune


def test_suggest_params_within_search_space():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(20):
        trial = study.ask()
        params = tune.suggest_params(trial)
        assert 50 <= params["n_estimators"] <= 500
        assert 0.01 <= params["learning_rate"] <= 0.3
        assert 2 <= params["max_depth"] <= 10
        assert 0.5 <= params["subsample"] <= 1.0
        assert 0.5 <= params["colsample_bytree"] <= 1.0
        study.tell(trial, 0.5)


def test_default_params_are_inside_search_space():
    trial = optuna.trial.FixedTrial(
        {
            **tune.DEFAULT_PARAMS,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
        }
    )
    assert tune.suggest_params(trial)["max_depth"] == tune.DEFAULT_PARAMS["max_depth"]


def test_time_series_folds_are_walk_forward(synthetic_data):
    _, _, dates = synthetic_data
    dates = dates.to_numpy()
    folds = list(tune.time_series_folds(dates, n_splits=3))
    assert len(folds) == 3

    covered = set()
    for train_idx, val_idx in folds:
        assert len(train_idx) > 0 and len(val_idx) > 0
        assert set(train_idx).isdisjoint(val_idx)
        # Doğrulama günleri eğitim günlerinden kesinlikle sonra
        assert dates[train_idx].max() < dates[val_idx].min()
        covered.update(val_idx)

    # Eğitim setleri büyüyerek ilerler
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes)
    # Son kat verinin sonuna kadar uzanır
    assert max(covered) == len(dates) - 1


def test_time_series_folds_keep_same_day_together():
    dates = np.repeat(np.arange(8), 5)  # 8 gün, her gün 5 hisse
    for train_idx, val_idx in tune.time_series_folds(dates, n_splits=3):
        assert set(dates[train_idx]).isdisjoint(dates[val_idx])


def test_time_series_folds_rejects_too_few_dates():
    with pytest.raises(ValueError):
        list(tune.time_series_folds(np.array([1, 1, 2]), n_splits=3))


def test_objective_returns_accuracy(synthetic_data):
    X, y, dates = synthetic_data
    objective = tune.make_objective(X, y, dates, n_splits=2)
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"n_estimators": 50, "max_depth": 2, "min_child_weight": 1})
    score = objective(study.ask())
    assert 0.0 <= score <= 1.0
    # Sinyal var: rastgele tahminden anlamlı derecede iyi olmalı
    assert score > 0.6


def test_run_study_starts_from_default_params(synthetic_data):
    X, y, dates = synthetic_data
    study = tune.run_study(X, y, dates, n_trials=3, n_splits=2)
    assert len(study.trials) == 3
    first = study.trials[0].params
    for key, value in tune.DEFAULT_PARAMS.items():
        assert first[key] == pytest.approx(value)
    assert set(study.best_params) == set(tune.suggest_params(study.best_trial).keys())
    assert 0.0 <= study.best_value <= 1.0


def test_run_study_is_reproducible_with_seed(synthetic_data):
    X, y, dates = synthetic_data
    a = tune.run_study(X, y, dates, n_trials=3, n_splits=2, seed=7)
    b = tune.run_study(X, y, dates, n_trials=3, n_splits=2, seed=7)
    assert a.best_params == b.best_params
    assert a.best_value == pytest.approx(b.best_value)


@pytest.mark.parametrize("metric", tune.METRICS)
def test_objective_supports_each_metric(synthetic_data, metric):
    X, y, dates = synthetic_data
    objective = tune.make_objective(X, y, dates, n_splits=2, metric=metric)
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"n_estimators": 50, "max_depth": 2, "min_child_weight": 1})
    score = objective(study.ask())
    assert 0.6 < score <= 1.0


def test_objective_rejects_unknown_metric(synthetic_data):
    X, y, dates = synthetic_data
    with pytest.raises(ValueError, match="metric"):
        tune.make_objective(X, y, dates, metric="f1")


def test_run_study_records_metric(synthetic_data):
    X, y, dates = synthetic_data
    study = tune.run_study(X, y, dates, n_trials=2, n_splits=2, metric="roc_auc")
    assert study.user_attrs["metric"] == "roc_auc"


@pytest.fixture
def imbalanced_data():
    """%80 yükseliş günü, zayıf sinyal: ağırlıksız model çoğunluk sınıfına kayar."""
    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f0", "f1", "f2", "f3"])
    y = pd.Series((X["f0"] + rng.normal(scale=1.5, size=n) > -1.6).astype(int))
    return X, y


def test_class_weight_params(imbalanced_data):
    _, y = imbalanced_data
    assert tune.class_weight_params(y, balance=False) == {}
    weight = tune.class_weight_params(y, balance=True)["scale_pos_weight"]
    assert weight == pytest.approx((y == 0).sum() / (y == 1).sum())


def test_class_weighting_counters_majority_class_bias(imbalanced_data):
    X, y = imbalanced_data
    X_tr, y_tr, X_te, y_te = X[:2000], y[:2000], X[2000:], y[2000:]

    def fit_score(balance):
        model = xgb.XGBClassifier(
            **tune.DEFAULT_PARAMS, **tune.FIXED_PARAMS, **tune.class_weight_params(y_tr, balance)
        )
        model.fit(X_tr, y_tr)
        return metrics.evaluate(y_te, model.predict_proba(X_te)[:, 1])

    plain, balanced = fit_score(False), fit_score(True)
    assert balanced["balanced_accuracy"] > plain["balanced_accuracy"] + 0.03


def test_run_study_with_class_balancing(synthetic_data):
    X, y, dates = synthetic_data
    study = tune.run_study(X, y, dates, n_trials=2, n_splits=2, balance_classes=True)
    assert study.user_attrs["balance_classes"] is True
    assert 0.0 <= study.best_value <= 1.0
