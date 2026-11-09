"""Çoklu model ve topluluk (Ensemble) tahmin motoru birim testleri."""

import numpy as np
import pandas as pd

from src import ensemble


def _make_dummy_dataset():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
        }
    )
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
    return X[:80], y[:80], X[80:], y[80:]


def test_build_classifiers():
    xgb_clf = ensemble.build_xgb_classifier({"n_estimators": 10})
    lgb_clf = ensemble.build_lgbm_classifier({"n_estimators": 10})
    cat_clf = ensemble.build_catboost_classifier({"iterations": 10})

    assert xgb_clf is not None
    assert lgb_clf is not None
    assert cat_clf is not None


def test_ensemble_predictor_predict_proba():
    X_train, y_train, X_test, _ = _make_dummy_dataset()

    xgb_clf = ensemble.build_xgb_classifier({"n_estimators": 5})
    lgb_clf = ensemble.build_lgbm_classifier({"n_estimators": 5})
    cat_clf = ensemble.build_catboost_classifier({"iterations": 5})

    xgb_clf.fit(X_train, y_train)
    lgb_clf.fit(X_train, y_train)
    cat_clf.fit(X_train, y_train)

    predictor = ensemble.EnsemblePredictor(
        models={"xgb": xgb_clf, "lgb": lgb_clf, "cat": cat_clf},
        weights={"xgb": 0.5, "lgb": 0.3, "cat": 0.2},
    )

    probs = predictor.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    # Olasılık toplamı her satır için 1.0 olmalıdır
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X_test)), atol=1e-5)

    preds = predictor.predict(X_test, threshold=0.5)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_evaluate_models_comparison():
    X_train, y_train, X_test, y_test = _make_dummy_dataset()

    trained_models, df_comparison = ensemble.evaluate_models_comparison(
        X_train, y_train, X_test, y_test
    )

    assert "xgboost" in trained_models
    assert "lightgbm" in trained_models
    assert "catboost" in trained_models

    assert isinstance(df_comparison, pd.DataFrame)
    assert len(df_comparison) == 4  # XGB, LGB, Cat, Ensemble
    assert "Model" in df_comparison.columns
    assert "Doğruluk (Accuracy)" in df_comparison.columns
    assert "ROC-AUC" in df_comparison.columns


def test_save_ensemble_models(tmp_path):
    X_train, y_train, X_test, y_test = _make_dummy_dataset()
    trained_models, df_comp = ensemble.evaluate_models_comparison(X_train, y_train, X_test, y_test)

    save_dir = str(tmp_path / "models")
    ensemble.save_ensemble_models(trained_models, save_dir, comparison_df=df_comp)

    import os

    assert os.path.exists(os.path.join(save_dir, "xgb_bist_model.json"))
    assert os.path.exists(os.path.join(save_dir, "lgbm_bist_model.txt"))
    assert os.path.exists(os.path.join(save_dir, "catboost_bist_model.cbm"))
    assert os.path.exists(os.path.join(save_dir, "ensemble_meta.json"))
