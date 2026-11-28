# tests/test_metrics.py - metrics.py birim testleri
import numpy as np
import pandas as pd
import pytest

import metrics


def test_evaluate_reports_all_metrics():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    result = metrics.evaluate(y, proba)

    assert set(result) == {"accuracy", "balanced_accuracy", "roc_auc", "log_loss", "precision"}
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(0.5)
    assert result["log_loss"] > 0


def test_evaluate_exposes_majority_class_predictor():
    # %80 yükseliş günü; her şeye "1" diyen model yüksek doğruluk ama şans seviyesinde AUC alır
    y = np.array([1] * 8 + [0] * 2)
    proba = np.full(len(y), 0.9)
    result = metrics.evaluate(y, proba)

    assert result["accuracy"] == pytest.approx(0.8)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(0.5)


def test_evaluate_respects_threshold():
    y = np.array([0, 1])
    proba = np.array([0.55, 0.65])
    assert metrics.evaluate(y, proba)["accuracy"] == pytest.approx(0.5)
    assert metrics.evaluate(y, proba, threshold=0.6)["accuracy"] == pytest.approx(1.0)


def test_evaluate_single_class_has_undefined_auc():
    result = metrics.evaluate(np.array([1, 1, 1]), np.array([0.7, 0.8, 0.9]))
    assert np.isnan(result["roc_auc"])
    assert result["accuracy"] == pytest.approx(1.0)


def test_class_distribution():
    dist = metrics.class_distribution(pd.Series([1, 0, 1, 1]))
    assert dist == {0: pytest.approx(0.25), 1: pytest.approx(0.75)}


def test_class_distribution_includes_missing_class():
    assert metrics.class_distribution([1, 1]) == {0: 0.0, 1: 1.0}


def test_scale_pos_weight_is_negative_to_positive_ratio():
    assert metrics.scale_pos_weight([1, 1, 1, 0]) == pytest.approx(1 / 3)
    assert metrics.scale_pos_weight([0, 0, 1]) == pytest.approx(2.0)


def test_scale_pos_weight_falls_back_to_one_for_single_class():
    assert metrics.scale_pos_weight([1, 1]) == 1.0
    assert metrics.scale_pos_weight([0, 0]) == 1.0


def test_evaluate_extended_reports_recall_and_f1():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    ext = metrics.evaluate_extended(y, proba)

    assert {"recall", "f1", "f1_macro"} <= set(ext.keys())
    assert ext["recall"] == pytest.approx(1.0)
    assert ext["f1"] == pytest.approx(1.0)
    assert ext["f1_macro"] == pytest.approx(1.0)
