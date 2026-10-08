# tests/test_signals.py - olasılığın 3 durumlu sinyale dönüştürülmesi testleri
import pytest

import config
from signals import DOWN, NEUTRAL, UP, classify_signal, symmetric_band


def test_config_defines_neutral_band_around_coin_flip():
    assert config.PROB_THRESHOLD_LOW == 0.47
    assert config.PROB_THRESHOLD_HIGH == 0.53


@pytest.mark.parametrize("prob", [0.49, 0.50, 0.51])
def test_coin_flip_probabilities_are_neutral(prob):
    assert classify_signal(prob, config.PROB_THRESHOLD_LOW, config.PROB_THRESHOLD_HIGH) == NEUTRAL


@pytest.mark.parametrize(
    "prob, expected",
    [
        (0.53, UP),  # üst eşik dahil
        (0.90, UP),
        (0.5299, NEUTRAL),
        (0.4701, NEUTRAL),
        (0.47, DOWN),  # alt eşik dahil
        (0.10, DOWN),
    ],
)
def test_thresholds_are_inclusive(prob, expected):
    assert classify_signal(prob, 0.47, 0.53) == expected


def test_wider_band_turns_weak_signals_neutral():
    assert classify_signal(0.54, 0.47, 0.53) == UP
    assert classify_signal(0.54, 0.45, 0.55) == NEUTRAL


@pytest.mark.parametrize("low, high", [(0.53, 0.47), (0.50, 0.50)])
def test_rejects_inverted_or_empty_band(low, high):
    with pytest.raises(ValueError):
        classify_signal(0.5, low, high)


@pytest.mark.parametrize(
    "min_confidence, expected",
    [(0.53, (0.47, 0.53)), (0.55, (0.45, 0.55)), (0.70, (0.30, 0.70))],
)
def test_symmetric_band_mirrors_min_confidence(min_confidence, expected):
    assert symmetric_band(min_confidence) == pytest.approx(expected)


@pytest.mark.parametrize("min_confidence", [0.50, 0.49, 1.0])
def test_symmetric_band_rejects_out_of_range_confidence(min_confidence):
    with pytest.raises(ValueError):
        symmetric_band(min_confidence)
