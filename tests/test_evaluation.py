import numpy as np
import pytest

from census_income.evaluation import (bootstrap_metric_ci, evaluate_at_threshold, f1_by_threshold,
                                       select_threshold)


def test_threshold_selection_finds_known_optimum():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.45, 0.4, 0.6, 0.7, 0.9])
    grid = np.round(np.arange(0.2, 0.71, 0.01), 2)
    thr, scores = select_threshold(y, proba, grid)
    assert 0.31 <= thr <= 0.40 and scores.max() == pytest.approx(scores[list(grid).index(thr)])


def test_f1_by_threshold_matches_manual_value():
    y = np.array([1, 1, 0, 0]); p = np.array([0.9, 0.2, 0.8, 0.1])
    # at 0.5: tp=1, fp=1, fn=1 -> F1 = 2/(2+1+1) = 0.5
    assert f1_by_threshold(y, p, [0.5])[0] == pytest.approx(0.5)


def test_evaluate_at_threshold_confusion_counts():
    y = np.array([1, 1, 0, 0, 0]); p = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
    m = evaluate_at_threshold(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_bootstrap_ci_brackets_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500); p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    a = bootstrap_metric_ci(y, p, 0.5, n_boot=200, seed=3)
    b = bootstrap_metric_ci(y, p, 0.5, n_boot=200, seed=3)
    assert a == b
    for k, v in a.items():
        assert v["lo"] <= v["estimate"] <= v["hi"], k
