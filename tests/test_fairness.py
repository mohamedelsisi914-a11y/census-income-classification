import numpy as np
import pandas as pd
import pytest

from census_income.fairness import gap_ci, group_report, group_report_ci


def _toy():
    groups = pd.Series(["A"] * 100 + ["B"] * 100)
    y_true = np.r_[np.ones(50), np.zeros(50), np.ones(20), np.zeros(80)]
    y_pred = np.r_[np.ones(40), np.zeros(10), np.ones(10), np.zeros(40),   # A: tp40 fn10 fp10 tn40
                   np.ones(10), np.zeros(10), np.ones(4), np.zeros(76)]     # B: tp10 fn10 fp4 tn76
    return y_true, y_pred, groups


def test_group_rates_are_computed_correctly():
    rep = group_report(*_toy()[:2], _toy()[2], "g")
    a, b = rep.loc["A"], rep.loc["B"]
    assert (a["recall"], a["precision"], a["FPR"]) == (0.8, 0.8, 0.2)
    assert (b["recall"], b["FPR"]) == (0.5, 0.05)


def test_gap_ci_contains_point_estimate():
    y, p, g = _toy()
    gap = gap_ci(y, p, g, "A", "B", n_boot=500, seed=1)
    d = gap["recall gap"]
    assert d["estimate"] == pytest.approx(0.3) and d["lo"] <= 0.3 <= d["hi"]
    assert gap["FPR ratio"]["estimate"] == pytest.approx(4.0)


def test_small_groups_are_flagged():
    y, p, g = _toy()
    ci = group_report_ci(y, p, g, "g", n_boot=100)
    assert ci["small_group"].all()          # both toy groups have fewer than 150 rows
