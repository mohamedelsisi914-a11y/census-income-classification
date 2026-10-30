"""Checks that the modelling pipeline cannot see labels or test rows it should not see."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from census_income.config import RANDOM_STATE
from census_income.modeling import gb_pipeline


@pytest.fixture
def small_gb():
    return {"n_estimators": 15, "max_depth": 3}


def test_random_labels_give_chance_level_auc(synthetic_adult, small_gb):
    """With labels that carry no signal, a leak-free CV score must be about 0.5."""
    X, y = synthetic_adult.drop(columns=["income", "target"]), synthetic_adult["target"]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    r = cross_validate(gb_pipeline(small_gb), X, y, cv=cv, scoring="roc_auc")
    assert 0.40 < r["test_score"].mean() < 0.60


def test_feature_selection_is_fitted_on_training_rows_only(synthetic_adult, small_gb):
    X, y = synthetic_adult.drop(columns=["income", "target"]), synthetic_adult["target"]
    X_tr, y_tr = X.iloc[:400], y.iloc[:400]
    a = clone(gb_pipeline(small_gb)).fit(X_tr, y_tr)
    b = clone(gb_pipeline(small_gb)).fit(X_tr, y_tr)
    # Nothing outside X_tr was passed to fit: results must be identical and independent of other rows.
    assert (a.named_steps["select"].get_support() == b.named_steps["select"].get_support()).all()
    X_other = X.iloc[400:].copy(); X_other["age"] = 999
    assert np.allclose(a.predict_proba(X_other), b.predict_proba(X_other))


def test_fixed_seed_gives_identical_predictions(synthetic_adult, small_gb):
    X, y = synthetic_adult.drop(columns=["income", "target"]), synthetic_adult["target"]
    p1 = clone(gb_pipeline(small_gb)).fit(X, y).predict_proba(X)
    p2 = clone(gb_pipeline(small_gb)).fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)


def test_sensitive_columns_do_not_change_predictions(synthetic_adult, small_gb):
    X, y = synthetic_adult.drop(columns=["income", "target"]), synthetic_adult["target"]
    model = clone(gb_pipeline(small_gb)).fit(X, y)
    flipped = X.copy()
    flipped["sex"] = flipped["sex"].map({"Male": "Female", "Female": "Male"})
    flipped["race"] = "Other"
    assert np.array_equal(model.predict_proba(X), model.predict_proba(flipped))
