"""Model and pipeline builders. All hyper-parameters live here, in one place."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import CAT_COLS, RANDOM_STATE
from .preprocessing import final_preprocessor


def baseline() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def default_models(random_state: int = RANDOM_STATE, n_jobs: int = 1) -> dict:
    """Logistic regression (scaled inputs), random forest, gradient boosting; default settings."""
    return {
        "Logistic Regression": Pipeline([
            ("prep", final_preprocessor(True)),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state))]),
        "Random Forest": Pipeline([
            ("prep", final_preprocessor(False)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=n_jobs))]),
        "Gradient Boosting": Pipeline([
            ("prep", final_preprocessor(False)),
            ("clf", GradientBoostingClassifier(random_state=random_state))]),
    }


def search_spaces(models: dict, fast: bool = False) -> dict:
    """Small per-model grids (same CV folds, scored on F1). `fast` is for smoke tests only."""
    if fast:
        return {
            "Logistic Regression": (models["Logistic Regression"], {"clf__C": [0.1, 1]}),
            "Random Forest": (models["Random Forest"], {"clf__max_depth": [8], "clf__min_samples_leaf": [5]}),
            "Gradient Boosting": (models["Gradient Boosting"], {"clf__n_estimators": [20], "clf__max_depth": [3]}),
        }
    return {
        "Logistic Regression": (models["Logistic Regression"], {"clf__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (models["Random Forest"],
                          {"clf__max_depth": [12, 20], "clf__min_samples_leaf": [2, 5]}),
        "Gradient Boosting": (models["Gradient Boosting"],
                              {"clf__n_estimators": [150, 250], "clf__max_depth": [3, 4, 5]}),  # learning_rate = 0.1
    }


def l1_selector(C: float = 0.1, random_state: int = RANDOM_STATE) -> SelectFromModel:
    """Keeps features whose L1-logistic-regression coefficient is non-zero (threshold 1e-5)."""
    return SelectFromModel(LogisticRegression(l1_ratio=1, C=C, solver="liblinear",
                                              max_iter=2000, random_state=random_state))


def regularised_lr(l1_ratio: float, C: float, random_state: int = RANDOM_STATE) -> Pipeline:
    """L1 (l1_ratio=1) or L2 (l1_ratio=0) logistic regression with scaled inputs."""
    return Pipeline([("prep", final_preprocessor(True)),
                     ("clf", LogisticRegression(l1_ratio=l1_ratio, C=C, solver="liblinear",
                                                max_iter=2000, random_state=random_state))])


def gb_pipeline(gb_params: dict, with_selection: bool = True, cat_cols=CAT_COLS,
                scale_inputs: bool = False, random_state: int = RANDOM_STATE) -> Pipeline:
    """Gradient boosting, optionally behind the L1 selection step.

    scale_inputs=True standardises numerics before the selector (the original pipeline does not;
    trees are unaffected by scaling, the L1 selector is not).
    """
    steps = [("prep", final_preprocessor(scale_inputs, cat_cols))]
    if with_selection:
        steps.append(("select", l1_selector(random_state=random_state)))
    steps.append(("clf", GradientBoostingClassifier(random_state=random_state, **gb_params)))
    return Pipeline(steps)
