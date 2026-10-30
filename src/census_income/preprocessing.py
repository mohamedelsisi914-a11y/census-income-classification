"""Leak-free preprocessing: everything is fitted inside a Pipeline, on training folds only."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CAP_COLS, CAT_COLS, CAT_COLS_ALL, NUM_COLS


def make_preprocessor(scale: bool = True, cat_cols=CAT_COLS_ALL, log_capital: bool = True) -> ColumnTransformer:
    """Numeric: median impute (+ log1p for capital columns) (+ standard scale).
    Categorical: impute the constant 'Unknown', then one-hot (unknown levels ignored).

    Columns not listed (fnlwgt, education, race, sex, ...) are dropped by the ColumnTransformer,
    so sensitive attributes never reach an estimator. Note: log1p is only applied when
    scale=True (kept from the original notebook; only used in one preprocessing probe).
    """
    def numeric(log: bool = False) -> Pipeline:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if log:
            steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        if scale:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)

    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric(), NUM_COLS),
        ("cap", numeric(log=log_capital and scale), CAP_COLS),
        ("cat", cat, list(cat_cols)),
    ])


def final_preprocessor(scale: bool = True, cat_cols=CAT_COLS) -> ColumnTransformer:
    """The preprocessing used by the final models (raw capital columns, no native-country)."""
    return make_preprocessor(scale=scale, cat_cols=cat_cols, log_capital=False)


def feature_group(encoded_name: str, cat_cols=CAT_COLS) -> str:
    """'cat__occupation_Sales' -> 'occupation'; 'num__age' -> 'age'."""
    kind, rest = encoded_name.split("__", 1)
    if kind != "cat":
        return rest
    for c in cat_cols:
        if rest.startswith(c + "_"):
            return c
    raise ValueError(f"Cannot map encoded feature '{encoded_name}' to an original column.")
