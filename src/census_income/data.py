"""Loading, validating and splitting the UCI Adult files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (COLUMN_NAMES, DEFAULT_DATA_DIR, POSITIVE_LABEL, RANDOM_STATE,
                     TEST_SIZE, UCI_URL)


def _find_file(data_dir: Path, name: str) -> Path:
    for base in (data_dir, data_dir / "adult"):
        candidate = base / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{name}' in '{data_dir}'. Download the Adult dataset from {UCI_URL} "
        f"and place adult.data and adult.test in that folder (see data/README.md).")


def _validate(part: pd.DataFrame, name: str) -> None:
    if part.shape[1] != len(COLUMN_NAMES):
        raise ValueError(f"{name}: expected {len(COLUMN_NAMES)} columns, found {part.shape[1]}.")
    labels = set(part["income"].dropna().unique())
    if not labels <= {"<=50K", ">50K"}:
        raise ValueError(f"{name}: unexpected income labels {sorted(labels)}.")
    if part["income"].isna().any():
        raise ValueError(f"{name}: missing income labels.")
    if not pd.api.types.is_numeric_dtype(part["age"]):
        raise ValueError(f"{name}: 'age' is not numeric; the file may be malformed.")


def load_adult_files(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (official_train, official_test) with '?' read as NaN and clean labels.

    adult.test starts with a comment line ('|1x3 Cross validator') and its labels end
    with a full stop ('>50K.'). Both are handled here.
    """
    data_dir = Path(data_dir)
    kwargs = dict(header=None, names=COLUMN_NAMES, skipinitialspace=True, na_values="?")
    train = pd.read_csv(_find_file(data_dir, "adult.data"), **kwargs)
    test = pd.read_csv(_find_file(data_dir, "adult.test"), comment="|", **kwargs)
    test["income"] = test["income"].str.rstrip(".")
    _validate(train, "adult.data")
    _validate(test, "adult.test")
    return train, test


def load_adult(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Stack the official train and test files and add a 0/1 'target' column."""
    train, test = load_adult_files(data_dir)
    df = pd.concat([train, test], ignore_index=True)
    df["target"] = (df["income"] == POSITIVE_LABEL).astype(int)
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["income", "target"]), df["target"]


def make_split(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """One stratified split. Returns X_train, X_test, y_train, y_test."""
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def exact_overlap_mask(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Boolean array (aligned with X_test) marking test rows identical to some training row."""
    cols = list(X_test.columns)
    key = X_train.drop_duplicates().assign(_seen=1)
    merged = X_test.reset_index(drop=True).merge(key, how="left", on=cols)
    return merged["_seen"].notna().to_numpy()


def data_quality_report(df: pd.DataFrame) -> dict:
    """Descriptive checks on the raw table (no modelling decision depends on them)."""
    feats = [c for c in df.columns if c not in ("income", "target")]
    missing = df[feats].isna().sum()
    conflicts = df.groupby(feats, dropna=False)["income"].nunique()
    text_cols = df[feats].select_dtypes(exclude="number").columns
    ws = sum(int((df[c].dropna() != df[c].dropna().str.strip()).sum()) for c in text_cols)
    return {
        "n_records": int(len(df)),
        "n_feature_columns": len(feats),
        "class_counts": {k: int(v) for k, v in df["income"].value_counts().items()},
        "pos_share": float(df["target"].mean()),
        "missing": {k: int(v) for k, v in missing[missing > 0].items()},
        "workclass_and_occupation_both_missing": int((df["workclass"].isna() & df["occupation"].isna()).sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "feature_duplicate_rows_ignoring_label": int(df[feats].duplicated().sum()),
        "feature_groups_with_conflicting_labels": int((conflicts > 1).sum()),
        "whitespace_issues_in_text_columns": ws,
        "race_counts": {k: int(v) for k, v in df["race"].value_counts().items()},
        "sex_counts": {k: int(v) for k, v in df["sex"].value_counts().items()},
    }
