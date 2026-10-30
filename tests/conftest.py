import numpy as np
import pandas as pd
import pytest

from census_income.config import COLUMN_NAMES

WORKCLASS = ["Private", "Self-emp-not-inc", "Local-gov"]
MARITAL = ["Married-civ-spouse", "Never-married", "Divorced", "Widowed"]
OCCUPATION = ["Sales", "Craft-repair", "Exec-managerial", "Adm-clerical"]
RELATIONSHIP = ["Husband", "Wife", "Not-in-family", "Own-child"]


@pytest.fixture
def synthetic_adult() -> pd.DataFrame:
    """Small Adult-shaped table with RANDOM labels (no signal), including missing values."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "age": rng.integers(17, 80, n), "workclass": rng.choice(WORKCLASS, n), "fnlwgt": rng.integers(1e4, 5e5, n),
        "education": rng.choice(["HS-grad", "Bachelors"], n), "education-num": rng.integers(1, 17, n),
        "marital-status": rng.choice(MARITAL, n), "occupation": rng.choice(OCCUPATION, n),
        "relationship": rng.choice(RELATIONSHIP, n), "race": rng.choice(["White", "Black"], n),
        "sex": rng.choice(["Male", "Female"], n), "capital-gain": rng.choice([0, 0, 0, 5000], n),
        "capital-loss": rng.choice([0, 0, 1500], n), "hours-per-week": rng.integers(10, 70, n),
        "native-country": rng.choice(["United-States", "Mexico"], n),
        "income": rng.choice(["<=50K", ">50K"], n, p=[0.75, 0.25]),
    })
    for col in ("workclass", "occupation"):          # real NaN (np.random.choice would turn NaN into the string "nan")
        df.loc[rng.random(n) < 0.15, col] = np.nan
    df["target"] = (df["income"] == ">50K").astype(int)
    return df[COLUMN_NAMES + ["target"]]


def write_uci_files(folder, df: pd.DataFrame, n_train: int = 400) -> None:
    """Write df in the exact UCI layout: no header, '?' for missing, test file with a comment
    line and labels ending in a full stop."""
    raw = df[COLUMN_NAMES].fillna("?")
    raw.iloc[:n_train].to_csv(folder / "adult.data", header=False, index=False)
    test = raw.iloc[n_train:].copy()
    test["income"] = test["income"] + "."
    with open(folder / "adult.test", "w") as fh:
        fh.write("|1x3 Cross validator\n")
        test.to_csv(fh, header=False, index=False)
