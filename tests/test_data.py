import pytest

from census_income.data import (data_quality_report, exact_overlap_mask, load_adult, load_adult_files,
                                make_split)
from conftest import write_uci_files


def test_loader_handles_uci_quirks(tmp_path, synthetic_adult):
    write_uci_files(tmp_path, synthetic_adult, n_train=400)
    train, test = load_adult_files(tmp_path)
    assert (len(train), len(test)) == (400, 200)               # comment line skipped, no row lost
    assert set(test["income"]) <= {"<=50K", ">50K"}             # trailing '.' removed
    assert train["workclass"].isna().sum() == synthetic_adult["workclass"].iloc[:400].isna().sum()  # '?' -> NaN
    df = load_adult(tmp_path)
    assert len(df) == 600 and df["target"].isin([0, 1]).all()


def test_missing_files_give_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_adult(tmp_path)


def test_malformed_labels_are_rejected(tmp_path, synthetic_adult):
    bad = synthetic_adult.copy()
    bad["income"] = "rich"
    write_uci_files(tmp_path, bad)
    with pytest.raises(ValueError, match="unexpected income labels"):
        load_adult(tmp_path)


def test_split_is_stratified_and_reproducible(synthetic_adult):
    a = make_split(synthetic_adult)
    b = make_split(synthetic_adult)
    assert a[0].index.equals(b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.02
    assert len(a[1]) == 120 and not set(a[0].index) & set(a[1].index)


def test_overlap_mask_finds_exact_copies(synthetic_adult):
    X = synthetic_adult.drop(columns=["income", "target"])
    X_train, X_test = X.iloc[:400], X.iloc[400:].copy()
    X_test.iloc[0] = X_train.iloc[5]
    mask = exact_overlap_mask(X_train, X_test)
    assert mask[0] and mask.sum() >= 1 and len(mask) == len(X_test)


def test_quality_report_counts(synthetic_adult):
    df = synthetic_adult.copy()
    rep = data_quality_report(df)
    assert rep["n_records"] == 600 and rep["missing"]["workclass"] == int(df["workclass"].isna().sum())
