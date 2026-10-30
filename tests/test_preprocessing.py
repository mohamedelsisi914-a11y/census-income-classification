import numpy as np
from scipy import sparse

from census_income.config import CAT_COLS, SENSITIVE_COLS, USED_COLS
from census_income.preprocessing import feature_group, final_preprocessor


def test_sensitive_and_dropped_columns_never_reach_the_model(synthetic_adult):
    X = synthetic_adult.drop(columns=["income", "target"])
    prep = final_preprocessor(scale=False).fit(X)
    names = " ".join(prep.get_feature_names_out())
    for col in SENSITIVE_COLS + ["fnlwgt", "education__", "native-country"]:
        assert col not in names
    assert {c for ct in prep.transformers_ if ct[0] != "remainder" for c in ct[2]} == set(USED_COLS)


def test_statistics_come_from_training_data_only(synthetic_adult):
    X = synthetic_adult.drop(columns=["income", "target"])
    X_train, X_test = X.iloc[:400].copy(), X.iloc[400:].copy()
    X_test["age"] = 1000                                   # extreme values must not influence the fit
    prep = final_preprocessor(scale=True).fit(X_train)
    scaler = prep.named_transformers_["num"].named_steps["scaler"]
    assert np.isclose(scaler.mean_[0], X_train["age"].mean())
    prep.transform(X_test)
    assert np.isclose(scaler.mean_[0], X_train["age"].mean())   # transform does not refit


def test_missing_values_and_unseen_categories_are_handled(synthetic_adult):
    X = synthetic_adult.drop(columns=["income", "target"])
    prep = final_preprocessor(scale=False).fit(X.iloc[:300])
    new = X.iloc[300:310].copy()
    new["occupation"] = "Never-seen-before"
    new["age"] = np.nan
    out = prep.transform(new)
    assert out.shape[0] == 10 and not np.isnan(out.toarray() if sparse.issparse(out) else out).any()
    assert "cat__occupation_Unknown" in prep.get_feature_names_out()


def test_feature_group_mapping():
    assert feature_group("cat__occupation_Sales") == "occupation"
    assert feature_group("cat__marital-status_Never-married") == "marital-status"
    assert feature_group("num__age") == "age"
    assert all(feature_group(f"cat__{c}_x") == c for c in CAT_COLS)
