"""Extra checks used by the audit: selection stability, robustness to the split, benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from .config import N_FOLDS, RANDOM_STATE
from .evaluation import evaluate_at_threshold, select_threshold
from .modeling import gb_pipeline, l1_selector
from .preprocessing import final_preprocessor


def eda_train_summary(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Target-related exploration on the TRAINING data only."""
    tr = X_train.assign(target=y_train.values)
    miss = tr["workclass"].isna()
    gain = tr["capital-gain"] > 0
    return {
        "positive_rate": float(tr["target"].mean()),
        "positive_rate_workclass_missing": float(tr.loc[miss, "target"].mean()),
        "n_workclass_missing": int(miss.sum()),
        "share_capital_gain_zero": float((tr["capital-gain"] == 0).mean()),
        "positive_rate_capital_gain_positive": float(tr.loc[gain, "target"].mean()),
        "capital_gain_top_coded_count": int((tr["capital-gain"] == 99999).sum()),
        "positive_rate_married_civ_spouse": float(tr.loc[tr["marital-status"] == "Married-civ-spouse", "target"].mean()),
        "positive_rate_never_married": float(tr.loc[tr["marital-status"] == "Never-married", "target"].mean()),
    }


def selection_stability(X: pd.DataFrame, y: pd.Series, n_boot: int = 30,
                        seed: int = RANDOM_STATE) -> pd.DataFrame:
    """How often does each encoded feature survive the L1 selector? Uses the 5 CV training
    parts and `n_boot` bootstrap resamples of the training data (all training-data only)."""
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    splits = [tr for tr, _ in cv.split(X, y)]
    splits += [rng.integers(0, len(X), len(X)) for _ in range(n_boot)]
    all_names = list(final_preprocessor(False).fit(X).get_feature_names_out())
    flags = []
    for idx in splits:
        pipe = Pipeline([("prep", final_preprocessor(False)), ("select", l1_selector())])
        pipe.fit(X.iloc[idx], y.iloc[idx])
        names = pipe.named_steps["prep"].get_feature_names_out()
        selected = pd.Series(pipe.named_steps["select"].get_support(), index=names)
        # a level absent from a resample cannot be selected: count it as "not selected"
        flags.append(selected.reindex(all_names, fill_value=False).to_numpy())
    freq = np.mean(flags, axis=0)
    return pd.DataFrame({"feature": all_names, "selection_frequency": freq}).sort_values(
        "selection_frequency", ascending=False).reset_index(drop=True)


def _oof_threshold(pipe, X, y, seed):
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    return select_threshold(y, oof)[0]


def fit_and_evaluate(X_tr, y_tr, X_te, y_te, gb_params, seed=RANDOM_STATE) -> dict:
    """Pick the threshold by OOF on (X_tr, y_tr), fit, evaluate once on (X_te, y_te)."""
    pipe = gb_pipeline(gb_params, with_selection=True, random_state=RANDOM_STATE)
    thr = _oof_threshold(pipe, X_tr, y_tr, seed)
    model = clone(pipe).fit(X_tr, y_tr)
    res = evaluate_at_threshold(y_te, model.predict_proba(X_te)[:, 1], thr)
    return res


def seed_sensitivity(df: pd.DataFrame, gb_params: dict, seeds=(0, 1, 2, 3, 4)) -> list[dict]:
    """Repeat split + threshold selection + evaluation for other split seeds.
    Hyper-parameters were tuned on the seed-42 training data, so other seeds' test rows were
    partly seen during tuning: this measures split-to-split variation, not an unbiased estimate."""
    X, y = df.drop(columns=["income", "target"]), df["target"]
    out = []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
        r = fit_and_evaluate(Xtr, ytr, Xte, yte, gb_params, seed=s)
        out.append({"split_seed": s, **{k: r[k] for k in ("threshold", "precision", "recall", "f1", "roc_auc")}})
    return out


def official_split_check(train_df: pd.DataFrame, test_df: pd.DataFrame, gb_params: dict) -> dict:
    """Train on the official adult.data, evaluate on the official adult.test (comparable to the
    literature). Hyper-parameters come from the random-split tuning, so this is a sanity check."""
    drop = ["income", "target"]
    tr = train_df.assign(target=(train_df["income"] == ">50K").astype(int))
    te = test_df.assign(target=(test_df["income"] == ">50K").astype(int))
    return fit_and_evaluate(tr.drop(columns=drop), tr["target"], te.drop(columns=drop), te["target"], gb_params)


def scaled_selector_sensitivity(X_train, y_train, gb_params, cv) -> list[dict]:
    """Does standardising numerics before the L1 selector change the outcome? (CV on train)"""
    from .evaluation import cross_validate_oof
    rows = []
    for label, scale in (("L1 selector on raw inputs (as in the final model)", False),
                         ("L1 selector on standardised inputs", True)):
        pipe = gb_pipeline(gb_params, with_selection=True, scale_inputs=scale)
        folds, oof = cross_validate_oof(pipe, X_train, y_train, cv)
        n_feat = []
        for tr, _ in cv.split(X_train, y_train):
            p = clone(pipe).fit(X_train.iloc[tr], y_train.iloc[tr])
            n_feat.append(int(p.named_steps["select"].get_support().sum()))
        thr, _ = select_threshold(y_train, oof)
        rows.append({"setup": label, "f1_at_0.5": float(folds["f1"].mean()),
                     "roc_auc": float(folds["roc_auc"].mean()), "avg_features": float(np.mean(n_feat)),
                     "oof_best_threshold": thr,
                     "oof_f1_at_best_threshold": float(f1_score(y_train, (oof >= thr).astype(int)))})
    return rows


def hist_gb_benchmark(X_train, y_train, cv) -> dict:
    """Sanity benchmark only: scikit-learn HistGradientBoosting with default settings."""
    from .evaluation import cross_validate_oof
    pipe = Pipeline([("prep", final_preprocessor(False).set_params(sparse_threshold=0)),  # dense output
                     ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE))])
    folds, oof = cross_validate_oof(pipe, X_train, y_train, cv)
    thr, _ = select_threshold(y_train, oof)
    return {"f1_at_0.5": float(folds["f1"].mean()), "roc_auc": float(folds["roc_auc"].mean()),
            "oof_best_threshold": thr,
            "oof_f1_at_best_threshold": float(f1_score(y_train, (oof >= thr).astype(int)))}


# --------------------------------------------------------------------------------------------
# Building blocks shared by scripts/run_pipeline.py and the notebook (all use TRAINING data only)
# --------------------------------------------------------------------------------------------
def preprocessing_probes(X_train, y_train, cv) -> pd.DataFrame:
    """Plain logistic regression as a probe for two preprocessing choices (log1p, native-country)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    from .config import CAT_COLS
    from .preprocessing import make_preprocessor
    variants = {
        "raw capital + native-country": make_preprocessor(log_capital=False),
        "log1p capital + native-country": make_preprocessor(log_capital=True),
        "raw capital, no native-country": make_preprocessor(log_capital=False, cat_cols=CAT_COLS),
    }
    rows = []
    for name, prep in variants.items():
        pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
        r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["f1", "roc_auc"], n_jobs=1)
        rows.append({"variant": name, "f1": r["test_f1"].mean(), "f1_std": r["test_f1"].std(),
                     "roc_auc": r["test_roc_auc"].mean(), "n_features": len(prep.fit(X_train).get_feature_names_out())})
    return pd.DataFrame(rows).set_index("variant")


def tune_all(models: dict, X_train, y_train, cv, fast: bool = False, log=print):
    """GridSearchCV per model (scored on F1, same folds). Returns (searches, table, edge_check)."""
    import time
    from sklearn.model_selection import GridSearchCV
    from .config import METRICS
    from .modeling import search_spaces
    spaces = search_spaces(models, fast=fast)
    searches, rows = {}, []
    for name, (pipe, grid) in spaces.items():
        t = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring={m: m for m in METRICS}, refit="f1", n_jobs=1).fit(X_train, y_train)
        searches[name] = gs
        i = gs.best_index_
        row = {"model": name, "best_params": str({k.replace("clf__", ""): v for k, v in gs.best_params_.items()})}
        for m in METRICS:
            row[m], row[m + "_std"] = gs.cv_results_[f"mean_test_{m}"][i], gs.cv_results_[f"std_test_{m}"][i]
        rows.append(row)
        log(f"   {name}: best CV F1 {row['f1']:.4f} {row['best_params']} ({time.time() - t:.0f}s)")
    edge = []
    for name, (_, grid) in spaces.items():
        for k, values in grid.items():
            b = searches[name].best_params_[k]
            edge.append({"model": name, "param": k.replace("clf__", ""), "best": b, "grid": values,
                         "on_edge": bool(b in (min(values), max(values)))})
    return searches, pd.DataFrame(rows).set_index("model"), edge


def gb_neighbour_settings(X_train, y_train, cv, grid_best_score: float) -> pd.DataFrame:
    """Settings just outside the grid edge for the winning gradient boosting."""
    from sklearn.model_selection import cross_validate
    rows = [{"setting": "depth 5, 150 trees (grid best)", "f1": grid_best_score}]
    for label, kw in {"depth 6, 150 trees": dict(max_depth=6, n_estimators=150),
                      "depth 5, 100 trees": dict(max_depth=5, n_estimators=100)}.items():
        r = cross_validate(gb_pipeline(kw, with_selection=False), X_train, y_train, cv=cv, scoring=["f1"], n_jobs=1)
        rows.append({"setting": label, "f1": r["test_f1"].mean()})
    return pd.DataFrame(rows).set_index("setting")


def l1_vs_l2(X_train, y_train, cv):
    """L2 baseline vs L1 at several C (CV on train). Returns (table, L1 C=0.1 coefficients)."""
    from sklearn.model_selection import cross_validate
    from .modeling import regularised_lr
    from .preprocessing import feature_group
    names = final_preprocessor(True).fit(X_train).get_feature_names_out()
    rows = []
    for label, l1_ratio, C in [("L2 (baseline)", 0, 1.0), ("L1", 1, 1.0), ("L1", 1, 0.1), ("L1", 1, 0.03), ("L1", 1, 0.01)]:
        pipe = regularised_lr(l1_ratio, C)
        r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["f1", "roc_auc"], n_jobs=1)
        coef = clone(pipe).fit(X_train, y_train).named_steps["clf"].coef_[0]
        kept = {feature_group(n) for n, c in zip(names, coef) if c != 0}
        rows.append({"penalty": label, "C": C, "f1": r["test_f1"].mean(), "f1_std": r["test_f1"].std(),
                     "roc_auc": r["test_roc_auc"].mean(), "non-zero coefficients": int((coef != 0).sum()),
                     "original features kept": len(kept)})
    l1 = regularised_lr(1, 0.1).fit(X_train, y_train).named_steps["clf"].coef_[0]
    return pd.DataFrame(rows), pd.Series(l1, index=names)


def gb_selection_comparison(X_train, y_train, gb_params, cv) -> pd.DataFrame:
    """Gradient boosting on all 42 encoded columns vs behind the L1 selector."""
    from sklearn.model_selection import cross_validate
    rows = []
    for name, with_sel in (("GB, all features", False), ("GB, L1-selected features", True)):
        r = cross_validate(gb_pipeline(gb_params, with_selection=with_sel), X_train, y_train, cv=cv,
                           scoring=["f1", "roc_auc"], n_jobs=1, return_estimator=True)
        n_feat = np.mean([e.named_steps["select"].get_support().sum() if "select" in e.named_steps
                          else len(e.named_steps["prep"].get_feature_names_out()) for e in r["estimator"]])
        rows.append({"setup": name, "f1": r["test_f1"].mean(), "f1_std": r["test_f1"].std(),
                     "roc_auc": r["test_roc_auc"].mean(), "avg features": round(float(n_feat), 1)})
    return pd.DataFrame(rows).set_index("setup")


def error_profile(X_test, y_test, y_pred, y_proba) -> dict:
    """Descriptive profile of false positives / false negatives / true positives (test set)."""
    err = X_test.copy()
    err["y_true"], err["y_pred"], err["proba"] = np.asarray(y_test), np.asarray(y_pred), np.asarray(y_proba)
    fp = err[(err.y_true == 0) & (err.y_pred == 1)]
    fn = err[(err.y_true == 1) & (err.y_pred == 0)]
    tp = err[(err.y_true == 1) & (err.y_pred == 1)]
    num = ["age", "education-num", "hours-per-week"]
    prof = pd.DataFrame({"FP": fp[num].mean(), "FN": fn[num].mean(), "TP": tp[num].mean(), "all test": X_test[num].mean()}).round(1)
    prof.loc["share capital-gain > 0"] = [(g["capital-gain"] > 0).mean() for g in (fp, fn, tp, X_test)]
    prof.loc["share married-civ-spouse"] = [(g["marital-status"] == "Married-civ-spouse").mean() for g in (fp, fn, tp, X_test)]
    return {"profile": prof.round(3),
            "occupation_top": {"FP": fp["occupation"].value_counts(normalize=True).head(4).round(3).to_dict(),
                               "FN": fn["occupation"].value_counts(normalize=True).head(4).round(3).to_dict()},
            "mean_proba": {"FP": float(fp["proba"].mean()), "FN": float(fn["proba"].mean())}}


def permutation_importance_train(final_model, X_train, y_train) -> pd.Series:
    """Permutation importance (drop in ROC AUC) on a validation slice cut from the TRAINING data."""
    from sklearn.inspection import permutation_importance
    from .config import USED_COLS
    X_fit, X_val, y_fit, y_val = train_test_split(X_train[USED_COLS], y_train, test_size=0.2,
                                                  stratify=y_train, random_state=RANDOM_STATE)
    pi = permutation_importance(clone(final_model).fit(X_fit, y_fit), X_val, y_val, scoring="roc_auc",
                                n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    return pd.Series(pi.importances_mean, index=USED_COLS).sort_values(ascending=False)


def sex_proxy_ablation(X_train, y_train, gb_params, cv, threshold, oof_current) -> pd.DataFrame:
    """Do marital-status / relationship act as proxies for sex? Out-of-fold, TRAINING data only.
    `sex` is used to split the results, never as a feature."""
    from .config import CAT_COLS
    from .evaluation import cross_validate_oof
    sex = X_train["sex"].to_numpy()
    variants = {"A: current (marital-status + relationship)": CAT_COLS,
                "B: without relationship": [c for c in CAT_COLS if c != "relationship"],
                "C: without relationship and marital-status": [c for c in CAT_COLS if c not in ("relationship", "marital-status")]}
    rows = []
    for name, cats in variants.items():
        p = oof_current if name.startswith("A") else cross_validate_oof(gb_pipeline(gb_params, True, cat_cols=cats), X_train, y_train, cv)[1]
        pred = (p >= threshold).astype(int)
        row = {"variant": name, "OOF F1": f1_score(y_train, pred), "ROC AUC": roc_auc_score(y_train, p)}
        for g in ("Male", "Female"):
            m = sex == g
            yt, yp = y_train.to_numpy()[m], pred[m]
            row[f"{g} predicted >50K"] = yp.mean()
            row[f"{g} recall"] = ((yp == 1) & (yt == 1)).sum() / (yt == 1).sum()
            row[f"{g} FPR"] = ((yp == 1) & (yt == 0)).sum() / (yt == 0).sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")
