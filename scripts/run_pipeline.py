#!/usr/bin/env python
"""Run the whole census-income pipeline: data checks -> CV model comparison -> tuning ->
feature selection -> threshold -> single test evaluation -> error / fairness analysis -> extras.

Writes results/results.json and the figures in reports/figures/.

    python scripts/run_pipeline.py                 # full run (about 13 min on one CPU core)
    python scripts/run_pipeline.py --fast          # tiny smoke test (about 1 min), NOT for reporting
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))          # lets the script run without `pip install -e .`

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from census_income import analysis, plots
from census_income.config import (DEFAULT_DATA_DIR, DEFAULT_FIGURES_DIR, DEFAULT_RESULTS_DIR, N_FOLDS,
                                  RANDOM_STATE, THRESHOLD_GRID, USED_COLS)
from census_income.data import (data_quality_report, exact_overlap_mask, load_adult, load_adult_files,
                                make_split)
from census_income.evaluation import (bootstrap_metric_ci, cross_validate_oof, evaluate_at_threshold,
                                       select_threshold, summarise_folds)
from census_income.fairness import gap_ci, group_report, group_report_ci
from census_income.modeling import baseline, default_models, gb_pipeline

T0 = time.time()
RESULTS: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def clean(o):
    """Make results JSON-safe (numpy types -> Python, NaN -> null)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def records(df: pd.DataFrame, decimals: int = 4) -> list[dict]:
    return df.round(decimals).reset_index().to_dict(orient="records")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    ap.add_argument("--fast", action="store_true", help="subsample + tiny grids; smoke test only")
    ap.add_argument("--skip-extras", action="store_true", help="skip robustness / benchmark checks")
    args = ap.parse_args()
    fig_dir, res_dir = Path(args.figures_dir), Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. data
    log("1. loading data")
    train_official, test_official = load_adult_files(args.data_dir)
    df = load_adult(args.data_dir)
    RESULTS["data_quality"] = data_quality_report(df)
    q = RESULTS["data_quality"]
    RESULTS["n_records"], RESULTS["class_counts"], RESULTS["pos_share"] = q["n_records"], q["class_counts"], q["pos_share"]
    if args.fast:
        df = df.sample(6000, random_state=RANDOM_STATE).reset_index(drop=True)
        RESULTS["fast_mode_subsample"] = 6000
    plots.class_distribution(df, fig_dir)
    plots.missing_values(df, fig_dir)

    # ------------------------------------------------------------------ 2. split
    X_train, X_test, y_train, y_test = make_split(df)
    RESULTS["n_train"], RESULTS["n_test"] = int(len(X_train)), int(len(X_test))
    overlap = exact_overlap_mask(X_train, X_test)
    RESULTS["test_rows_identical_to_train_rows"] = int(overlap.sum())
    RESULTS["eda_train"] = analysis.eda_train_summary(X_train, y_train)
    plots.eda_target_relations(X_train, y_train, fig_dir)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    RESULTS["features_used"] = USED_COLS
    assert not {"race", "sex"} & set(USED_COLS), "race / sex must not be model features"

    # ------------------------------------------------------------------ 3. preprocessing probes (LR, CV on train)
    log("3. preprocessing probes")
    RESULTS["prep_choice"] = records(analysis.preprocessing_probes(X_train, y_train, cv))

    # ------------------------------------------------------------------ 4. model comparison (default settings)
    log("4. default-settings CV")
    models = default_models()
    cv_rows, oof = [], {}
    folds, _ = cross_validate_oof(baseline(), X_train, y_train, cv)
    cv_rows.append(summarise_folds("Majority-class baseline", folds))
    for name, pipe in models.items():
        folds, oof[name] = cross_validate_oof(pipe, X_train, y_train, cv)
        cv_rows.append(summarise_folds(name, folds))
        log(f"   {name}: F1 {cv_rows[-1]['f1']:.4f}")
    RESULTS["cv_default"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in cv_rows]
    plots.cv_confusion_roc(y_train, oof, fig_dir)

    # ------------------------------------------------------------------ 5. tuning
    log("5. tuning")
    searches, tuned_df, edge = analysis.tune_all(models, X_train, y_train, cv, fast=args.fast, log=log)
    RESULTS["cv_tuned"] = records(tuned_df)
    RESULTS["grid_edge_check"] = edge
    best_gb = {k.replace("clf__", ""): v for k, v in searches["Gradient Boosting"].best_params_.items()}
    if not args.fast:
        nb_df = analysis.gb_neighbour_settings(X_train, y_train, cv, searches["Gradient Boosting"].best_score_)
        RESULTS["gb_neighbour_settings"] = records(nb_df)

    # ------------------------------------------------------------------ 6. feature selection
    log("6. feature selection")
    fs_df, l1_coef = analysis.l1_vs_l2(X_train, y_train, cv)
    RESULTS["fs_lr"] = records(fs_df)
    plots.l1_coefficients(l1_coef, fig_dir)
    RESULTS["l1_top_coefficients"] = l1_coef[l1_coef != 0].sort_values().tail(6).round(2).to_dict()
    RESULTS["fs_gb"] = records(analysis.gb_selection_comparison(X_train, y_train, best_gb, cv))

    # ------------------------------------------------------------------ 7. final model + threshold (train only)
    log("7. final model and threshold")
    final_model = gb_pipeline(best_gb, with_selection=True)
    oof_final = cross_validate_oof(final_model, X_train, y_train, cv)[1]
    THRESHOLD, f1_scores = select_threshold(y_train, oof_final)
    RESULTS["threshold"] = THRESHOLD
    RESULTS["oof_f1_at_0.50"] = float(f1_scores[list(THRESHOLD_GRID).index(0.5)])
    RESULTS["oof_f1_at_threshold"] = float(f1_scores.max())
    plots.threshold_curves(y_train, oof_final, THRESHOLD_GRID, f1_scores, THRESHOLD, fig_dir)
    final_model = clone(final_model).fit(X_train, y_train)
    sel = final_model.named_steps["select"]
    RESULTS["final_model_details"] = {
        "gb_params": best_gb, "selector_threshold": float(sel.threshold_),
        "features_kept": int(sel.get_support().sum()),
        "features_total": int(len(final_model.named_steps["prep"].get_feature_names_out()))}

    # ------------------------------------------------------------------ 8. the single test evaluation
    log("8. test evaluation (used once)")
    y_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)
    RESULTS["test"] = {"model": "Gradient Boosting + L1 feature selection",
                       **evaluate_at_threshold(y_test, y_proba, THRESHOLD)}
    RESULTS["test_ci95"] = bootstrap_metric_ci(y_test, y_proba, THRESHOLD, n_boot=200 if args.fast else 1000)
    plots.final_test(y_test, y_pred, y_proba, fig_dir)
    plots.precision_recall(y_test, y_proba, THRESHOLD, fig_dir)
    log(f"   test F1 {RESULTS['test']['f1']:.4f}  AUC {RESULTS['test']['roc_auc']:.4f}")

    # ------------------------------------------------------------------ 9. error analysis (descriptive)
    ep = analysis.error_profile(X_test, y_test, y_pred, y_proba)
    RESULTS["errors"] = ep["profile"].to_dict()
    RESULTS["errors_occupation_top"], RESULTS["errors_mean_proba"] = ep["occupation_top"], ep["mean_proba"]
    pi_s = analysis.permutation_importance_train(final_model, X_train, y_train)
    RESULTS["perm_importance"] = pi_s.round(4).to_dict()
    plots.permutation_importance_plot(pi_s, fig_dir)

    # ------------------------------------------------------------------ 10. fairness inspection (sex / race NOT features)
    log("10. fairness inspection")
    nb = 200 if args.fast else 1000
    sex_rep = group_report(y_test, y_pred, X_test["sex"], "sex")
    race_rep = group_report(y_test, y_pred, X_test["race"], "race")
    RESULTS["fair_sex"], RESULTS["fair_race"] = records(sex_rep, 3), records(race_rep, 3)
    ci_sex = group_report_ci(y_test, y_pred, X_test["sex"], "sex", n_boot=nb)
    ci_race = group_report_ci(y_test, y_pred, X_test["race"], "race", n_boot=nb)
    RESULTS["fair_sex_ci95"], RESULTS["fair_race_ci95"] = records(ci_sex), records(ci_race)
    RESULTS["sex_gaps_male_minus_female_ci95"] = gap_ci(y_test, y_pred, X_test["sex"], "Male", "Female", n_boot=2 * nb)
    RESULTS["race_gaps_white_minus_black_ci95"] = gap_ci(y_test, y_pred, X_test["race"], "White", "Black", n_boot=2 * nb)
    plots.fairness_groups(ci_sex, ci_race, fig_dir)

    # ablation: do marital-status / relationship act as proxies for sex? (training OOF only)
    RESULTS["ablation_sex_proxy"] = records(
        analysis.sex_proxy_ablation(X_train, y_train, best_gb, cv, THRESHOLD, oof_final), 3)
    RESULTS["relationship_by_sex_train"] = pd.crosstab(X_train["relationship"], X_train["sex"]).to_dict()

    # ------------------------------------------------------------------ 11. audit extras
    if not args.skip_extras:
        log("11. audit extras")
        RESULTS["test_f1_excluding_overlap_rows"] = float(evaluate_at_threshold(y_test.values[~overlap], y_proba[~overlap], THRESHOLD)["f1"])
        stab = analysis.selection_stability(X_train, y_train, n_boot=5 if args.fast else 30)
        RESULTS["selection_stability"] = {
            "n_resamples": (5 if args.fast else 30) + N_FOLDS,
            "always_selected": int((stab.selection_frequency == 1).sum()),
            "never_selected": int((stab.selection_frequency == 0).sum()),
            "sometimes_selected": int(((stab.selection_frequency > 0) & (stab.selection_frequency < 1)).sum()),
            "sometimes_selected_features": records(stab[(stab.selection_frequency > 0) & (stab.selection_frequency < 1)].set_index("feature"), 3)}
        RESULTS["selector_scaling_sensitivity"] = analysis.scaled_selector_sensitivity(X_train, y_train, best_gb, cv)
        RESULTS["histgb_benchmark_cv"] = analysis.hist_gb_benchmark(X_train, y_train, cv)
        if not args.fast:
            RESULTS["official_split_check"] = analysis.official_split_check(train_official, test_official, best_gb)
            RESULTS["seed_sensitivity"] = analysis.seed_sensitivity(df, best_gb)
            ss = pd.DataFrame(RESULTS["seed_sensitivity"])
            RESULTS["seed_sensitivity_summary"] = {c: {"mean": float(ss[c].mean()), "std": float(ss[c].std(ddof=1))} for c in ("precision", "recall", "f1", "roc_auc")}

    # ------------------------------------------------------------------ 12. environment + write
    RESULTS["environment"] = {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    RESULTS["runtime_s"] = round(time.time() - T0)
    out = res_dir / ("results_fast_smoke_test.json" if args.fast else "results.json")
    out.write_text(json.dumps(clean(RESULTS), indent=1))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
