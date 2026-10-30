"""Figure helpers. Each function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             precision_score, recall_score)

BLUE, ORANGE, RED, GREY = "#4C72B0", "#DD8452", "#C44E52", "#8c8c8c"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.labelsize": 10})


def _save(fig, out_dir, name) -> Path:
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def class_distribution(df, out_dir):
    counts = df["income"].value_counts()
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(counts.index, counts.values, color=[BLUE, ORANGE])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Class distribution (income)"); ax.set_ylabel("records")
    return _save(fig, out_dir, "class_distribution.png")


def missing_values(df, out_dir):
    feats = [c for c in df.columns if c not in ("income", "target")]
    miss = (df[feats].isna().mean() * 100).sort_values()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.barh(miss.index, miss.values, color=BLUE)
    for i, (c, v) in enumerate(miss.items()):
        ax.text(v, i, f" {v:.2f}% ({int(df[c].isna().sum()):,})", va="center", fontsize=9)
    ax.set_xlim(0, miss.max() * 1.4); ax.set_xlabel("missing values (% of 48,842 records)")
    ax.set_title("Missing values ('?' read as NaN)")
    return _save(fig, out_dir, "missing_values.png")


def eda_target_relations(X_train, y_train, out_dir):
    tr = X_train.assign(target=y_train.values)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    tr.groupby(pd.cut(tr["age"], [0, 25, 35, 45, 55, 65, 100]), observed=True)["target"].mean() \
        .plot(kind="bar", ax=axes[0], title="P(>50K) by age group", color=BLUE)
    tr.groupby("education-num")["target"].mean().plot(kind="bar", ax=axes[1], title="P(>50K) by education-num", color=BLUE)
    tr.groupby(pd.cut(tr["hours-per-week"], [0, 30, 40, 50, 100]), observed=True)["target"].mean() \
        .plot(kind="bar", ax=axes[2], title="P(>50K) by hours per week", color=BLUE)
    tr.groupby("marital-status")["target"].mean().sort_values() \
        .plot(kind="barh", ax=axes[3], title="P(>50K) by marital status", color=BLUE)
    for a in axes[:3]:
        a.set_xlabel("")
    fig.tight_layout()
    return _save(fig, out_dir, "eda_target_relations.png")


def cv_confusion_roc(y_train, oof: dict, out_dir):
    n = len(oof)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.3 * (n + 1), 3.8))
    for ax, (name, p) in zip(axes[:n], oof.items()):
        ConfusionMatrixDisplay.from_predictions(y_train, (p >= 0.5).astype(int), display_labels=["<=50K", ">50K"],
                                                ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name} (threshold 0.5)")
    for name, p in oof.items():
        RocCurveDisplay.from_predictions(y_train, p, ax=axes[n], name=name)
    axes[n].plot([0, 1], [0, 1], "k--", lw=0.8); axes[n].set_title("ROC curves (out-of-fold)")
    fig.tight_layout()
    return _save(fig, out_dir, "cv_confusion_roc.png")


def threshold_curves(y_train, oof_final, grid, f1_scores, threshold, out_dir):
    y = np.asarray(y_train)
    prec = [precision_score(y, (oof_final >= t).astype(int), zero_division=0) for t in grid]
    rec = [recall_score(y, (oof_final >= t).astype(int)) for t in grid]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(grid, f1_scores, color=BLUE, lw=2, label="F1")
    ax.plot(grid, prec, color=ORANGE, lw=1.2, label="precision")
    ax.plot(grid, rec, color=RED, lw=1.2, label="recall")
    ax.axvline(threshold, color="k", ls="--", lw=1, label=f"chosen = {threshold:.2f}")
    ax.axvline(0.5, color=GREY, ls=":", label="default 0.50")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("out-of-fold score (>50K class, training set)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out_dir, "threshold.png")


def l1_coefficients(coef: pd.Series, out_dir, C=0.1):
    nz = coef[coef != 0].sort_values()
    top = pd.concat([nz.head(10), nz.tail(10)])
    fig, ax = plt.subplots(figsize=(7, 6))
    top.plot(kind="barh", ax=ax, color=[RED if v < 0 else BLUE for v in top])
    ax.set_title(f"L1 logistic regression (C={C}): 10 most negative / positive coefficients")
    fig.tight_layout()
    return _save(fig, out_dir, "l1_coefficients.png")


def final_test(y_test, y_pred, y_proba, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["<=50K", ">50K"], ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion matrix (counts)")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["<=50K", ">50K"], normalize="true",
                                            values_format=".2f", ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("Confusion matrix (row-normalised)")
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[2], name="Final model")
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8); axes[2].set_title("ROC curve (test)")
    fig.tight_layout()
    return _save(fig, out_dir, "final_test.png")


def precision_recall(y_test, y_proba, threshold, out_dir):
    y = np.asarray(y_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y, y_proba, ax=ax, name="Final model")
    pred = (np.asarray(y_proba) >= threshold).astype(int)
    ax.plot(recall_score(y, pred), precision_score(y, pred), "o", color=RED, label=f"threshold {threshold:.2f}")
    ax.axhline(y.mean(), color=GREY, ls="--", lw=0.8, label=f"prevalence {y.mean():.2f}")
    ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.set_title("Precision-recall curve (test)")
    return _save(fig, out_dir, "precision_recall_curve.png")


def permutation_importance_plot(series: pd.Series, out_dir):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    series.sort_values().plot(kind="barh", ax=ax, color=BLUE)
    ax.set_title("Permutation importance (drop in ROC AUC)")
    fig.tight_layout()
    return _save(fig, out_dir, "permutation_importance.png")


def fairness_groups(ci_sex: pd.DataFrame, ci_race: pd.DataFrame, out_dir):
    """Recall and false-positive rate per group with 95% bootstrap CIs; small groups hatched."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for row, (tbl, key) in enumerate(((ci_sex, "sex"), (ci_race, "race"))):
        for col, metric in enumerate(("recall", "FPR")):
            ax = axes[row, col]
            d = tbl[tbl["metric"] == metric].reset_index(drop=True)
            err = np.vstack([d["estimate"] - d["lo"], d["hi"] - d["estimate"]])
            bars = ax.bar(d[key], d["estimate"], yerr=err, capsize=3, color=BLUE if col == 0 else ORANGE)
            for b, small in zip(bars, d["small_group"]):
                if small:
                    b.set_hatch("//"); b.set_alpha(0.6)
            ax.set_title(f"{metric} by {key} (95% bootstrap CI)")
            ax.set_ylim(0, 1 if metric == "recall" else max(0.35, float(d["hi"].max()) * 1.1))
            ax.tick_params(axis="x", labelrotation=20 if key == "race" else 0)
    fig.text(0.5, -0.01, "Hatched bars: groups with fewer than 150 test records (very noisy).", ha="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "fairness_groups.png")
