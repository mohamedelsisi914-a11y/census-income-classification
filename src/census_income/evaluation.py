"""Cross-validation, threshold selection and held-out evaluation helpers."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from .config import METRICS, RANDOM_STATE, THRESHOLD_GRID


def fold_metrics(y_true, y_pred, proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def cross_validate_oof(estimator, X: pd.DataFrame, y: pd.Series, cv):
    """Manual CV loop: per-fold metrics (at the model's own 0.5 decision rule) plus
    out-of-fold probabilities. Every fold fits a fresh clone on the training part only."""
    rows, oof = [], np.full(len(y), np.nan)
    for tr, va in cv.split(X, y):
        est = clone(estimator)
        t0 = time.perf_counter()
        est.fit(X.iloc[tr], y.iloc[tr])
        fit_time = time.perf_counter() - t0
        proba = est.predict_proba(X.iloc[va])[:, 1]
        oof[va] = proba
        row = fold_metrics(y.iloc[va], est.predict(X.iloc[va]), proba)
        row["fit_time_s"] = fit_time
        rows.append(row)
    return pd.DataFrame(rows), oof


def summarise_folds(name: str, folds: pd.DataFrame) -> dict:
    """Mean and population std (ddof=0, as in the original report) over the CV folds."""
    row = {"model": name}
    for m in METRICS:
        row[m] = float(folds[m].mean())
        row[m + "_std"] = float(folds[m].std(ddof=0))
    row["fit_time_s"] = float(folds["fit_time_s"].mean())
    return row


def f1_by_threshold(y_true, proba, grid=THRESHOLD_GRID) -> np.ndarray:
    y = np.asarray(y_true).astype(bool)
    out = []
    for t in grid:
        p = np.asarray(proba) >= t
        tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
        denom = 2 * tp + fp + fn
        out.append(2 * tp / denom if denom else 0.0)
    return np.array(out)


def select_threshold(y_true, proba, grid=THRESHOLD_GRID) -> tuple[float, np.ndarray]:
    """Threshold maximising F1 of the positive class. Must be called on TRAINING-set
    out-of-fold probabilities only. Ties go to the lowest threshold."""
    scores = f1_by_threshold(y_true, proba, grid)
    return float(grid[int(scores.argmax())]), scores


def evaluate_at_threshold(y_true, proba, threshold: float) -> dict:
    y = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "threshold": float(threshold),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "avg_precision": average_precision_score(y, proba),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def bootstrap_metric_ci(y_true, proba, threshold: float, n_boot: int = 1000,
                        seed: int = RANDOM_STATE, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI over test rows (the fitted model and threshold stay fixed, so this
    captures test-sample noise only, not training variability)."""
    y, p = np.asarray(y_true), np.asarray(proba)
    rng = np.random.default_rng(seed)
    keys = ["precision", "recall", "f1", "accuracy", "roc_auc", "avg_precision"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = evaluate_at_threshold(y[idx], p[idx], threshold)
        for k in keys:
            draws[k].append(m[k])
    point = evaluate_at_threshold(y, p, threshold)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: {"estimate": float(point[k]), "lo": float(np.percentile(v, lo)),
                "hi": float(np.percentile(v, hi))} for k, v in draws.items()}
