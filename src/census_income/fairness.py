"""Group-level error rates with bootstrap confidence intervals (inspection only: sensitive
attributes are never model inputs)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE

MIN_RELIABLE_GROUP = 150


def _safe_div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b > 0)


def _rates(yt, yp, axis=-1) -> dict:
    """Rates from boolean arrays; works for 1-D input or (B, n) bootstrap matrices."""
    tp = (yp & yt).sum(axis=axis); fp = (yp & ~yt).sum(axis=axis)
    fn = (~yp & yt).sum(axis=axis); tn = (~yp & ~yt).sum(axis=axis)
    return {
        "actual >50K": yt.mean(axis=axis), "predicted >50K": yp.mean(axis=axis),
        "precision": _safe_div(tp, tp + fp), "recall": _safe_div(tp, tp + fn),
        "FPR": _safe_div(fp, fp + tn), "FNR": _safe_div(fn, tp + fn),
    }


def group_report(y_true, y_pred, groups: pd.Series, name: str) -> pd.DataFrame:
    """Per-group counts and error rates, largest group first."""
    yt, yp, g = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool), np.asarray(groups)
    rows = []
    for level in pd.unique(g):
        m = g == level
        rows.append({name: level, "n": int(m.sum()), **_rates(yt[m], yp[m])})
    return pd.DataFrame(rows).set_index(name).sort_values("n", ascending=False).round(3)


def _boot(yt, yp, n_boot, rng) -> dict:
    n = len(yt)
    idx = rng.integers(0, n, (n_boot, n))
    return _rates(yt[idx], yp[idx], axis=1)


def group_report_ci(y_true, y_pred, groups: pd.Series, name: str, n_boot: int = 1000,
                    seed: int = RANDOM_STATE, alpha: float = 0.05) -> pd.DataFrame:
    """Long table: one row per group x metric with estimate and percentile CI."""
    yt, yp, g = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool), np.asarray(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for level in pd.unique(g):
        m = g == level
        point = _rates(yt[m], yp[m])
        boot = _boot(yt[m], yp[m], n_boot, rng)
        for metric in ("precision", "recall", "FPR", "FNR", "predicted >50K"):
            lo, hi = np.nanpercentile(boot[metric], [100 * alpha / 2, 100 * (1 - alpha / 2)])
            rows.append({name: level, "n": int(m.sum()), "metric": metric,
                         "estimate": float(point[metric]), "lo": float(lo), "hi": float(hi),
                         "small_group": bool(m.sum() < MIN_RELIABLE_GROUP)})
    return pd.DataFrame(rows)


def gap_ci(y_true, y_pred, groups: pd.Series, group_a: str, group_b: str, n_boot: int = 2000,
           seed: int = RANDOM_STATE, alpha: float = 0.05) -> dict:
    """Difference (a - b) in recall, FPR and predicted-positive rate, plus the FPR ratio, with
    percentile CIs. The two groups are resampled independently."""
    yt, yp, g = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool), np.asarray(groups)
    rng = np.random.default_rng(seed)
    ma, mb = g == group_a, g == group_b
    pa, pb = _rates(yt[ma], yp[ma]), _rates(yt[mb], yp[mb])
    ba, bb = _boot(yt[ma], yp[ma], n_boot, rng), _boot(yt[mb], yp[mb], n_boot, rng)
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for metric in ("recall", "FPR", "predicted >50K"):
        d = ba[metric] - bb[metric]
        out[f"{metric} gap"] = {"estimate": float(pa[metric] - pb[metric]),
                                "lo": float(np.nanpercentile(d, lo_q)), "hi": float(np.nanpercentile(d, hi_q))}
    ratio = _safe_div(ba["FPR"], bb["FPR"])
    out["FPR ratio"] = {"estimate": float(pa["FPR"] / pb["FPR"]),
                        "lo": float(np.nanpercentile(ratio, lo_q)), "hi": float(np.nanpercentile(ratio, hi_q))}
    out["groups"] = {"a": group_a, "b": group_b, "n_a": int(ma.sum()), "n_b": int(mb.sum())}
    return out
