"""Shared evaluation: every model (baselines and CNN) is scored by exactly this code.

Protocol
  1. Cutoff: chosen on the black-box set so that at least 95% of normal recordings score at or below it.
  2. Decision: a recording is flagged abnormal when p > cutoff. The cutoff is never re-tuned on test.
  3. Metrics on test: AUROC, AUPRC, sensitivity, specificity, macro F1, and per-category recall
     (fraction of recordings carrying each abnormal category that get flagged).
  4. 95% CIs: bootstrap that resamples *patients* (not recordings), because a patient's
     recordings are correlated. Paired differences reuse the same resamples for both models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

TARGET_SPECIFICITY = 0.95


def cutoff_at_specificity(y, p, spec=TARGET_SPECIFICITY):
    """Smallest cutoff such that at least `spec` of normal recordings (y == 0) have p <= cutoff."""
    neg = np.sort(np.asarray(p)[np.asarray(y) == 0])
    assert len(neg), "need normal recordings to set a specificity cutoff"
    k = int(np.ceil(spec * len(neg)))
    return float(neg[k - 1])


def point_metrics(y, p, cutoff, cats=None):
    """Metrics for one set of predictions. `cats` maps category name -> boolean mask of carriers."""
    y, p = np.asarray(y), np.asarray(p)
    flag = p > cutoff
    out = {"AUROC": roc_auc_score(y, p),
           "AUPRC": average_precision_score(y, p),
           "sensitivity": flag[y == 1].mean(),
           "specificity": 1 - flag[y == 0].mean(),
           "macro_F1": f1_score(y, flag, average="macro")}
    for c, m in (cats or {}).items():
        m = np.asarray(m, dtype=bool)
        out[f"recall_{c}"] = flag[m].mean() if m.any() else np.nan
    return out


def bootstrap_indices(groups, n_boot=1000, seed=0):
    """Index arrays for a cluster bootstrap: resample groups (patients) with replacement,
    keep all of each sampled group's rows. Pass the result to several evaluations to pair them."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    sizes = np.bincount(inv, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        g = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([order[starts[i]:starts[i] + sizes[i]] for i in g]))
    return out


def _ci(samples, alpha=0.05):
    s = pd.DataFrame(samples)
    return s.quantile(alpha / 2), s.quantile(1 - alpha / 2)


def evaluate(y_val, p_val, y_test, p_test, cats_test=None, groups_test=None, boot=None,
             n_boot=1000, seed=0):
    """Full protocol for one model. Returns a dict:
       cutoff, point (Series), ci_low (Series), ci_high (Series), confusion (2x2 array: rows true 0/1)."""
    cutoff = cutoff_at_specificity(y_val, p_val)
    y_test, p_test = np.asarray(y_test), np.asarray(p_test)
    cats_test = {c: np.asarray(m, dtype=bool) for c, m in (cats_test or {}).items()}
    point = pd.Series(point_metrics(y_test, p_test, cutoff, cats_test))

    if boot is None:
        groups = np.arange(len(y_test)) if groups_test is None else groups_test
        boot = bootstrap_indices(groups, n_boot, seed)
    samples = [point_metrics(y_test[i], p_test[i], cutoff, {c: m[i] for c, m in cats_test.items()})
               for i in boot]
    lo, hi = _ci(samples)
    return {"cutoff": cutoff, "point": point, "ci_low": lo, "ci_high": hi,
            "confusion": confusion_matrix(y_test, p_test > cutoff, labels=[0, 1])}


def paired_difference(y_val, pa_val, pb_val, y_test, pa_test, pb_test, cats_test, boot):
    """Metric(A) - Metric(B) on the same test recordings, each model using its own black-box cutoff.
    Returns a DataFrame (rows = metrics) with delta, 95% CI and whether the CI excludes 0."""
    ca, cb = cutoff_at_specificity(y_val, pa_val), cutoff_at_specificity(y_val, pb_val)
    y_test, pa_test, pb_test = map(np.asarray, (y_test, pa_test, pb_test))
    cats_test = {c: np.asarray(m, dtype=bool) for c, m in cats_test.items()}

    def diff(i):
        sub = {c: m[i] for c, m in cats_test.items()}
        a, b = point_metrics(y_test[i], pa_test[i], ca, sub), point_metrics(y_test[i], pb_test[i], cb, sub)
        return {k: a[k] - b[k] for k in a}

    point = pd.Series(diff(np.arange(len(y_test))))
    lo, hi = _ci([diff(i) for i in boot])
    return pd.DataFrame({"delta": point, "ci_low": lo, "ci_high": hi,
                         "significant": (lo > 0) | (hi < 0)})


def fmt_ci(point, lo, hi, pct=False):
    """'0.927 [0.915, 0.938]' or '74.5% [71.0, 77.9]'."""
    if pct:
        return f"{point:.1%} [{lo * 100:.1f}, {hi * 100:.1f}]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"
