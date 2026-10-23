"""Checks for evaluate.py on synthetic data where the right answer is known. Run: pytest test_evaluate.py"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from evaluate import (cutoff_at_specificity, point_metrics, bootstrap_indices, evaluate,
                      paired_difference, fmt_ci)

rng = np.random.default_rng(0)


def make_data(n=2000, sep=1.5, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(r.normal(0, 1, n) + sep * (y - 0.5))))
    return y, p


def test_cutoff_gives_at_least_95_specificity_on_the_set_it_was_chosen_on():
    y, p = make_data()
    c = cutoff_at_specificity(y, p)
    n_neg = (y == 0).sum()
    # continuous scores: exactly ceil(0.95 * n) normals sit at or below the smallest valid cutoff
    assert (p[y == 0] <= c).sum() == np.ceil(0.95 * n_neg)


def test_cutoff_with_ties_never_drops_below_target():
    y = np.array([0] * 100 + [1] * 10)
    p = np.array([0.2] * 90 + [0.7] * 10 + [0.9] * 10)   # 10% of normals tie at the top
    c = cutoff_at_specificity(y, p)
    assert (p[y == 0] <= c).mean() >= 0.95
    # tree models produce tied scores: the decision rule (p > cutoff) must not flag the tied normals
    assert point_metrics(y, p, c)["specificity"] >= 0.95


def test_point_metrics_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = point_metrics(y, p, cutoff=0.5)
    assert m["AUROC"] == 1 and m["sensitivity"] == 1 and m["specificity"] == 1 and m["macro_F1"] == 1


def test_point_metrics_matches_sklearn_and_hand_counts():
    y, p = make_data(500)
    m = point_metrics(y, p, cutoff=0.6)
    assert m["AUROC"] == pytest.approx(roc_auc_score(y, p))
    assert m["sensitivity"] == pytest.approx(((p > 0.6) & (y == 1)).sum() / (y == 1).sum())
    assert m["specificity"] == pytest.approx(((p <= 0.6) & (y == 0)).sum() / (y == 0).sum())


def test_category_recall_is_fraction_of_carriers_flagged():
    y = np.array([0, 1, 1, 1, 1])
    p = np.array([0.1, 0.9, 0.9, 0.2, 0.2])
    mi = np.array([0, 1, 1, 0, 0], bool)          # both MI recordings flagged
    hyp = np.array([0, 0, 1, 1, 1], bool)         # 1 of 3 HYP recordings flagged
    m = point_metrics(y, p, 0.5, {"MI": mi, "HYP": hyp})
    assert m["recall_MI"] == 1.0 and m["recall_HYP"] == pytest.approx(1 / 3)


def test_bootstrap_keeps_each_patients_recordings_together():
    groups = np.array([10, 10, 10, 20, 30, 30])
    for idx in bootstrap_indices(groups, n_boot=50, seed=1):
        counts = {g: (groups[idx] == g).sum() for g in (10, 20, 30)}
        assert counts[10] % 3 == 0 and counts[30] % 2 == 0
        assert len(idx) >= 1


def test_bootstrap_is_reproducible_with_seed():
    g = np.arange(100)
    a, b = bootstrap_indices(g, 20, seed=3), bootstrap_indices(g, 20, seed=3)
    assert all((x == y).all() for x, y in zip(a, b))


def test_evaluate_ci_brackets_point_estimate_and_random_model_is_near_half():
    y, p = make_data(3000)
    r = evaluate(y[:1500], p[:1500], y[1500:], p[1500:], n_boot=200)
    for k in ["AUROC", "sensitivity", "specificity"]:
        assert r["ci_low"][k] <= r["point"][k] <= r["ci_high"][k]
    noise = rng.random(3000)
    r0 = evaluate(y[:1500], noise[:1500], y[1500:], noise[1500:], n_boot=200)
    assert r0["ci_low"]["AUROC"] < 0.5 < r0["ci_high"]["AUROC"]


def test_evaluate_specificity_on_test_is_close_to_target():
    y, p = make_data(20000)
    r = evaluate(y[:10000], p[:10000], y[10000:], p[10000:], n_boot=50)
    assert abs(r["point"]["specificity"] - 0.95) < 0.015
    assert r["confusion"].sum() == 10000


def test_paired_difference_zero_for_identical_models_and_detects_a_real_gap():
    y, good = make_data(4000, sep=2.5, seed=1)
    _, bad = make_data(4000, sep=0.5, seed=1)
    _, good2 = y, good.copy()
    boot = bootstrap_indices(np.arange(2000), 200, seed=0)
    same = paired_difference(y[:2000], good[:2000], good2[:2000], y[2000:], good[2000:], good2[2000:], {}, boot)
    assert (same["delta"].abs() < 1e-12).all() and not same["significant"].any()
    gap = paired_difference(y[:2000], good[:2000], bad[:2000], y[2000:], good[2000:], bad[2000:], {}, boot)
    assert gap.loc["AUROC", "delta"] > 0 and gap.loc["AUROC", "significant"]


def test_fmt_ci():
    assert fmt_ci(0.9271, 0.915, 0.938) == "0.927 [0.915, 0.938]"
    assert fmt_ci(0.745, 0.71, 0.779, pct=True) == "74.5% [71.0, 77.9]"
