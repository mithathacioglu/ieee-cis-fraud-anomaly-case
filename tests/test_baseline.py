import numpy as np
import pytest
from supervised_baseline import bootstrap_gap, evaluate, top_k_hits


def test_fixed_budget_counts_only_the_top_slice():
    y = np.array([1, 0, 1, 0, 1])
    scores = np.array([.9, .8, .7, .6, .5])
    ids = np.arange(5)
    assert top_k_hits(y, scores, ids, 1) == 1
    assert top_k_hits(y, scores, ids, 3) == 2
    assert top_k_hits(y, scores, ids, 5) == 3


def test_ties_break_on_transaction_id_not_row_order():
    y = np.array([0, 1])
    scores = np.array([.5, .5])
    # Esit skorda kucuk ID once gelir; siralamayi ters cevirmek sonucu degistirmemeli
    assert top_k_hits(y, scores, np.array([2, 1]), 1) == 1
    assert top_k_hits(y[::-1], scores, np.array([1, 2]), 1) == 1


def test_evaluate_reports_every_budget_and_orders_them():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < .1).astype(int)
    scores = y + rng.normal(0, .5, 1000)
    row = evaluate(y, scores, np.arange(1000))
    assert set(row["budgets"]) == {"0.01", "0.05", "0.1"}
    assert row["budgets"]["0.01"]["reviews"] == 10
    # Daha genis butce daha az fraud yakalayamaz
    counts = [row["budgets"][k]["tp"] for k in ("0.01", "0.05", "0.1")]
    assert counts == sorted(counts)
    assert .5 <= row["roc_auc"] <= 1


def test_bootstrap_interval_contains_the_observed_gap():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < .05).astype(int)
    strong = y + rng.normal(0, .4, 2000)
    weak = rng.normal(0, 1, 2000)
    gap = bootstrap_gap(y, weak, strong, np.arange(2000))
    assert gap["ci_low"] <= gap["observed_gap"] <= gap["ci_high"]
    assert gap["observed_gap"] > 0 and gap["excludes_zero"]


def test_identical_rankings_produce_a_gap_interval_around_zero():
    rng = np.random.default_rng(2)
    y = (rng.random(1500) < .08).astype(int)
    same = rng.normal(0, 1, 1500)
    gap = bootstrap_gap(y, same, same.copy(), np.arange(1500))
    assert gap["observed_gap"] == 0
    assert gap["ci_low"] <= 0 <= gap["ci_high"]
    assert not gap["excludes_zero"]


def test_result_does_not_depend_on_row_order():
    rng = np.random.default_rng(3)
    y = (rng.random(800) < .1).astype(int)
    scores = y + rng.normal(0, .6, 800)
    ids = np.arange(800)
    shuffle = rng.permutation(800)
    first = evaluate(y, scores, ids)
    second = evaluate(y[shuffle], scores[shuffle], ids[shuffle])
    assert first["budgets"] == second["budgets"]
    assert first["roc_auc"] == pytest.approx(second["roc_auc"])
