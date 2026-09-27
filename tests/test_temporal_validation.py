from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from supervised_baseline import sensitivity_statement
from temporal_validation import (
    COMPARISONS,
    chronological_folds,
    feature_sets,
    hybrid_ranking,
    load_development_data,
)

from fraud_case.anomaly import REQUIRED_FEATURES
from fraud_case.evaluation import paired_block_intervals
from fraud_case.features import FEATURE_SPECS


def metric_row():
    return {"average_precision": .2, "roc_auc": .7,
            "budgets": {str(rate): {"tp": count} for rate, count in ((.01, 1), (.05, 4), (.1, 8))}}


@pytest.mark.parametrize(("change", "expected"), [
    ({}, "0 artış, 0 düşüş, 5 eşitlik"),
    ({"average_precision": .3}, "1 artış, 0 düşüş, 4 eşitlik"),
    ({"roc_auc": .6}, "0 artış, 1 düşüş, 4 eşitlik"),
    ({"average_precision": .3, "roc_auc": .6}, "1 artış, 1 düşüş, 3 eşitlik"),
])
def test_ablation_comment_handles_equality_and_auc(change, expected):
    full = metric_row()
    ablated = {**deepcopy(full), **change}
    assert expected in sensitivity_statement("Model", full, ablated)


def test_budget_decline_cannot_be_hidden_by_ap_improvement():
    full = metric_row()
    ablated = deepcopy(full)
    ablated["average_precision"] += .1
    ablated["budgets"]["0.05"]["tp"] -= 1
    assert "farklı yönlerde" in sensitivity_statement("Model", full, ablated)


def source_frame():
    times = np.repeat(np.arange(20) * 86400, 2)
    return pd.DataFrame({"TransactionID": np.arange(len(times)), "TransactionDT": times,
                         "split": ["train"] * 20 + ["validation"] * 20, "isFraud": np.arange(40) % 2})


def test_folds_keep_ties_together_cover_validation_once_and_honor_delay():
    source = source_frame()
    folds = chronological_folds(source, 3, 2)
    covered = []
    for _, train, audit in folds:
        covered.extend(audit)
        assert source.TransactionDT.iloc[train].max() < source.TransactionDT.iloc[audit].min() - 2 * 86400
        assert not set(train) & set(audit)
        for timestamp in source.TransactionDT.iloc[audit].unique():
            assert set(source.index[source.TransactionDT.eq(timestamp)]).issubset(audit)
    assert sorted(covered) == source.index[source.split.eq("validation")].tolist()


def test_final_test_rows_are_rejected_by_fold_builder():
    source = source_frame()
    source.loc[source.index[-1], "split"] = "test"
    with pytest.raises(ValueError, match="Only train and validation"):
        chronological_folds(source, 3, 0)


def test_loading_excludes_final_test_and_rejects_misalignment(tmp_path):
    source = source_frame()
    source.loc[source.index[-2:], "split"] = "test"
    features = pd.DataFrame({name: np.arange(len(source), dtype=float) for name in FEATURE_SPECS})
    features["TransactionID"] = source.TransactionID
    features["split"] = source.split
    # Test labels are deliberately invalid: loading development must never validate them.
    source.loc[source.split.eq("test"), "isFraud"] = 999
    source.to_parquet(tmp_path / "transactions.parquet")
    features.to_parquet(tmp_path / "features.parquet")
    loaded, rows = load_development_data(tmp_path)
    assert len(rows) == len(loaded) == len(source) - 2
    assert set(rows.isFraud) == {0, 1}
    features.loc[0, "TransactionID"] = 999
    features.to_parquet(tmp_path / "features.parquet")
    with pytest.raises(ValueError, match="aligned"):
        load_development_data(tmp_path)


def test_feature_variants_remove_both_proxies_and_match_anomaly_inputs():
    train = pd.DataFrame({name: [1., 2., 3.] for name in FEATURE_SPECS})
    train["hour"] = np.nan
    columns = feature_sets(train)
    assert "hour" not in columns["full"]
    assert {"prior_global_count", "relative_day"}.issubset(columns["full"])
    assert not {"prior_global_count", "relative_day"} & set(columns["without_time"])
    assert set(columns["matched"]) == set(REQUIRED_FEATURES)


def score_variants(values):
    return {name: values.copy() for pair in COMPARISONS.values() for name in pair}


def test_block_bootstrap_preserves_pairing_with_unequal_block_sizes():
    times = np.repeat(np.arange(10) * 86400, np.arange(1, 11)).astype(float)
    y = np.arange(len(times)) % 2
    scores = score_variants(np.linspace(0, 1, len(times)))
    result = paired_block_intervals(y, scores, np.arange(len(y)), times, COMPARISONS, hours=24,
                                    resamples=100, min_blocks=8, rate=.05, seed=42)
    assert result["blocks"] == 10
    assert result["sample_rows_min"] < result["sample_rows_max"]
    for gap in result["comparisons"].values():
        assert gap["observed_tp_gap"] == gap["ci_low"] == gap["ci_high"] == 0


def test_block_bootstrap_perfect_ranker_gap_is_positive():
    times = np.repeat(np.arange(10) * 86400, 10).astype(float)
    y = np.tile([1] * 5 + [0] * 5, 10)
    scores = score_variants(-y.astype(float))
    scores["matched_gradient_boosting"] = y.astype(float)
    result = paired_block_intervals(y, scores, np.arange(len(y)), times, COMPARISONS, hours=24,
                                    resamples=100, min_blocks=8, rate=.1, seed=42)
    gap = result["comparisons"]["matched_gb_minus_raw"]
    assert gap["observed_tp_gap"] == gap["ci_low"] == gap["ci_high"] == 10


def test_insufficient_blocks_do_not_produce_a_confidence_interval():
    y = np.array([0, 1])
    result = paired_block_intervals(y, score_variants(y.astype(float)), np.arange(2), np.array([0., 1.]),
                                    COMPARISONS, hours=24, resamples=100, min_blocks=8, rate=.05, seed=42)
    assert result["status"] == "insufficient_blocks"
    assert result["comparisons"] == {}


def test_bootstrap_rejects_comparisons_that_name_an_absent_ranking():
    """Genel fonksiyon artık iki deney tarafından kullanılıyor; sessizce eksik
    sıralamayla çalışmaması gerekir."""
    times = np.repeat(np.arange(10) * 86400, 10).astype(float)
    y = np.arange(len(times)) % 2
    scores = {"raw": np.linspace(0, 1, len(times))}
    with pytest.raises(ValueError, match="unknown rankings"):
        paired_block_intervals(y, scores, np.arange(len(y)), times, {"a_minus_raw": ("raw", "absent")},
                               hours=24, resamples=10, min_blocks=8, rate=.05, seed=42)
    with pytest.raises(ValueError, match="unknown rankings"):
        paired_block_intervals(y, scores, np.arange(len(y)), times, {},
                               hours=24, resamples=10, min_blocks=8, rate=.05, seed=42)


def test_hybrid_refills_overlaps_without_exceeding_budget():
    ids = np.arange(6)
    first = np.array([6., 5., 4., 3., 2., 1.])
    second = np.array([6., 1., 2., 3., 4., 5.])
    combined = hybrid_ranking(first, second, ids, .5)
    assert np.argsort(-combined).tolist() == [0, 5, 1, 4, 2, 3]
    assert len(np.unique(combined)) == 6
    shuffle = np.array([5, 4, 3, 2, 1, 0])
    shuffled = hybrid_ranking(first[shuffle], second[shuffle], ids[shuffle], .5)
    assert ids[shuffle][np.argsort(-shuffled)].tolist() == np.argsort(-combined).tolist()
