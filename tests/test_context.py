from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fraud_case.anomaly import LAYERS
from fraud_case.context import ContextConfig, ContextEngine
from fraud_case.evaluation import metrics, paired_effect


def inputs():
    features = pd.DataFrame({
        "TransactionID": [1], "TransactionDT": [1000000], "prior_transaction_count": [30],
        "amount_to_prior_mean": [1.0], "prior_count_1h": [2], "frequent_entity": [1],
        "usual_amount": [1], "entity_hour_probability": [0.1], "entity_product_is_new": [0.0],
        "entity_email_is_new": [0.0], "entity_device_is_new": [0.0],
        "calendar_available": [1], "is_business_hours": [1], "is_weekend": [0],
    })
    scores = pd.DataFrame({"TransactionID": [1], "raw_anomaly_score": [0.8],
                           "column_normalized": [0.95], "multivariate_normalized": [0.8], "entity_raw": [0.2],
                           **{f"{layer}_contribution": [0.2] for layer in LAYERS}})
    return features, scores


def test_familiar_context_changes_relevant_contributions_only():
    f, s = inputs()
    result = ContextEngine().apply(f, s)
    assert result.context_entity_reduction.item() == 0
    assert result.context_column_reduction.item() > 0
    assert result.context_temporal_reduction.item() > 0
    assert result.adjusted_anomaly_score.item() < s.raw_anomaly_score.item()
    assert result.context_reduction.item() == pytest.approx(0.05)
    total = result[[f"context_{layer}_contribution" for layer in LAYERS]].sum(axis=1).item()
    assert total == pytest.approx(result.adjusted_anomaly_score.item())
    assert s.raw_anomaly_score.item() == 0.8


@pytest.mark.parametrize("field,value,table", [
    ("entity_raw", 3, "scores"), ("multivariate_normalized", .99, "scores"),
    ("prior_count_1h", 10, "features"), ("entity_email_is_new", 1, "features"),
])
def test_strong_signals_block_discounts(field, value, table):
    f, s = inputs()
    (s if table == "scores" else f)[field] = value
    result = ContextEngine().apply(f, s)
    assert result.context_guard_active.item()
    assert result.context_reduction.item() == 0


def test_business_hours_are_supported_without_other_rules():
    f, s = inputs()
    result = ContextEngine().apply(f, s, enabled_rules={"business_hours"})
    assert result.context_temporal_reduction.item() == pytest.approx(.02)
    assert result.context_temporal_reason.item() == "business_hours"
    f["calendar_available"] = 0
    assert ContextEngine().apply(f, s, enabled_rules={"business_hours"}).context_reduction.item() == 0


def test_weekend_requires_prior_external_schedule():
    f, s = inputs()
    f["is_weekend"], f["is_business_hours"] = 1, 0
    engine = ContextEngine()
    assert engine.apply(f, s, enabled_rules={"expected_weekend"}).context_reduction.item() == 0
    external = pd.DataFrame({"TransactionID": [1], "weekend_activity_expected": [True],
                             "schedule_observed_at": [900000], "schedule_source": ["synthetic-test-schedule"]})
    assert engine.apply(f, s, external, {"expected_weekend"}).context_temporal_reduction.item() == pytest.approx(.02)


def test_trust_needs_evidence_available_before_transaction():
    f, s = inputs()
    engine = ContextEngine()
    external = pd.DataFrame({"TransactionID": [1], "trusted_entity": [True],
                             "trust_observed_at": [999999], "trust_source": ["synthetic-test-review"]})
    assert engine.apply(f, s, external, {"verified_trust"}).context_multivariate_reduction.item() == pytest.approx(.02)
    for late_time in (1000000, 1000001):
        external["trust_observed_at"] = late_time
        assert engine.apply(f, s, external, {"verified_trust"}).context_reduction.item() == 0


def test_incomplete_and_duplicate_external_evidence_is_rejected():
    f, s = inputs()
    external = pd.DataFrame({"TransactionID": [1], "trusted_entity": [True]})
    with pytest.raises(ValueError, match="source and observation"):
        ContextEngine().apply(f, s, external)
    with pytest.raises(ValueError, match="unique"):
        ContextEngine().apply(f, s, pd.concat([external, external]))


def test_same_layer_discounts_do_not_stack():
    f, s = inputs()
    config = replace(ContextConfig(), max_absolute_reduction=1, max_relative_reduction=1)
    result = ContextEngine(config).apply(f, s, enabled_rules={"frequent_familiar_activity", "business_hours"})
    assert result.context_temporal_reduction.item() == pytest.approx(.04)
    assert result.context_temporal_reason.item() == "frequent_familiar_activity"


def test_relative_cap_and_zero_strength():
    f, s = inputs()
    config = replace(ContextConfig(), max_relative_reduction=.01)
    assert ContextEngine(config).apply(f, s).context_reduction.item() == pytest.approx(.008)
    assert ContextEngine(replace(config, strength=0)).apply(f, s).context_reduction.item() == 0


def test_missing_history_does_not_create_trust_or_a_discount():
    f, s = inputs()
    f["prior_transaction_count"] = np.nan
    assert ContextEngine().apply(f, s).context_reduction.item() == 0


def test_misaligned_or_inconsistent_inputs_fail():
    f, s = inputs()
    s["TransactionID"] = 2
    with pytest.raises(ValueError, match="identical"):
        ContextEngine().apply(f, s)
    s["TransactionID"], s["raw_anomaly_score"] = 1, .5
    with pytest.raises(ValueError, match="sum"):
        ContextEngine().apply(f, s)


def test_explanation_uses_actual_discounts():
    f, s = inputs()
    explanation = ContextEngine().explain(f, s)
    assert sum(item["reduction"] for item in explanation["applied"]) == pytest.approx(explanation["reduction"])
    assert explanation["raw_score"] - explanation["adjusted_score"] == pytest.approx(explanation["reduction"])
    assert not explanation["trust_available"]


def test_metrics_keep_false_positives_and_missed_fraud_separate():
    labels = [0, 0, 1, 1]
    raw = [.9, .6, .8, .1]
    adjusted = [.7, .6, .7, .1]
    before = metrics(labels, raw, threshold=.75)
    after = metrics(labels, adjusted, threshold=.75)
    assert (before["tp"], before["fp"], before["fn"], before["tn"]) == (1, 1, 1, 1)
    assert before["false_positive_rate"] == .5
    assert after["precision"] is None
    assert paired_effect(labels, raw, adjusted, .75) == {"removed_false_positives": 1, "added_false_positives": 0, "lost_true_positives": 1, "added_true_positives": 0}


def test_fixed_review_budget_resolves_ties_by_id():
    result = metrics([0, 1, 0], [.5, .5, .2], review_count=1, transaction_ids=[2, 1, 3])
    assert result["alerts"] == 1
    assert result["tp"] == 1
    with pytest.raises(ValueError, match="exactly one"):
        metrics([0, 1], [.1, .9])


def test_invalid_context_configuration_is_rejected():
    with pytest.raises(ValueError, match=r"in \[0,1\]"):
        ContextConfig(strength=2)
    with pytest.raises(ValueError, match="contain one"):
        ContextConfig(stable_ratio_min=2)


def test_product_reference_uses_training_only_and_handles_unseen_codes():
    train = pd.DataFrame({"ProductCD": ["C"] * 10, "amount": list(range(10)), "split": "train"})
    engine = ContextEngine(replace(ContextConfig(), product_minimum_support=5)).fit_product_context(train)
    assert engine.product_reference["C"]["lower"] == pytest.approx(.9)
    assert engine.product_reference["C"]["upper"] == pytest.approx(8.1)
    f, s = inputs()
    f["ProductCD"], f["amount"] = "C", 5
    assert engine.apply(f, s, enabled_rules={"product_typical_amount"}).context_column_reduction.item() == pytest.approx(.02)
    f["ProductCD"] = "unseen"
    assert engine.apply(f, s, enabled_rules={"product_typical_amount"}).context_reduction.item() == 0
    train["split"] = "validation"
    with pytest.raises(ValueError, match="training partition"):
        engine.fit_product_context(train)


def test_context_selection_rejects_excess_recall_loss():
    from fraud_case.evaluate_context import select_strength
    strength, _ = select_strength([0, 1, 1], [.9, .9, .9], {0: [.9, .9, .9], .5: [.7, .9, .9], 1: [.7, .7, .9]}, .8, .01)
    assert strength == .5


def test_validation_boundary_preserves_equal_timestamps():
    from fraud_case.evaluate_context import validation_sections
    frame = pd.DataFrame({"TransactionDT": [1, 2, 2, 2, 3, 4]})
    cal, audit, boundary = validation_sections(frame, .5)
    assert boundary == 2
    assert cal.tolist() == [True, False, False, False, False, False]
    assert frame.loc[cal, "TransactionDT"].max() < frame.loc[audit, "TransactionDT"].min()
