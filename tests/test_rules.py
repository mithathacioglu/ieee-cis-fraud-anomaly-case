import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_case.rules import RuleEngine


def leaf(field="amount", op="gte", value=100):
    return {"field": field, "op": op, "value": value}


def rule(rule_id="R1", priority=10, action="review", condition=None, enabled=True):
    return {"id": rule_id, "enabled": enabled, "priority": priority,
            "when": condition or leaf(), "then": {"action": action}, "description": "Test policy"}


def config(*rules):
    return {"version": 1, "default_action": "no_rule_match", "rules": list(rules) or [rule()]}


def frame(values=(99., 100., np.nan)):
    return pd.DataFrame({"TransactionID": range(1, len(values) + 1), "amount": values})


def test_priority_conflict_and_trace():
    engine = RuleEngine(config(rule("Monitor", 5, "monitor"), rule("Review", 20)))
    data = frame([100.])
    before = data.copy(deep=True)
    result = engine.evaluate(data).iloc[0]
    assert result.winning_rule == "Review" and result.rule_action == "review"
    assert result.rule_action_conflict and result.matched_rule_count == 2
    explanation = engine.explain(data)
    assert [r["status"] for r in explanation["rules"]] == ["selected", "superseded"]
    assert explanation["rules"][0]["condition"]["observed"] == 100.
    json.dumps(explanation, allow_nan=False)
    pd.testing.assert_frame_equal(data, before)


def test_tie_resolution_independent_of_file_order():
    a, b = rule("Alpha", action="monitor"), rule("Beta")
    first = RuleEngine(config(b, a)).evaluate(frame())
    second = RuleEngine(config(a, b)).evaluate(frame())
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[1, "winning_rule"] == "Alpha"


@pytest.mark.parametrize("op,value,expected", [
    ("eq", 100, [False, True, False]), ("ne", 100, [True, False, False]),
    ("gt", 100, [False, False, False]), ("gte", 100, [False, True, False]),
    ("lt", 100, [True, False, False]), ("lte", 100, [True, True, False]),
    ("in", [100, 101], [False, True, False]),
])
def test_numeric_operators_and_missing(op, value, expected):
    result = RuleEngine(config(rule(condition=leaf(op=op, value=value)))).evaluate(frame())
    assert result.rule_match_R1.tolist() == expected


@pytest.mark.parametrize("op,expected", [("is_missing", [False, False, True]), ("is_present", [True, True, False])])
def test_explicit_missing_operators(op, expected):
    engine = RuleEngine(config(rule(condition={"field": "amount", "op": op})))
    assert engine.evaluate(frame()).rule_match_R1.tolist() == expected
    if op == "is_missing":
        detail = engine.explain(frame().iloc[[2]])["rules"][0]["condition"]
        assert detail["observed"] is None and detail["missing"]
        json.dumps(detail, allow_nan=False)


def test_nested_all_any_and_nullable_inputs():
    condition = {"all": [{"any": [leaf(op="lt"), leaf(op="eq")]}, leaf(op="gt", value=98)]}
    data = frame()
    data["amount"] = data.amount.astype("Float64")
    assert RuleEngine(config(rule(condition=condition))).evaluate(data).rule_match_R1.tolist() == [True, True, False]


def test_disabled_rule_needs_no_observation_and_is_explained():
    engine = RuleEngine(config(rule(enabled=False)))
    data = frame().drop(columns="amount")
    result = engine.evaluate(data)
    assert result.rule_action.eq("no_rule_match").all() and result.winning_rule.isna().all()
    assert engine.explain(data.iloc[[0]])["rules"][0]["status"] == "disabled"


@pytest.mark.parametrize("condition", [
    leaf("isFraud"), leaf("TransactionID"), leaf(op="eval"), leaf(value="__import__('os')"),
    leaf(value=float("inf")), leaf(value=None), leaf(field=[]), leaf(op=[]), {"all": []}, {"any": [], "all": []},
    {"field": "amount", "op": "is_missing", "value": 1}, {"all": [leaf()], "extra": 1},
])
def test_invalid_conditions_rejected(condition):
    with pytest.raises(ValueError):
        RuleEngine(config(rule(condition=condition)))


def test_depth_and_node_limits():
    node = leaf()
    for _ in range(8):
        node = {"all": [node]}
    with pytest.raises(ValueError, match="depth"):
        RuleEngine(config(rule(condition=node)))
    with pytest.raises(ValueError, match="100"):
        RuleEngine(config(rule(condition={"all": [{"any": [leaf()] * 20}] * 6})))


@pytest.mark.parametrize("mutation", [
    lambda c: c["rules"].append(deepcopy(c["rules"][0])),
    lambda c: c["rules"][0].update(priority=True),
    lambda c: c["rules"][0].update(enabled="true"),
    lambda c: c["rules"][0].update(then={"action": "approve"}),
    lambda c: c.update(version=2), lambda c: c.update(default_action="monitor"),
    lambda c: c["rules"][0].update(extra="typo"),
])
def test_invalid_configuration_rejected(mutation):
    spec = config()
    mutation(spec)
    with pytest.raises(ValueError):
        RuleEngine(spec)


def test_missing_column_is_an_error_not_an_unmatched_rule():
    with pytest.raises(ValueError, match="Missing"):
        RuleEngine(config()).evaluate(frame().drop(columns="amount"))


@pytest.mark.parametrize("change", [
    lambda d: d.assign(amount="100"), lambda d: d.assign(amount=np.inf),
    lambda d: d.assign(TransactionID=1), lambda d: d.set_axis([0, 0, 1]),
])
def test_bad_inputs_rejected(change):
    with pytest.raises(ValueError):
        RuleEngine(config()).evaluate(change(frame()))


def test_config_snapshot_and_reload(tmp_path):
    spec = config()
    engine = RuleEngine(spec)
    spec["rules"][0]["when"]["value"] = 999
    assert engine.evaluate(frame()).rule_match_R1.sum() == 1
    assert RuleEngine(spec).evaluate(frame()).rule_match_R1.sum() == 0
    exported = engine.config
    exported["rules"].clear()
    assert len(engine.config["rules"]) == 1
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(engine.config))
    assert RuleEngine.from_json(path).config_sha256 == engine.config_sha256
    path.write_text('{"version":1,"version":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        RuleEngine.from_json(path)


def test_shipped_rules_have_live_paths_and_risk_wins():
    engine = RuleEngine.from_json(Path(__file__).resolve().parents[1] / "config/rules.json")
    assert len(engine.config["rules"]) == 10
    # Alışıldık tutar, şüpheli işlem sıklığını geçersiz kılmıyor
    data = pd.DataFrame([{field: 0. for field in engine.required_fields}])
    data["TransactionID"] = 1
    data["prior_count_1h"] = 10
    data["frequent_entity"] = data["usual_amount"] = 1
    result = engine.evaluate(data).iloc[0]
    assert result.rule_match_R10_familiar_activity
    assert result.winning_rule == "R02_hourly_velocity" and result.rule_action_conflict
    # Her kuralı ayrı ayrı tetikle; erişilemeyen kural kalmasın
    for spec in engine.config["rules"]:
        sample = pd.DataFrame([{field: np.nan for field in engine.required_fields}])
        sample["TransactionID"] = 2
        leaves = spec["when"].get("all", [spec["when"]])
        for condition in leaves:
            value = condition["value"]
            sample[condition["field"]] = value - .00001 if condition["op"] == "lt" else value
        assert engine.evaluate(sample).iloc[0][f"rule_match_{spec['id']}"]


def test_batch_matches_single_rows_and_ignores_label():
    engine = RuleEngine(config())
    data = frame().assign(isFraud=[0, 1, 1])
    batch = engine.evaluate(data)
    single = pd.concat([engine.evaluate(data.iloc[[i]]) for i in range(len(data))])
    pd.testing.assert_frame_equal(batch, single)
    pd.testing.assert_frame_equal(batch, engine.evaluate(data.assign(isFraud=[1, 0, 0])))


def test_empty_input_and_explain_cardinality():
    engine = RuleEngine(config())
    assert engine.evaluate(frame().iloc[:0]).empty
    with pytest.raises(ValueError, match="one transaction"):
        engine.explain(frame())


def test_out_of_range_normalized_score_rejected():
    engine = RuleEngine(config(rule(condition=leaf("adjusted_anomaly_score", value=.8))))
    with pytest.raises(ValueError, match="outside"):
        engine.evaluate(frame().assign(adjusted_anomaly_score=1.1))


def test_decision_metrics_have_correct_denominators():
    from fraud_case.evaluate_rules import decision_metrics

    result = decision_metrics([1, 0, 0, 0], np.array([True, True, False, False]))
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 0, 2)
    assert result["recall"] == 1 and result["precision"] == .5
    assert result["false_positive_rate"] == pytest.approx(1 / 3)
    empty = decision_metrics([0, 0], np.array([False, False]))
    assert empty["precision"] is None and empty["recall"] is None
    with pytest.raises(ValueError):
        decision_metrics([1, 0], [1, 0])
