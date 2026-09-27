"""JSON kural yorumlayıcısı. Koşullar doğrulanmış literal, hiçbir şey eval'lenmiyor."""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_case.features import FEATURE_SPECS

FIELDS = set(FEATURE_SPECS) | {
    "raw_anomaly_score", "adjusted_anomaly_score", "available_weight",
    "column_normalized", "multivariate_normalized", "entity_normalized", "temporal_normalized",
    "column_raw", "multivariate_raw", "entity_raw", "temporal_raw", "context_reduction",
    "context_guard_active", "context_calendar_available", "context_trust_available",
    "context_weekend_schedule_available",
}
OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "is_missing", "is_present"}


def _finite_number(value):
    return type(value) in (int, float) and np.isfinite(value)


def _validate_condition(node, depth=0):
    if not isinstance(node, dict) or depth > 6:
        raise ValueError("Condition must be an object with depth <= 6")
    groups = set(node) & {"all", "any"}
    if groups:
        if len(groups) != 1 or len(node) != 1:
            raise ValueError("Condition group must contain only all or any")
        children = node[next(iter(groups))]
        if not isinstance(children, list) or not 1 <= len(children) <= 20:
            raise ValueError("Condition group requires 1..20 children")
        fields, size = set(), 1
        for child in children:
            child_fields, child_size = _validate_condition(child, depth + 1)
            fields |= child_fields
            size += child_size
        if size > 100:
            raise ValueError("At most 100 condition nodes per rule")
        return fields, size
    if not isinstance(node.get("field"), str) or not isinstance(node.get("op"), str) or node["field"] not in FIELDS or node["op"] not in OPS:
        raise ValueError("Unknown rule field or operator")
    op = node["op"]
    keys = {"field", "op"} if op in {"is_missing", "is_present"} else {"field", "op", "value"}
    if set(node) != keys:
        raise ValueError("Unexpected or missing condition keys")
    if "value" in node:
        value = node["value"]
        if op == "in":
            valid = isinstance(value, list) and 1 <= len(value) <= 100 and all(_finite_number(v) or type(v) is bool for v in value)
        elif op in {"eq", "ne"}:
            valid = _finite_number(value) or type(value) is bool
        else:
            valid = _finite_number(value)
        if not valid:
            raise ValueError("Condition values must be finite numeric/boolean literals")
    return {node["field"]}, 1


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class RuleEngine:
    """Highest numeric priority wins; ties use ascending rule ID, independent of file order."""

    def __init__(self, config: dict):
        if not isinstance(config, dict) or set(config) != {"version", "default_action", "rules"}:
            raise ValueError("Expected version, default_action and rules")
        if type(config["version"]) is not int or config["version"] != 1 or config["default_action"] != "no_rule_match":
            raise ValueError("Unsupported version or default action")
        rules = config["rules"]
        if not isinstance(rules, list) or not 1 <= len(rules) <= 100:
            raise ValueError("Expected 1..100 rules")
        seen, fields = set(), set()
        for rule in rules:
            if not isinstance(rule, dict) or set(rule) != {"id", "enabled", "priority", "when", "then", "description"}:
                raise ValueError("Invalid rule keys")
            rule_id = rule["id"]
            if not isinstance(rule_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", rule_id) or rule_id in seen:
                raise ValueError("Rule IDs must be unique simple identifiers")
            seen.add(rule_id)
            if type(rule["enabled"]) is not bool or type(rule["priority"]) is not int or not 0 <= rule["priority"] <= 1000:
                raise ValueError("Invalid enabled flag or priority")
            if not isinstance(rule["description"], str) or not rule["description"].strip() or len(rule["description"]) > 1000:
                raise ValueError("Rule requires a nonempty description of at most 1000 characters")
            action = rule["then"]
            if not isinstance(action, dict) or set(action) != {"action"} or not isinstance(action["action"], str) or action["action"] not in {"review", "monitor"}:
                raise ValueError("Rule action must be review or monitor")
            used, _ = _validate_condition(rule["when"])
            if rule["enabled"]:
                fields |= used
        self._config = deepcopy(config)
        self._rules = sorted(deepcopy(rules), key=lambda r: (-r["priority"], r["id"]))
        self.required_fields = frozenset(fields)
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.config_sha256 = hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_json(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys))

    @property
    def config(self):
        return deepcopy(self._config)

    def _validate_frame(self, frame):
        if not frame.columns.is_unique or not frame.index.is_unique:
            raise ValueError("Input columns and index must be unique")
        missing = ({"TransactionID"} | self.required_fields) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing rule input columns: {sorted(missing)}")
        ids = frame.TransactionID
        if ids.isna().any() or ids.duplicated().any() or not pd.api.types.is_integer_dtype(ids):
            raise ValueError("Transaction IDs must be unique non-null integers")
        for name in self.required_fields:
            values = frame[name]
            if not (pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)):
                raise ValueError(f"Rule input must be numeric/boolean: {name}")
            if not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
                raise ValueError(f"Infinite rule input: {name}")
        for name in self.required_fields & {"raw_anomaly_score", "adjusted_anomaly_score", "available_weight",
                                            "column_normalized", "multivariate_normalized", "entity_normalized", "temporal_normalized"}:
            if not frame[name].dropna().between(0, 1).all():
                raise ValueError(f"Normalized input outside [0,1]: {name}")

    @staticmethod
    def _condition(node, frame):
        if "all" in node or "any" in node:
            key = "all" if "all" in node else "any"
            result = pd.Series(key == "all", index=frame.index)
            for child in node[key]:
                matched = RuleEngine._condition(child, frame)
                result = result & matched if key == "all" else result | matched
            return result
        values, op = frame[node["field"]], node["op"]
        if op == "is_missing":
            return values.isna()
        if op == "is_present":
            return values.notna()
        value = node["value"]
        if op == "in":
            matched = values.isin(value)
        else:
            method = {"eq": "eq", "ne": "ne", "gt": "gt", "gte": "ge", "lt": "lt", "lte": "le"}[op]
            matched = getattr(values, method)(value)
        # Özellikle NaN != eşik ifadesi eşleşme sayılmasın
        return matched.fillna(False) & values.notna()

    def evaluate(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._validate_frame(frame)
        output = frame[["TransactionID"]].copy()
        output["rule_action"] = "no_rule_match"
        output["winning_rule"] = pd.Series(None, index=frame.index, dtype="string")
        output["winning_priority"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        output["matched_rule_count"] = 0
        proposed = {action: pd.Series(False, index=frame.index) for action in ("review", "monitor")}
        for rule in self._rules:
            matched = self._condition(rule["when"], frame) if rule["enabled"] else pd.Series(False, index=frame.index)
            output[f"rule_match_{rule['id']}"] = matched
            output["matched_rule_count"] += matched.astype(int)
            action = rule["then"]["action"]
            proposed[action] |= matched
            wins = matched & output.winning_rule.isna()
            output.loc[wins, "winning_rule"] = rule["id"]
            output.loc[wins, "winning_priority"] = rule["priority"]
            output.loc[wins, "rule_action"] = action
        output["rule_action_conflict"] = proposed["review"] & proposed["monitor"]
        return output

    @classmethod
    def _trace(cls, node, frame):
        trace = deepcopy(node)
        trace["matched"] = bool(cls._condition(node, frame).iloc[0])
        if "all" in node or "any" in node:
            key = "all" if "all" in node else "any"
            trace[key] = [cls._trace(child, frame) for child in node[key]]
        else:
            value = frame[node["field"]].iloc[0]
            trace["observed"] = None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
            trace["missing"] = bool(pd.isna(value))
        return trace

    def explain(self, frame: pd.DataFrame) -> dict:
        if len(frame) != 1:
            raise ValueError("Explain expects one transaction")
        row = self.evaluate(frame).iloc[0]
        winner = None if pd.isna(row.winning_rule) else str(row.winning_rule)
        rules = []
        for rule in self._rules:
            matched = bool(row[f"rule_match_{rule['id']}"])
            status = "disabled" if not rule["enabled"] else "selected" if rule["id"] == winner else "superseded" if matched else "not_matched"
            rules.append({"id": rule["id"], "priority": rule["priority"], "action": rule["then"]["action"],
                          "description": rule["description"], "status": status, "matched": matched,
                          "condition": self._trace(rule["when"], frame) if rule["enabled"] else None})
        return {"transaction_id": int(row.TransactionID), "action": row.rule_action, "winning_rule": winner,
                "action_conflict": bool(row.rule_action_conflict), "config_sha256": self.config_sha256,
                "resolution": "Highest numeric priority wins; ties use ascending rule ID. Scores are unchanged.",
                "rules": rules}
