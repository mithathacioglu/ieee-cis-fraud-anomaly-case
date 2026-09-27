"""Kayıtlı kararları bir örneklemde bağımsız olarak yeniden üretir."""

import json
import operator
from pathlib import Path

import pandas as pd

from fraud_case.evaluate_rules import load_inputs


def condition_matches(node, row):
    if "all" in node:
        return all(condition_matches(child, row) for child in node["all"])
    if "any" in node:
        return any(condition_matches(child, row) for child in node["any"])
    value = row[node["field"]]
    missing = bool(pd.isna(value))
    op = node["op"]
    if op == "is_missing":
        return missing
    if op == "is_present":
        return not missing
    if missing:
        return False
    target = node["value"]
    if op == "in":
        return value in target
    return {"eq": operator.eq, "ne": operator.ne, "gt": operator.gt,
            "gte": operator.ge, "lt": operator.lt, "lte": operator.le}[op](value, target)


def main():
    root = Path(__file__).resolve().parents[1]
    frame = load_inputs(root)
    result = pd.read_parquet(root / "data/processed/rule_decisions.parquet")
    pd.testing.assert_frame_equal(frame[["TransactionID", "split"]], result[["TransactionID", "split"]])
    spec = json.loads((root / "artifacts/rules/rules_snapshot.json").read_text())
    rules = sorted(spec["rules"], key=lambda r: (-r["priority"], r["id"]))
    indices = set(result.sample(min(128, len(result)), random_state=42).index)
    # Her aktif kuralın gerçek bir tetiklenmesini ve varsa bir çatışmayı dahil et
    for name in [f"rule_match_{r['id']}" for r in rules] + ["rule_action_conflict"]:
        matched = result.index[result[name]]
        if len(matched):
            indices.add(matched[0])
    for index in sorted(indices):
        observed, saved = frame.loc[index], result.loc[index]
        matches = [r for r in rules if r["enabled"] and condition_matches(r["when"], observed)]
        matched_ids = {r["id"] for r in matches}
        assert all(bool(saved[f"rule_match_{r['id']}"]) == (r["id"] in matched_ids) for r in rules)
        assert saved.matched_rule_count == len(matches)
        winner = matches[0] if matches else None
        assert saved.rule_action == (winner["then"]["action"] if winner else "no_rule_match")
        assert (saved.winning_rule == winner["id"]) if winner else pd.isna(saved.winning_rule)
        assert (saved.winning_priority == winner["priority"]) if winner else pd.isna(saved.winning_priority)
        assert bool(saved.rule_action_conflict) == (len({r["then"]["action"] for r in matches}) > 1)
    print(f"Verified ID/split alignment for {len(result):,} saved decisions.")
    print(f"Independent scalar replay matched {len(indices)} rows, including rule triggers and conflict; no labels read.")


if __name__ == "__main__":
    main()
