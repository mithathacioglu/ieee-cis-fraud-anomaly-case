"""Recompute report metrics from saved per-transaction predictions."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def verify(folder):
    summary = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    config = summary["manifest"]["protocol"]
    checks, files, coverage, totals = 0, {}, {}, {}

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    for f in summary["folds"]:
        delay, fold = f["label_delay_days"], f["fold"]
        path = folder / f"delay_{delay}_fold_{fold}_predictions.parquet"
        data = pd.read_parquet(path)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        ids, labels = data.TransactionID.to_numpy(), data.isFraud.to_numpy()
        check(len(data) == f["rows"] and int(labels.sum()) == f["positives"], "Row/label counts")
        check(not data.TransactionID.duplicated().any(), "Duplicate predictions")
        check(data.TransactionDT.is_monotonic_increasing, "Unsorted predictions")
        check(f["train_end"] < f["evaluation_start"] - delay * 86400, "Training cutoff")
        check(data.TransactionDT.min() == f["evaluation_start"], "Window start")
        check(data.TransactionDT.max() == f["evaluation_end"], "Window end")
        coverage.setdefault(delay, []).extend(ids.tolist())
        totals.setdefault(delay, {"reviews": 0, "tp": dict.fromkeys(f["models"], 0)})
        for name, metrics in f["models"].items():
            values = data[name].to_numpy()
            check(np.isfinite(values).all(), f"Nonfinite {name}")
            check(math.isclose(average_precision_score(labels, values), metrics["average_precision"], abs_tol=1e-12), "AP")
            check(math.isclose(roc_auc_score(labels, values), metrics["roc_auc"], abs_tol=1e-12), "AUC")
            # Separate implementation from the experiment's np.lexsort/top_k_hits.
            order = sorted(range(len(data)), key=lambda i: (-float(values[i]), int(ids[i])))
            for rate in config["review_rates"]:
                k = math.ceil(len(data) * rate)
                tp = sum(int(labels[i]) for i in order[:k])
                row = metrics["budgets"][str(rate)]
                expected = {"reviews": k, "tp": tp, "fp": k - tp, "fn": int(labels.sum()) - tp,
                            "precision": tp / k, "recall": tp / int(labels.sum())}
                for key, value in expected.items():
                    check(math.isclose(row[key], value, abs_tol=1e-12), f"{delay}/{fold}/{name}/{rate}/{key}")
            totals[delay]["tp"][name] += metrics["budgets"][str(config["primary_review_rate"])]["tp"]
        totals[delay]["reviews"] += f["models"]["raw"]["budgets"][str(config["primary_review_rate"])]["reviews"]
        for interval in f["block_bootstrap"]:
            time_blocks = np.floor((data.TransactionDT.to_numpy() - data.TransactionDT.iloc[0]) / (interval["hours"] * 3600))
            check(len(np.unique(time_blocks)) == interval["blocks"], "Block count")
            for gap in interval["comparisons"].values():
                check(np.isfinite([gap["ci_low"], gap["ci_high"]]).all() and gap["ci_low"] <= gap["ci_high"], "Interval bounds")
        k = math.ceil(len(data) * config["primary_review_rate"])
        lists = [set(sorted(range(len(data)), key=lambda i: (-float(data[name].iloc[i]), int(ids[i])))[:k])
                 for name in ("raw", "matched_gradient_boosting")]
        raw, gb = lists
        expected = {"reviews_each": k, "shared_reviews": len(raw & gb), "raw_only_tp": int(labels[list(raw - gb)].sum()),
                    "supervised_only_tp": int(labels[list(gb - raw)].sum()), "union_reviews": len(raw | gb),
                    "union_tp": int(labels[list(raw | gb)].sum())}
        check(f["overlap_raw_matched_gb"] == expected, "Overlap accounting")
    reference = coverage[config["label_delay_days"][0]]
    for rows in coverage.values():
        check(len(rows) == len(set(rows)), "Overlapping evaluation windows")
        check(rows == reference, "Delay scenarios must cover identical transactions")
    snapshot = folder / "source_snapshot"
    if snapshot.exists():
        for name, digest in summary["manifest"]["source_sha256"].items():
            check(hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == digest, f"Snapshot hash {name}")
    return {"status": "passed", "checks": checks, "fold_evaluations": len(summary["folds"]),
            "unique_evaluation_transactions": len(reference), "totals_at_primary_budget": totals,
            "prediction_sha256": files,
            "scope": "Saved prediction replay: counts, budgets, AP/AUC, confusion counts, overlap, windows, source snapshots. "
                     "Bootstrap distributions and model fitting are not rerun by this verifier."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/temporal_validation")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/temporal_validation_checks.json")
    args = parser.parse_args()
    result = verify(args.input)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
