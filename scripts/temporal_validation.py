"""Expanding-window development evaluation; reads train/validation only.

Protocol is saved before fitting. No hyperparameter or block-width selection
uses the resulting metrics. See docs/validation_protocol.md for assumptions.
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from supervised_baseline import BUDGETS, SEED, TIME_PROXIES, evaluate, fit_rank
from threadpoolctl import threadpool_limits

from fraud_case.aggregation import ScoreAggregator
from fraud_case.anomaly import REQUIRED_FEATURES, AnomalyEngine
from fraud_case.evaluation import paired_block_intervals
from fraud_case.features import FEATURE_SPECS

ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = {
    "matched_gb_minus_raw": ("raw", "matched_gradient_boosting"),
    "full_gb_minus_raw": ("raw", "full_gradient_boosting"),
    "without_time_gb_minus_full": ("full_gradient_boosting", "without_time_gradient_boosting"),
    "matched_lr_minus_raw": ("raw", "matched_logistic_regression"),
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_protocol(config):
    if config["folds"] < 2 or config["bootstrap_resamples"] < 100 or config["min_blocks"] < 2:
        raise ValueError("Use >=2 folds, >=100 resamples and >=2 blocks")
    if config["review_rates"] != BUDGETS or config["seed"] != SEED:
        raise ValueError("Review rates and seed must match supervised_baseline")
    if config["primary_review_rate"] not in BUDGETS:
        raise ValueError("Primary budget must be one of the reported rates")
    for key in ("label_delay_days", "block_hours"):
        values = config[key]
        if not values or len(set(values)) != len(values) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key}")
        if any(v < 0 or (key == "block_hours" and v == 0) for v in values):
            raise ValueError(f"Invalid {key}")
    if not 0 < config["hybrid_supervised_share"] < 1:
        raise ValueError("Hybrid share must be between zero and one")


def chronological_folds(source, folds, delay_days):
    """Equal-duration validation windows. Timestamp ties cannot cross a boundary."""
    times = source.TransactionDT.to_numpy(dtype=float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Transactions must be finite and chronological")
    if not source.split.isin(["train", "validation"]).all():
        raise ValueError("Only train and validation are accepted")
    train_times = times[source.split.eq("train")]
    val_times = times[source.split.eq("validation")]
    if not len(train_times) or not len(val_times) or train_times.max() >= val_times.min():
        raise ValueError("Validation must strictly follow training")
    if folds < 2 or not np.isfinite(delay_days) or delay_days < 0:
        raise ValueError("Invalid fold count or label delay")
    edges = np.linspace(val_times.min(), np.nextafter(val_times.max(), np.inf), folds + 1)
    result = []
    for number, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True), 1):
        train = np.flatnonzero(times < start - delay_days * 86400)
        audit = np.flatnonzero((times >= start) & (times < end))
        if not len(train) or not len(audit):
            raise ValueError("Empty training or evaluation window")
        result.append((number, train, audit))
    return result


def feature_sets(train):
    # Selection uses this fold's training rows only.
    full = [c for c in FEATURE_SPECS if train[c].nunique(dropna=True) > 1]
    return {"full": full, "without_time": [c for c in full if c not in TIME_PROXIES],
            "matched": [c for c in REQUIRED_FEATURES]}


def hybrid_ranking(supervised, raw, ids, share):
    """Fixed quota interleaving; duplicate reviews count once, then refill.

    Uses only current-window scores and IDs, never labels. For each prefix,
    ceil(prefix * share) slots prefer the supervised ranking, others prefer raw.
    """
    orders = [np.lexsort((ids, -supervised)), np.lexsort((ids, -raw))]
    cursor, picked, order = [0, 0], set(), []
    for slot in range(1, len(ids) + 1):
        which = 0 if np.ceil(slot * share) > np.ceil((slot - 1) * share) else 1
        while int(orders[which][cursor[which]]) in picked:
            cursor[which] += 1
        row = int(orders[which][cursor[which]])
        picked.add(row)
        order.append(row)
    scores = np.empty(len(ids), dtype=float)
    scores[order] = np.arange(len(ids), 0, -1, dtype=float)
    return scores


def overlap_at_budget(y, ids, baseline, challenger, rate):
    count = int(np.ceil(len(y) * rate))
    first = set(np.lexsort((ids, -baseline))[:count])
    second = set(np.lexsort((ids, -challenger))[:count])
    return {"reviews_each": count, "shared_reviews": len(first & second),
            "raw_only_tp": int(y[list(first - second)].sum()),
            "supervised_only_tp": int(y[list(second - first)].sum()),
            "union_reviews": len(first | second), "union_tp": int(y[list(first | second)].sum())}


def scoring_engine(scoring):
    """Ana skorlama akisiyla ayni yapilandirmadan kurar.

    Onceden burada AnomalyEngine() ve ScoreAggregator() varsayilanlariyla
    kuruluyordu. Degerler config/scoring.json ile ayni oldugu icin sonuclar
    dogruydu, ama config degistiginde ana akis ile bu deney sessizce farkli
    modeller olcmeye baslardi. Tek kaynak: config/scoring.json.
    """
    return (AnomalyEngine(min_history=scoring["minimum_entity_history"], **scoring["isolation_forest"]),
            ScoreAggregator(scoring["weights"]))


def run_fold(features, source, train_idx, audit_idx, config, scoring):
    train, audit = features.iloc[train_idx].copy(), features.iloc[audit_idx].copy()
    y_train = source.isFraud.iloc[train_idx].to_numpy()
    y = source.isFraud.iloc[audit_idx].to_numpy()
    if len(np.unique(y_train)) != 2 or len(np.unique(y)) != 2:
        raise ValueError("Each training/evaluation window must contain both classes")
    ids = source.TransactionID.iloc[audit_idx].to_numpy()
    times = source.TransactionDT.iloc[audit_idx].to_numpy(dtype=float)
    # The engine deliberately refuses validation-labelled partitions on fit.
    # Pass only feature columns: the fold's training membership was checked above.
    engine, aggregator = scoring_engine(scoring)
    engine = engine.fit(train)
    aggregator = aggregator.fit(engine.score(train))
    scores = {"raw": aggregator.transform(engine.score(audit)).raw_anomaly_score.to_numpy()}
    columns = feature_sets(train)
    for variant, names in columns.items():
        # No random internal holdout in this temporal protocol; fixed 100 boosting iterations.
        for model, values in fit_rank(names, train, y_train, audit, early_stopping=False).items():
            scores[f"{variant}_{model}"] = values
    scores["hybrid"] = hybrid_ranking(scores["matched_gradient_boosting"], scores["raw"], ids,
                                      config["hybrid_supervised_share"])
    metrics = {name: evaluate(y, values, ids) for name, values in scores.items()}
    for result in metrics.values():
        for budget in result["budgets"].values():
            budget.update(fp=budget["reviews"] - budget["tp"], fn=int(y.sum()) - budget["tp"],
                          precision=budget["tp"] / budget["reviews"], recall=budget["tp"] / int(y.sum()))
    intervals = [paired_block_intervals(y, scores, ids, times, COMPARISONS, hours=hours,
                 resamples=config["bootstrap_resamples"], min_blocks=config["min_blocks"],
                 rate=config["primary_review_rate"], seed=config["seed"]) for hours in config["block_hours"]]
    predictions = pd.DataFrame({"TransactionID": ids, "TransactionDT": times, "isFraud": y, **scores})
    return {"train_rows": len(train), "train_positives": int(y_train.sum()), "rows": len(audit),
            "positives": int(y.sum()), "prevalence": float(y.mean()),
            "train_start": float(source.TransactionDT.iloc[train_idx[0]]),
            "train_end": float(source.TransactionDT.iloc[train_idx[-1]]),
            "evaluation_start": float(times[0]), "evaluation_end": float(times[-1]),
            "feature_columns": columns, "models": metrics, "block_bootstrap": intervals,
            "overlap_raw_matched_gb": overlap_at_budget(y, ids, scores["raw"], scores["matched_gradient_boosting"],
                                                       config["primary_review_rate"])}, predictions


def load_development_data(data_root):
    filters = [("split", "in", ["train", "validation"])]
    source = pd.read_parquet(data_root / "transactions.parquet", filters=filters,
                             columns=["TransactionID", "TransactionDT", "isFraud", "split"])
    features = pd.read_parquet(data_root / "features.parquet", filters=filters,
                               columns=["TransactionID", *FEATURE_SPECS])
    source, features = source.reset_index(drop=True), features.reset_index(drop=True)
    if source.TransactionID.duplicated().any() or not source.TransactionID.equals(features.TransactionID):
        raise ValueError("Development rows are not uniquely aligned")
    if not source.isFraud.isin([0, 1]).all():
        raise ValueError("Invalid fraud labels")
    return features[list(FEATURE_SPECS)], source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/temporal_validation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/temporal_validation")
    parser.add_argument("--scoring", type=Path, default=ROOT / "config/scoring.json")
    args = parser.parse_args()
    config = json.loads(args.protocol.read_text(encoding="utf-8"))
    scoring = json.loads(args.scoring.read_text(encoding="utf-8"))
    validate_protocol(config)
    # Never silently overwrite a previous experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "protocol": config,
                "python": platform.python_version(), "sklearn": sklearn.__version__,
                "scope": "Development only: previously inspected train/validation; final test excluded.",
                "scoring": scoring,
                # Guven araliklarini hesaplayan evaluation.py ve skorlama yapilandirmasi da
                # bu listede olmali; yoksa deneyin izlenebilirligi eksik kalir.
                "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), ROOT / "scripts/supervised_baseline.py",
                                            ROOT / "src/fraud_case/anomaly.py", ROOT / "src/fraud_case/aggregation.py",
                                            ROOT / "src/fraud_case/evaluation.py", ROOT / "config/scoring.json"]}}
    dump(args.output / "protocol.json", manifest)
    features, source = load_development_data(args.data_root)
    manifest["filtered_input_sha256"] = {
        "features": hashlib.sha256(pd.util.hash_pandas_object(features, index=False).values.tobytes()).hexdigest(),
        "transactions": hashlib.sha256(pd.util.hash_pandas_object(source, index=False).values.tobytes()).hexdigest(),
    }
    dump(args.output / "protocol.json", manifest)
    results = []
    with threadpool_limits(limits=4):
        for delay in config["label_delay_days"]:
            for fold, train, audit in chronological_folds(source, config["folds"], delay):
                print(f"delay={delay} days, fold={fold}, train={len(train)}, evaluation={len(audit)}", flush=True)
                result, predictions = run_fold(features, source, train, audit, config, scoring)
                result.update(label_delay_days=delay, fold=fold)
                name = f"delay_{delay}_fold_{fold}"
                predictions.to_parquet(args.output / f"{name}_predictions.parquet", index=False)
                dump(args.output / f"{name}.json", result)
                results.append(result)
                print({key: value["budgets"]["0.05"]["tp"] for key, value in result["models"].items()}, flush=True)
    dump(args.output / "evaluation.json", {"manifest": manifest, "folds": results})
    print(f"Saved {len(results)} evaluations to {args.output}", flush=True)


if __name__ == "__main__":
    main()
