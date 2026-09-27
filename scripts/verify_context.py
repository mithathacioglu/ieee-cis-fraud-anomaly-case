"""Kayıtlı context invariant'larını kontrol eder, bir örneklemi yeniden hesaplar."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_case.anomaly import LAYERS
from fraud_case.context import ContextConfig, ContextEngine


def main():
    root = Path(__file__).resolve().parents[1]
    scores = pd.read_parquet(root / "data/processed/scores.parquet")
    adjusted = pd.read_parquet(root / "data/processed/context_scores.parquet")
    pd.testing.assert_frame_equal(adjusted[scores.columns], scores)
    raw = adjusted.raw_anomaly_score
    reduction = adjusted.context_reduction
    final = adjusted.adjusted_anomaly_score
    config = ContextConfig(**json.loads((root / "artifacts/context/selected_config.json").read_text()))
    assert final.between(0, 1).all() and final.le(raw + 1e-12).all()
    assert reduction.ge(0).all()
    assert reduction.le(config.max_absolute_reduction + 1e-12).all()
    assert reduction.le(raw * config.max_relative_reduction + 1e-12).all()
    contributions = adjusted[[f"context_{layer}_contribution" for layer in LAYERS]]
    assert contributions.ge(-1e-12).all().all()
    np.testing.assert_allclose(contributions.sum(axis=1), final, atol=1e-12)
    np.testing.assert_allclose(raw - reduction, final, atol=1e-12)
    assert reduction.loc[adjusted.context_guard_active].eq(0).all()

    sample = scores.sample(n=min(128, len(scores)), random_state=42).index
    features = pd.read_parquet(root / "data/processed/features.parquet").loc[sample].copy()
    source = pd.read_parquet(root / "data/processed/transactions.parquet",
                             columns=["TransactionID", "TransactionDT", "ProductCD"]).loc[sample]
    pd.testing.assert_series_equal(features.TransactionID, source.TransactionID)
    features["TransactionDT"] = source.TransactionDT
    features["ProductCD"] = source.ProductCD
    reference = json.loads((root / "artifacts/context/product_reference.json").read_text())
    replay = ContextEngine(config, reference).apply(features, scores.loc[sample])
    pd.testing.assert_frame_equal(replay, adjusted.loc[sample])
    print(f"Verified {len(scores):,} rows: raw preservation, caps, contributions and guards.")
    print(f"Replayed {len(sample)} rows from persisted configuration and product references; no labels read.")


if __name__ == "__main__":
    main()
