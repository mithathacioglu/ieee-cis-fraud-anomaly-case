"""Causal feature'ları gerçek veride bağımsız bir toplu hesapla denetler."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_case.data import entity_proxy
from fraud_case.features import FEATURE_SPECS


def main():
    root = Path(__file__).resolve().parents[1]
    columns = ["TransactionID", "TransactionDT", "TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "split"]
    source = pd.read_parquet(root / "data/processed/transactions.parquet", columns=columns)
    actual = pd.read_parquet(root / "data/processed/features.parquet")
    pd.testing.assert_frame_equal(actual[["TransactionID", "split"]], source[["TransactionID", "split"]])
    assert set(actual.columns) == {"TransactionID", "split", *FEATURE_SPECS}
    assert "isFraud" not in actual
    times = source.TransactionDT.to_numpy()
    np.testing.assert_array_equal(actual.prior_global_count, np.searchsorted(times, times, side="left"))
    source["proxy"] = entity_proxy(source)
    eligible = source.dropna(subset=["proxy"])
    groups = eligible.groupby(["proxy", "TransactionDT"]).agg(count=("TransactionID", "size"), total=("TransactionAmt", "sum"))
    cumulative = groups.groupby(level=0).cumsum() - groups
    expected = eligible[["proxy", "TransactionDT"]].join(cumulative, on=["proxy", "TransactionDT"])
    np.testing.assert_array_equal(actual.loc[eligible.index, "prior_transaction_count"], expected["count"])
    means = expected["total"] / expected["count"].replace(0, np.nan)
    np.testing.assert_allclose(actual.loc[eligible.index, "prior_mean_amount"], means, rtol=1e-9, atol=1e-8, equal_nan=True)
    assert actual.loc[source.proxy.isna(), "prior_transaction_count"].isna().all()
    for indices in eligible.groupby("proxy").groups.values():
        event_times = times[indices]
        prior = np.searchsorted(event_times, event_times, side="left")
        for seconds, feature in ((3600, "prior_count_1h"), (86400, "prior_count_24h")):
            window_start = np.searchsorted(event_times, event_times - seconds, side="left")
            np.testing.assert_array_equal(actual.loc[indices, feature], prior - window_start)
    summary = {"rows": len(actual), "eligible_entity_rows": len(eligible), "checks": [
        "IDs and split alignment", "feature contract excludes target",
        "strictly earlier global count including ties", "strictly earlier entity count and mean",
        "missing entities have unavailable history", "1h and 24h velocity including tied timestamps",
    ], "status": "passed"}
    path = root / "artifacts/features/verification.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
