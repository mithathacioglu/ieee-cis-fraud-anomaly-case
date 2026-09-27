"""Yükleme, birleştirme, bölme. Aynı zaman damgası aynı bölümde kalır."""

from pathlib import Path

import numpy as np
import pandas as pd


def validate_key(frame: pd.DataFrame, name: str) -> None:
    if "TransactionID" not in frame:
        raise ValueError(f"{name}: missing TransactionID")
    key = frame["TransactionID"]
    if key.isna().any() or key.duplicated().any():
        raise ValueError(f"{name}: TransactionID must be non-null and unique")
    if not pd.api.types.is_integer_dtype(key.dtype):
        raise ValueError(f"{name}: TransactionID must be integer")


def merge_tables(transactions: pd.DataFrame, identity: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    validate_key(transactions, "transactions")
    validate_key(identity, "identity")
    required = {"TransactionDT", "TransactionAmt", "isFraud"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"transactions: missing columns {sorted(missing)}")
    for column in ("TransactionDT", "TransactionAmt"):
        values = transactions[column]
        if not pd.api.types.is_numeric_dtype(values) or not np.isfinite(values).all():
            raise ValueError(f"transactions: {column} must contain finite numbers")
        if (values < 0).any():
            raise ValueError(f"transactions: negative {column}")
    if not transactions["isFraud"].isin([0, 1]).all():
        raise ValueError("transactions: isFraud must be 0 or 1")
    collisions = (set(transactions.columns) & set(identity.columns)) - {"TransactionID"}
    if collisions or "has_identity" in transactions or "has_identity" in identity:
        raise ValueError(f"Unexpected overlapping/reserved columns: {sorted(collisions)}")
    orphan_ids = ~identity["TransactionID"].isin(transactions["TransactionID"])
    if orphan_ids.any():
        raise ValueError(f"identity: {int(orphan_ids.sum())} IDs absent from transactions")
    joined = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one", indicator=True)
    joined["has_identity"] = joined.pop("_merge").eq("both")
    audit = {
        "transaction_rows": len(transactions),
        "identity_rows": len(identity),
        "joined_rows": len(joined),
        "joined_columns": len(joined.columns),
        "identity_coverage": float(joined["has_identity"].mean()),
        "orphan_identity_ids": 0,
        "duplicate_transaction_ids": 0,
        "fraud_rows": int(joined["isFraud"].sum()),
        "fraud_rate": float(joined["isFraud"].mean()),
        "zero_amount_rows": int(joined["TransactionAmt"].eq(0).sum()),
        "rows_without_identity": int((~joined["has_identity"]).sum()),
    }
    return joined.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True), audit


def load_data(directory: Path) -> tuple[pd.DataFrame, dict]:
    # Geniş join'den önce parser bloklarını bir kez topla
    transactions = pd.read_csv(directory / "train_transaction.csv", low_memory=False).copy()
    identity = pd.read_csv(directory / "train_identity.csv", low_memory=False).copy()
    return merge_tables(transactions, identity)


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.6, validation_fraction: float = 0.2) -> pd.Series:
    """Keep equal timestamps together; never shuffle future rows into training."""
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 - train_fraction):
        raise ValueError("Invalid split fractions")
    times = frame["TransactionDT"]
    if times.isna().any() or not times.is_monotonic_increasing:
        raise ValueError("Sort by TransactionDT before splitting")
    n = len(times)
    if n < 3:
        raise ValueError("At least three rows are required")
    first = times.iloc[int(n * train_fraction)]
    second = times.iloc[int(n * (train_fraction + validation_fraction))]
    split = pd.Series(np.where(times < first, "train", np.where(times < second, "validation", "test")), index=frame.index, name="split")
    if split.nunique() != 3:
        raise ValueError("Not enough distinct timestamps for three nonempty splits")
    return split


def entity_proxy(frame: pd.DataFrame) -> pd.Series:
    """A card/address grouping, not an authenticated customer identifier."""
    columns = ["card1", "card2", "card3", "card5", "addr1"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing proxy components: {sorted(missing)}")
    components = frame[columns].astype("string")
    valid = components.notna().all(axis=1)
    result = components.fillna("").agg("|".join, axis=1).astype("string")
    # Eksik anahtar, alakasız satırları aynı entity'de toplamasın
    return result.where(valid, pd.NA).rename("entity_proxy")
