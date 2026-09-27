import numpy as np
import pandas as pd
import pytest

from fraud_case.data import chronological_split, entity_proxy, merge_tables
from fraud_case.profiling import column_profiles
from fraud_case.schema import column_role


def tables():
    transactions = pd.DataFrame({"TransactionID": [3, 1, 2], "TransactionDT": [30, 10, 20],
                                 "TransactionAmt": [8., 10., 0.], "isFraud": [0, 1, 0]})
    identity = pd.DataFrame({"TransactionID": [1], "DeviceType": ["desktop"]})
    return transactions, identity


def test_left_join_keeps_transactions_without_identity():
    tx, identity = tables()
    result, audit = merge_tables(tx, identity)
    assert result.TransactionID.tolist() == [1, 2, 3]
    assert result.has_identity.tolist() == [True, False, False]
    assert audit["joined_rows"] == 3
    assert audit["zero_amount_rows"] == 1
    assert result.isFraud.sum() == tx.isFraud.sum()


def test_duplicate_identity_cannot_multiply_transactions():
    tx, identity = tables()
    with pytest.raises(ValueError, match="unique"):
        merge_tables(tx, pd.concat([identity, identity]))


def test_orphan_identity_is_rejected():
    tx, identity = tables()
    identity.loc[0, "TransactionID"] = 900
    with pytest.raises(ValueError, match="absent"):
        merge_tables(tx, identity)


def test_identity_presence_does_not_depend_on_populated_fields():
    tx, identity = tables()
    identity["DeviceType"] = pd.NA
    result, _ = merge_tables(tx, identity)
    assert result.loc[result.TransactionID.eq(1), "has_identity"].item()
    assert result.DeviceType.isna().all()


def test_null_join_key_is_rejected():
    tx, identity = tables()
    identity["TransactionID"] = pd.Series([pd.NA], dtype="Int64")
    with pytest.raises(ValueError, match="non-null"):
        merge_tables(tx, identity)


@pytest.mark.parametrize("column,value", [("TransactionAmt", -1), ("TransactionAmt", np.nan), ("TransactionDT", np.inf), ("isFraud", 2)])
def test_invalid_core_values_fail_before_analysis(column, value):
    tx, identity = tables()
    tx[column] = tx[column].astype(float)
    tx.loc[0, column] = value
    with pytest.raises(ValueError):
        merge_tables(tx, identity)


def test_equal_timestamps_never_cross_split_boundaries():
    frame = pd.DataFrame({"TransactionDT": [1, 1, 2, 3, 4, 4, 4, 5, 6, 7]})
    split = chronological_split(frame)
    assert split.nunique() == 3
    assert frame.assign(split=split).groupby("TransactionDT").split.nunique().max() == 1
    assert frame.loc[split.eq("train"), "TransactionDT"].max() < frame.loc[split.eq("validation"), "TransactionDT"].min()
    assert frame.loc[split.eq("validation"), "TransactionDT"].max() < frame.loc[split.eq("test"), "TransactionDT"].min()


def test_unsorted_data_and_degenerate_time_are_rejected():
    with pytest.raises(ValueError, match="Sort"):
        chronological_split(pd.DataFrame({"TransactionDT": [3, 1, 2]}))
    with pytest.raises(ValueError, match="distinct"):
        chronological_split(pd.DataFrame({"TransactionDT": [1] * 10}))


def test_incomplete_entities_are_not_grouped_together():
    frame = pd.DataFrame({c: [1, np.nan, np.nan] for c in ["card1", "card2", "card3", "card5", "addr1"]})
    proxy = entity_proxy(frame)
    assert proxy.notna().tolist() == [True, False, False]


def test_semantic_type_is_not_just_physical_dtype():
    numbers = pd.Series([1, 2, 3])
    assert column_role("card1", numbers)[0] == "categorical"
    assert column_role("TransactionID", numbers)[0] == "identifier"
    assert column_role("isFraud", numbers)[0] == "target"
    assert column_role("TransactionDT", numbers)[0] == "elapsed_time"
    assert column_role("TransactionAmt", numbers)[0] == "numeric"
    assert column_role("event_date", pd.Series(["2026-09-25", "2026-09-26"]))[0] == "datetime"
    assert column_role("postal_code", pd.Series(["34000", "22000"]))[0] == "categorical"


def test_profile_distinguishes_null_zero_and_infinity():
    frame = pd.DataFrame({"value": [0., 0., 0., 0., 100., np.nan, np.inf]})
    result = column_profiles(frame).iloc[0]
    assert result["null_ratio"] == pytest.approx(1 / 7)
    assert result["infinite_count"] == 1
    assert result["zero_iqr"]
    assert result["iqr_outlier_ratio"] == pytest.approx(1 / 5)
