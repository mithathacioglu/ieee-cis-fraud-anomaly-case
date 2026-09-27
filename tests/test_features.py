import numpy as np
import pandas as pd
import pytest

from fraud_case.features import FEATURE_SPECS, FeatureBuilder


def transactions(times=(0, 10, 20), amounts=(100., 120., 10000.)):
    n = len(times)
    return pd.DataFrame({
        "TransactionID": list(range(1, n + 1)), "TransactionDT": times, "TransactionAmt": amounts,
        "card1": [1] * n, "card2": [2.] * n, "card3": [3.] * n,
        "card5": [5.] * n, "addr1": [6.] * n, "card4": ["visa"] * n,
        "ProductCD": ["W"] * n, "P_emaildomain": ["mail.example"] * n,
        "R_emaildomain": ["recipient.example"] * n,
        "DeviceInfo": ["device-description"] * n, "has_identity": [True] * n,
    })


def test_current_amount_does_not_enter_its_own_history():
    result = FeatureBuilder().transform(transactions())
    assert result.prior_transaction_count.tolist() == [0, 1, 2]
    assert np.isnan(result.prior_mean_amount.iloc[0])
    assert result.prior_mean_amount.iloc[1:].tolist() == [100., 110.]
    assert result.amount_to_prior_mean.iloc[2] == pytest.approx(10000 / 110)
    assert result.prior_std_amount.iloc[2] == pytest.approx(10)


def test_future_changes_do_not_change_past_features():
    original = transactions()
    changed = original.copy()
    changed.loc[2, "TransactionAmt"] = 1e8
    changed.loc[2, "ProductCD"] = "unseen"
    before = FeatureBuilder().transform(original)
    after = FeatureBuilder().transform(changed)
    pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])


def test_equal_timestamps_do_not_observe_each_other():
    frame = transactions(times=(0, 0, 1))
    result = FeatureBuilder().transform(frame)
    assert result.prior_transaction_count.tolist() == [0, 0, 2]
    assert result.prior_global_count.tolist() == [0, 0, 2]
    assert result.prior_mean_amount.iloc[:2].isna().all()
    assert result.prior_mean_amount.iloc[2] == 110
    assert result.product_card_prior_support.tolist() == [0, 0, 2]


def test_tied_input_order_does_not_change_features():
    frame = transactions(times=(0, 0, 1))
    swapped = frame.iloc[[1, 0, 2]]
    left = FeatureBuilder().transform(frame).sort_values("TransactionID").reset_index(drop=True)
    right = FeatureBuilder().transform(swapped).sort_values("TransactionID").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_split_batches_match_a_single_chronological_pass():
    frame = transactions()
    builder = FeatureBuilder()
    chunks = [builder.transform(frame.iloc[:2]), builder.transform(frame.iloc[2:])]
    actual = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_frame_equal(actual, FeatureBuilder().transform(frame), check_dtype=False)


def test_batch_cannot_split_a_timestamp_or_go_backwards():
    builder = FeatureBuilder()
    frame = transactions(times=(0, 0, 1))
    builder.transform(frame.iloc[:1])
    with pytest.raises(ValueError, match="complete timestamp"):
        builder.transform(frame.iloc[1:])


def test_velocity_windows_are_left_inclusive_right_exclusive():
    frame = transactions(times=(0, 3600, 3601, 86400, 86401), amounts=(1.,) * 5)
    result = FeatureBuilder().transform(frame)
    assert result.prior_count_1h.tolist() == [0, 1, 1, 0, 1]
    assert result.prior_count_24h.tolist() == [0, 1, 2, 3, 3]


def test_missing_entities_do_not_share_history():
    frame = transactions()
    frame.loc[:1, "addr1"] = np.nan
    result = FeatureBuilder().transform(frame)
    assert result.entity_known.tolist() == [0, 0, 1]
    assert result.prior_transaction_count.iloc[:2].isna().all()
    assert result.prior_transaction_count.iloc[2] == 0


def test_missing_device_history_is_not_a_new_device_alert():
    frame = transactions()
    frame.loc[:1, "DeviceInfo"] = None
    result = FeatureBuilder().transform(frame)
    assert result.entity_device_is_new.isna().all()


def test_observed_new_device_and_product_are_flagged():
    frame = transactions()
    frame.loc[2, "DeviceInfo"] = "different-description"
    frame.loc[2, "ProductCD"] = "C"
    result = FeatureBuilder().transform(frame)
    assert result.entity_device_is_new.iloc[1:].tolist() == [0, 1]
    assert result.entity_product_is_new.iloc[1:].tolist() == [0, 1]
    assert result.product_prior_frequency.iloc[2] == 0


def test_missing_categories_are_excluded_from_frequency_denominator():
    frame = transactions()
    frame.loc[0, "P_emaildomain"] = None
    result = FeatureBuilder().transform(frame)
    assert np.isnan(result.email_prior_frequency.iloc[1])
    assert result.email_prior_frequency.iloc[2] == 1
    assert result.email_pair_prior_support.iloc[2] == 1


def test_calendar_is_unknown_without_reference():
    result = FeatureBuilder().transform(transactions())
    assert result[["hour", "day_of_week", "is_weekend", "is_business_hours"]].isna().all().all()
    assert result.calendar_available.eq(0).all()
    assert result.hour_phase.eq(0).all()


def test_explicit_calendar_scenario_supports_weekend_and_business_hours():
    frame = transactions(times=(9 * 3600, 17 * 3600, 86400 + 9 * 3600), amounts=(1.,) * 3)
    result = FeatureBuilder("2026-09-25T00:00:00+03:00").transform(frame)
    assert result.day_of_week.tolist() == [4, 4, 5]
    assert result.is_business_hours.tolist() == [1, 0, 0]
    assert result.is_weekend.tolist() == [0, 0, 1]
    with pytest.raises(ValueError, match="UTC offset"):
        FeatureBuilder("2026-09-25T00:00:00")


def test_labels_and_ids_are_not_model_features():
    frame = transactions()
    frame["isFraud"] = [0, 1, 1]
    first = FeatureBuilder().transform(frame)
    frame["isFraud"] = [1, 0, 0]
    second = FeatureBuilder().transform(frame)
    pd.testing.assert_frame_equal(first, second)
    assert {"TransactionID", "TransactionDT", "isFraud", "split"}.isdisjoint(FEATURE_SPECS)
    assert set(first.columns) == {"TransactionID", *FEATURE_SPECS}


def test_zero_mean_and_zero_variance_do_not_produce_infinity():
    frame = transactions(times=tuple(range(7)), amounts=(0.,) * 6 + (10000.,))
    result = FeatureBuilder().transform(frame)
    assert not np.isinf(result[list(FEATURE_SPECS)].to_numpy(dtype=float)).any()
    assert result.amount_zscore.isna().all()
    assert result.amount_to_prior_mean.isna().all()


def test_frequent_entity_requires_history_length_and_elapsed_days():
    frame = transactions(times=tuple(range(21)) + (8 * 86400,), amounts=(100.,) * 22)
    result = FeatureBuilder().transform(frame)
    assert result.frequent_entity.iloc[:-1].eq(0).all()
    assert result.frequent_entity.iloc[-1] == 1
    assert result.history_sufficient.iloc[4] == 0
    assert result.history_sufficient.iloc[5] == 1


def test_bad_batch_is_rejected_before_updating_state():
    builder = FeatureBuilder()
    frame = transactions()
    frame.loc[2, "TransactionAmt"] = -1
    with pytest.raises(ValueError, match="TransactionAmt"):
        builder.transform(frame)
    assert builder.global_count == 0
    assert not builder.entities


def test_integer_and_float_codes_resolve_to_same_entity():
    builder = FeatureBuilder()
    frame = transactions()
    builder.transform(frame.iloc[:2])
    last = frame.iloc[2:].copy()
    last["card1"] = last.card1.astype(float)
    assert builder.transform(last).prior_transaction_count.item() == 2
