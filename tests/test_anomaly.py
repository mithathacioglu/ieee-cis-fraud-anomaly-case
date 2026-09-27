import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_case.aggregation import ScoreAggregator
from fraud_case.anomaly import LAYERS, REQUIRED_FEATURES, AnomalyEngine, RobustReference


def features():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(0.0, index=range(100), columns=REQUIRED_FEATURES)
    frame["amount"] = rng.lognormal(np.log(100), 0.2, len(frame))
    frame["log_amount"] = np.log1p(frame.amount)
    frame["prior_mean_amount"] = 100.
    frame["prior_std_amount"] = 20.
    frame["prior_transaction_count"] = 10.
    frame["amount_to_prior_mean"] = frame.amount / 100
    frame["prior_count_1h"] = 1.
    frame["prior_count_24h"] = 3.
    frame["entity_hour_probability"] = 1 / 24
    frame["entity_known"] = 1.
    frame["has_identity"] = 1.
    frame["split"] = "train"
    return frame


def fitted():
    return AnomalyEngine(n_estimators=16, max_samples=64).fit(features())


def test_extreme_amount_increases_column_and_entity_scores():
    engine = fitted()
    rows = features().iloc[:2].copy()
    rows["amount"] = [100., 100000.]
    rows["log_amount"] = np.log1p(rows.amount)
    result = engine.score(rows)
    assert result.column_raw.iloc[1] > result.column_raw.iloc[0]
    assert result.entity_raw.iloc[1] > result.entity_raw.iloc[0]


def test_missing_history_leaves_two_layers_unavailable():
    rows = features().iloc[:2].copy()
    rows["prior_transaction_count"] = [0, np.nan]
    result = fitted().score(rows)
    assert result[["entity_raw", "temporal_raw"]].isna().all().all()
    assert result[["column_available", "multivariate_available"]].all().all()


def test_higher_velocity_increases_temporal_score():
    rows = features().iloc[:2].copy()
    rows["prior_count_1h"] = [1, 100]
    result = fitted().score(rows)
    assert result.temporal_raw.iloc[1] > result.temporal_raw.iloc[0]


def test_score_does_not_depend_on_other_rows_in_request():
    engine = fitted()
    rows = features().iloc[:3].copy()
    batch = engine.score(rows)
    single = engine.score(rows.iloc[[0]])
    pd.testing.assert_frame_equal(batch.iloc[[0]], single)


def test_engine_rejects_validation_fit_and_infinite_inputs():
    rows = features()
    rows["split"] = "validation"
    with pytest.raises(ValueError, match="training partition"):
        AnomalyEngine().fit(rows)
    rows["split"] = "train"
    rows.loc[0, "amount"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        AnomalyEngine().fit(rows)


def test_forest_and_scoring_ignore_label_and_identifier_columns():
    rows = features()
    rows["isFraud"] = 0
    rows["TransactionID"] = range(len(rows))
    first = AnomalyEngine(n_estimators=16, max_samples=64).fit(rows).score(rows)
    rows["isFraud"] = 1
    rows["TransactionID"] += 10000
    second = AnomalyEngine(n_estimators=16, max_samples=64).fit(rows).score(rows)
    pd.testing.assert_frame_equal(first, second)


def test_constant_robust_reference_has_defined_scale():
    reference = RobustReference.fit(pd.Series([0.] * 10))
    assert reference.scale == 1
    assert reference.deviation(pd.Series([0., 3.])).tolist() == [0., 3.]


def raw_scores(values):
    return pd.DataFrame({f"{layer}_raw": values for layer in LAYERS})


def test_normalization_handles_ties_and_unseen_extremes():
    aggregator = ScoreAggregator().fit(raw_scores([0., 0., 0., 2.]))
    result = aggregator.transform(raw_scores([0., 1., 2., 3.]))
    assert result.column_normalized.tolist() == [0., 0.75, 0.75, 1.]
    assert result.raw_anomaly_score.tolist() == [0., 0.75, 0.75, 1.]


def test_unavailable_scores_are_not_treated_as_safe_zeros():
    aggregator = ScoreAggregator().fit(raw_scores([0., 1., 2.]))
    rows = raw_scores([3.])
    rows[["entity_raw", "temporal_raw"]] = np.nan
    result = aggregator.transform(rows).iloc[0]
    assert result.raw_anomaly_score == 1
    assert result.available_weight == 0.5
    assert result.column_effective_weight == 0.5
    assert result.entity_effective_weight == 0
    assert np.isnan(result.entity_normalized)
    assert sum(result[f"{layer}_contribution"] for layer in LAYERS) == result.raw_anomaly_score


def test_all_missing_layers_fail_instead_of_producing_zero_risk():
    aggregator = ScoreAggregator().fit(raw_scores([0., 1.]))
    with pytest.raises(ValueError, match="no positively weighted"):
        aggregator.transform(raw_scores([np.nan]))


def test_bad_weights_and_empty_calibration_are_rejected():
    with pytest.raises(ValueError, match="four layer"):
        ScoreAggregator({"column": 1})
    with pytest.raises(ValueError, match="positive total"):
        ScoreAggregator(dict.fromkeys(LAYERS, 0))
    with pytest.raises(ValueError, match="No training reference"):
        ScoreAggregator().fit(raw_scores([np.nan]))


def test_model_roundtrip_preserves_scores_and_explanation(tmp_path):
    engine = fitted()
    rows = features().iloc[:3]
    aggregator = ScoreAggregator().fit(engine.score(features()))
    expected = aggregator.transform(engine.score(rows))
    path = tmp_path / "model.joblib"
    joblib.dump((engine, aggregator), path)
    loaded_engine, loaded_aggregator = joblib.load(path)
    actual = loaded_aggregator.transform(loaded_engine.score(rows))
    pd.testing.assert_frame_equal(expected, actual)
    assert loaded_engine.explain(rows.iloc[[0]]) == engine.explain(rows.iloc[[0]])
