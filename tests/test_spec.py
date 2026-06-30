import math

from qrc_dataset_profiler.spec import CORE_AXIS_FIELDS, DatasetRecord, FRONTIER_TIER_A_FIELDS, TIER_A_UPGRADE_FIELDS, SCHEMA_VERSION


def test_dataset_record_defaults_and_serializes_params():
    rec = DatasetRecord(params={"a": 1, "b": [2, 3]})

    assert rec.schema_version == SCHEMA_VERSION == "v2"
    assert math.isnan(rec.ac_timescale)
    assert math.isnan(rec.sample_entropy)
    assert math.isnan(rec.hurst_rs)
    assert math.isnan(rec.pred_nrmse_linear)
    assert math.isnan(rec.predictability_gap_linear_gbm)
    assert math.isnan(rec.true_lyapunov)
    assert rec.mem_capacity_valid is False
    assert rec.lyapunov_valid is False

    row = rec.to_row()
    assert row["params"] == "{'a': 1, 'b': [2, 3]}"


def test_primary_feature_contract_has_20_features():
    assert len(CORE_AXIS_FIELDS) == 20
    assert "sample_entropy" in CORE_AXIS_FIELDS
    assert "hurst_rs" in CORE_AXIS_FIELDS


def test_frontier_feature_contract_has_30_tier_a_features():
    assert len(TIER_A_UPGRADE_FIELDS) == 10
    assert len(FRONTIER_TIER_A_FIELDS) == 30
    assert "predictability_gap_linear_gbm" in FRONTIER_TIER_A_FIELDS
