import math

from qrc_dataset_profiler.spec import DatasetRecord, SCHEMA_VERSION


def test_dataset_record_defaults_and_serializes_params():
    rec = DatasetRecord(params={"a": 1, "b": [2, 3]})

    assert rec.schema_version == SCHEMA_VERSION == "v1"
    assert math.isnan(rec.ac_timescale)
    assert math.isnan(rec.true_lyapunov)
    assert rec.mem_capacity_valid is False
    assert rec.lyapunov_valid is False

    row = rec.to_row()
    assert row["params"] == "{'a': 1, 'b': [2, 3]}"
