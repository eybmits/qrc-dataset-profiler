import numpy as np
import pandas as pd

from qrc_dataset_profiler.spec import FRONTIER_TIER_A_FIELDS
from qrc_dataset_profiler.triage import (
    fit_triage_model,
    read_series_csv,
    report_to_text,
    triage_feature_row,
    window_series,
)


def _fake_discovery(n=80):
    rng = np.random.default_rng(12)
    rows = []
    for i in range(n):
        trend = i / max(1, n - 1)
        centroid = 1.0 - trend
        row = {
            "dataset_id": f"d{i}:0:800",
            "name": f"d{i}",
            "family": "unit_root_trend" if trend > 0.7 else "chaotic_map",
            "source": "synthetic",
            "task_type": "forecast",
            "params": "{'generator': 'fake'}",
            "seed": i,
            "length": 800,
            "base_generator": "fake",
            "best_qrc_advantage_vs_esn": 0.2 if trend > 0.75 and centroid < 0.25 else -0.1,
        }
        for feature in FRONTIER_TIER_A_FIELDS:
            row[feature] = float(rng.normal(scale=0.1))
        row["ext_trend_strength"] = trend
        row["ext_spectral_centroid"] = centroid
        rows.append(row)
    return pd.DataFrame(rows)


def test_read_series_csv_selects_named_column(tmp_path):
    path = tmp_path / "series.csv"
    pd.DataFrame({"time": [0, 1, 2], "value": [1.0, 2.5, 3.0]}).to_csv(path, index=False)

    values = read_series_csv(path, column="value")

    assert np.allclose(values, [1.0, 2.5, 3.0])


def test_window_series_uses_tail_by_default():
    values = window_series(np.arange(10), max_length=4, window="tail")

    assert values.tolist() == [6, 7, 8, 9]


def test_triage_feature_row_returns_recommendation_and_support():
    discovery = _fake_discovery()
    model = fit_triage_model(discovery, seed=3)
    row = discovery.iloc[[79]].copy()
    row["dataset_id"] = "user:0:800"
    row["name"] = "user"

    report = triage_feature_row(row, model, discovery_table=discovery)

    assert report["prediction"]["qrc_useful_probability"] >= 0.0
    assert report["prediction"]["recommendation"] in {
        "high_priority_qrc_test",
        "worth_testing_qrc_if_available",
        "esn_first_qrc_low_priority",
        "outside_atlas_support__run_direct_benchmark",
    }
    assert "support_score" in report["support"]
    assert "ext_trend_strength" in report["key_features"]
    assert "boundary:" in report_to_text(report)
