import json

import numpy as np
import pandas as pd

from qrc_dataset_profiler.real_validation import _windows, select_real_label_subset, summarize_real_predictions


def _fake_predictions(n=24):
    rng = np.random.default_rng(0)
    rows = []
    domains = ["weather", "electricity", "finance"]
    for i in range(n):
        rows.append(
            {
                "dataset_id": f"real_{i}",
                "name": f"real_{i}",
                "family": f"real_{domains[i % len(domains)]}",
                "source": "real",
                "task_type": "forecast",
                "params": "{}",
                "seed": 0,
                "length": 800,
                "horizon": 1,
                "window_key": f"w{i:05d}",
                "source_name": f"source_{i % 4}",
                "series_id": f"s{i}",
                "real_domain": domains[i % len(domains)],
                "source_url": "https://example.com",
                "predicted_qrc_advantage": float(rng.normal()),
                "predicted_prob_qrc_useful": float(rng.random()),
                "support_score": float(rng.random()),
                "ood_flag": bool(i % 7 == 0),
            }
        )
    return pd.DataFrame(rows)


def test_windows_are_deterministic_and_respect_short_series():
    short = np.arange(30.0)
    long = np.arange(1000.0)

    assert _windows(short, window_length=100, min_length=50, max_windows=3) == []
    windows = _windows(long, window_length=100, min_length=50, max_windows=3)

    assert len(windows) == 3
    assert windows[0][1] == 0
    assert windows[-1][1] == 900
    assert all(values.size == 100 for _, _, values in windows)


def test_select_real_label_subset_is_target_free_and_writes_manifest(tmp_path):
    pred = _fake_predictions()
    pred_path = tmp_path / "pred.csv"
    pred.to_csv(pred_path, index=False)

    selected, path = select_real_label_subset(predictions_path=pred_path, out_dir=tmp_path, n_rows=12, seed=2)

    assert path.name == "real_label_selection.csv"
    assert len(selected) == 12
    assert "real_selection_role" in selected.columns
    assert selected["real_selection_role"].notna().all()
    manifest = json.loads((tmp_path / "real_label_selection_manifest.json").read_text())
    assert manifest["selection_uses_real_labels"] is False


def test_summarize_real_predictions_groups_sources_and_domains():
    pred = _fake_predictions()
    summary = summarize_real_predictions(pred)

    assert "overall" in set(summary["group"])
    assert any(str(v).startswith("source:") for v in summary["group"])
    assert any(str(v).startswith("domain:") for v in summary["group"])
    assert summary["ood_rate"].between(0.0, 1.0).all()

