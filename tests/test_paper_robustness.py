import json

import numpy as np
import pandas as pd

from qrc_dataset_profiler.paper_robustness import (
    feature_family_ablation,
    regime_enrichment_table,
    run_paper_robustness_suite,
    threshold_robustness,
)
from qrc_dataset_profiler.spec import FRONTIER_TIER_A_FIELDS


def _fake_frontier(n=90, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    families = ["chaotic_flow", "unit_root_trend", "oscillatory_quasiperiodic", "linear_stochastic"]
    for i in range(n):
        persistence = rng.normal()
        trend = rng.normal()
        chaos = rng.normal()
        adv = 0.08 * persistence + 0.06 * trend - 0.03 * chaos + rng.normal(scale=0.04)
        if families[i % len(families)] == "unit_root_trend":
            adv += 0.07
        row = {
            "dataset_id": f"d{i}",
            "name": f"d{i}",
            "family": families[i % len(families)],
            "source": "synthetic",
            "task_type": "forecast",
            "params": "{'generator': 'ar'}",
            "seed": i,
            "length": 800,
            "horizon": 1,
            "base_generator": "ar",
            "qrc_advantage": adv,
            "nrmse_esn_matched": 0.8,
            "nrmse_qrc_spin": 0.8 - adv,
            "ac_timescale": 24 + persistence,
            "dfa_alpha": 1.0 + 0.1 * persistence,
            "ext_trend_strength": abs(trend),
            "ext_psd_slope": -1.0 - persistence,
            "dom_freq": 0.01 if i % 3 else 0.2,
            "perm_entropy": 0.65 + 0.05 * rng.normal(),
            "spectral_entropy": 0.5 + 0.05 * rng.normal(),
            "lyapunov": chaos,
            "nl_gain": -0.1 * chaos,
            "zero_one_K": abs(chaos),
            "ext_seasonality_strength": 0.9 if families[i % len(families)] == "oscillatory_quasiperiodic" else 0.1,
        }
        for feature in FRONTIER_TIER_A_FIELDS:
            row.setdefault(feature, float(rng.normal()))
        rows.append(row)
    return pd.DataFrame(rows)


def test_feature_and_threshold_robustness_return_named_feature_sets():
    discovery = _fake_frontier(seed=1)
    validation = _fake_frontier(seed=2)

    ablation = feature_family_ablation(discovery, validation, threshold=0.05, seed=0)
    thresholds = threshold_robustness(discovery, validation, thresholds=(0.0, 0.05), seed=0)

    assert {"full_30", "chaos_nonlinearity_complexity_only", "without_direct_predictability"} <= set(ablation["feature_set"])
    assert set(thresholds["threshold"]) == {0.0, 0.05}
    assert ablation["roc_auc"].notna().any()


def test_regime_enrichment_contains_overall_and_target_pocket():
    validation = _fake_frontier(seed=3)
    table = regime_enrichment_table(validation, rng=np.random.default_rng(0), n_bootstraps=10)

    assert "overall" in set(table["regime"])
    assert "persistence_drift_low_frequency_moderate_complexity" in set(table["regime"])
    assert (table["qrc_useful_rate"].between(0.0, 1.0)).all()


def test_paper_robustness_suite_writes_core_outputs(tmp_path):
    discovery = _fake_frontier(seed=4)
    validation = _fake_frontier(seed=5)
    discovery_path = tmp_path / "discovery.csv"
    validation_path = tmp_path / "validation.csv"
    discovery.to_csv(discovery_path, index=False)
    validation.to_csv(validation_path, index=False)

    manifest = run_paper_robustness_suite(
        discovery_table=discovery_path,
        validation_table=validation_path,
        out_dir=tmp_path / "robustness",
        calibration_config=None,
        thresholds=(0.0, 0.05),
        metric_subset_n=0,
        mechanism_rows=0,
        real_probes=False,
        formats=("png",),
    )

    out_dir = tmp_path / "robustness"
    assert manifest["n_discovery"] == len(discovery)
    assert (out_dir / "feature_family_ablation.csv").exists()
    assert (out_dir / "threshold_robustness.csv").exists()
    assert (out_dir / "regime_enrichment.csv").exists()
    assert (out_dir / "PAPER_ROBUSTNESS_REPORT.md").exists()
    written = json.loads((out_dir / "paper_robustness_manifest.json").read_text())
    assert written["metric_subset_n_written"] == 0

