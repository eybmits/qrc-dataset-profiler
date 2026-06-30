import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from qrc_dataset_profiler.spec import FRONTIER_TIER_A_FIELDS


def _fake_evaluated_frontier(n=90, seed=0, split="discovery"):
    rng = np.random.default_rng(seed)
    rows = []
    families = ["chaotic_flow", "nonstationary", "oscillatory"]
    generators = ["lorenz63", "chirp", "mso"]
    for i in range(n):
        signal = rng.normal()
        advantage = 0.12 * signal + 0.08 * (i % 9 == 0) - 0.05 * (i % 5 == 0) + rng.normal(scale=0.04)
        row = {
            "dataset_id": f"{split}_{i}:s{i}:{800+i}",
            "name": f"{split}_{i}",
            "family": families[i % len(families)],
            "source": "synthetic",
            "task_type": "forecast",
            "params": "{'generator': 'lorenz63'}",
            "seed": i,
            "length": 800 + i,
            "horizon": 1,
            "base_generator": generators[i % len(generators)],
            "qrc_advantage": advantage,
            "nrmse_linear": 1.0,
            "nrmse_esn_matched": 0.8,
            "nrmse_qrc_spin": 0.8 - advantage,
            "nrmse_gbm": 0.7,
        }
        for j, feature in enumerate(FRONTIER_TIER_A_FIELDS):
            row[feature] = float(signal + 0.02 * j + rng.normal(scale=0.35))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_analysis_dir(path):
    path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model_suite": "frontier_30_tier_a",
                "n_rows": 90,
                "n_features_declared": 30,
                "n_features_used": 25,
                "qrc_win_rate": 0.3,
                "qrc_useful_rate": 0.15,
                "mean_advantage": -0.02,
                "median_advantage": -0.01,
                "regression_r2_mean": 0.4,
                "regression_mae_mean": 0.1,
                "classification_roc_auc_mean": 0.75,
                "classification_pr_auc_mean": 0.2,
                "classification_brier_mean": 0.08,
                "top_features": "ext_psd_slope,nl_gain",
                "notes": "",
            }
        ]
    ).to_csv(path / "frontier_meta_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "group_col": "base_generator",
                "n_groups": 3,
                "n_splits": 3,
                "regression_r2_mean": 0.2,
                "regression_mae_mean": 0.12,
                "classification_roc_auc_mean": 0.65,
                "classification_pr_auc_mean": 0.18,
            },
            {
                "group_col": "family",
                "n_groups": 3,
                "n_splits": 3,
                "regression_r2_mean": -0.1,
                "regression_mae_mean": 0.2,
                "classification_roc_auc_mean": 0.55,
                "classification_pr_auc_mean": 0.12,
            },
        ]
    ).to_csv(path / "frontier_grouped_validation.csv", index=False)
    pd.DataFrame(
        [
            {
                "rule": "dfa_alpha > 0.8 and nl_gain <= 0.1",
                "n": 22,
                "qrc_useful_rate": 0.31,
                "qrc_win_rate": 0.45,
                "mean_advantage": 0.04,
            }
        ]
    ).to_csv(path / "frontier_rule_table.csv", index=False)


def test_frontier_plot_cli_writes_publication_figures(tmp_path):
    discovery = _fake_evaluated_frontier(n=90, seed=1, split="discovery")
    validation = _fake_evaluated_frontier(n=90, seed=2, split="validation")
    discovery_path = tmp_path / "discovery.csv"
    validation_path = tmp_path / "validation.csv"
    discovery.to_csv(discovery_path, index=False)
    validation.to_csv(validation_path, index=False)
    discovery_analysis = tmp_path / "discovery_analysis"
    validation_analysis = tmp_path / "validation_analysis"
    _write_analysis_dir(discovery_analysis)
    _write_analysis_dir(validation_analysis)

    out_dir = tmp_path / "plots"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qrc_dataset_profiler.run_frontier",
            "plot",
            "--discovery-table",
            str(discovery_path),
            "--validation-table",
            str(validation_path),
            "--discovery-analysis-dir",
            str(discovery_analysis),
            "--validation-analysis-dir",
            str(validation_analysis),
            "--out",
            str(out_dir),
            "--formats",
            "png",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    expected = {
        "frontier_01_validation_regime_map.png",
        "frontier_02_prospective_meta_model.png",
        "frontier_03_all_points_feature_regressions.png",
        "frontier_04_rules_and_claim_boundary.png",
        "frontier_prospective_predictions.csv",
        "frontier_publication_metrics.csv",
        "frontier_publication_plots_manifest.json",
        "index.html",
    }
    assert expected.issubset({p.name for p in out_dir.iterdir()})
    manifest = json.loads((out_dir / "frontier_publication_plots_manifest.json").read_text())
    assert manifest["n_validation"] == 90
    assert len(manifest["figures"]) == 4
