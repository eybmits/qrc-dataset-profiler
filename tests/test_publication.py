import json
import os
import subprocess
import sys

import pandas as pd


def _write_inputs(root):
    atlas_dir = root / "atlas"
    analysis_dir = root / "analysis"
    attribution_dir = root / "attribution"
    atlas_dir.mkdir()
    analysis_dir.mkdir()
    attribution_dir.mkdir()
    pd.DataFrame(
        {
            "name": ["a", "b", "c"],
            "family": ["chaotic_map", "chaotic_map", "oscillatory"],
            "qrc_advantage": [0.1, 0.0, -0.2],
            "predicted_qrc_advantage": [0.08, 0.01, -0.19],
            "actual_usefulness_label": ["qrc_useful", "near_tie", "baseline_preferred"],
            "property_pc1": [0.0, 1.0, -1.0],
            "property_pc2": [0.2, 0.5, -0.4],
        }
    ).to_csv(atlas_dir / "qrc_usefulness_map.csv", index=False)
    pd.DataFrame(
        {
            "family": ["chaotic_map", "oscillatory"],
            "n": [2, 1],
            "mean_qrc_advantage": [0.05, -0.2],
            "qrc_useful_rate": [0.5, 0.0],
            "near_tie_rate": [0.5, 0.0],
            "baseline_preferred_rate": [0.0, 1.0],
            "label_accuracy": [1.0, 1.0],
            "dominant_category": ["qrc_useful", "baseline_preferred"],
        }
    ).to_csv(atlas_dir / "family_usefulness_summary.csv", index=False)
    pd.DataFrame(
        {
            "model": ["gradient_boosting"],
            "n_samples": [3],
            "n_features_used": [2],
            "regression_r2_mean": [0.7],
            "regression_mae_mean": [0.1],
            "classification_roc_auc_mean": [0.8],
            "top_features": ["r2_linear,nl_gain"],
        }
    ).to_csv(atlas_dir / "meta_model_summary.csv", index=False)
    pd.DataFrame({"feature": ["r2_linear", "nl_gain"], "importance_mean": [0.4, 0.2]}).to_csv(atlas_dir / "atlas_importances.csv", index=False)
    pd.DataFrame(
        {
            "family": ["overall", "chaotic_map", "oscillatory"],
            "n": [3, 2, 1],
            "mean_advantage": [-0.03, 0.05, -0.2],
            "mean_ci_low": [-0.2, -0.01, -0.3],
            "mean_ci_high": [0.1, 0.2, -0.1],
        }
    ).to_csv(analysis_dir / "family_advantage_bootstrap.csv", index=False)
    pd.DataFrame(
        {
            "feature_set": ["all", "without_predictability_proxies"],
            "regression_r2_mean": [0.7, 0.6],
            "classification_roc_auc_mean": [0.8, 0.75],
        }
    ).to_csv(analysis_dir / "robustness_summary.csv", index=False)
    pd.DataFrame(
        {
            "family": ["overall", "chaotic_map"],
            "n": [2, 2],
            "mean_delta_J0_minus_J1": [0.0, 0.1],
            "delta_ci_low": [-0.1, 0.01],
            "delta_ci_high": [0.1, 0.2],
            "mechanism_signal": ["null_or_mixed", "positive"],
        }
    ).to_csv(attribution_dir / "family_attribution_bootstrap.csv", index=False)
    return atlas_dir, analysis_dir, attribution_dir


def test_publication_package_cli_writes_figures_and_report(tmp_path):
    atlas_dir, analysis_dir, attribution_dir = _write_inputs(tmp_path)
    out_dir = tmp_path / "publication"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qrc_dataset_profiler.run_publication_package",
            "--atlas-dir",
            str(atlas_dir),
            "--analysis-dir",
            str(analysis_dir),
            "--attribution-dir",
            str(attribution_dir),
            "--out",
            str(out_dir),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    expected = {
        "ATLAS_REPORT.md",
        "fig1_qrc_usefulness_atlas.png",
        "fig1_qrc_usefulness_atlas.pdf",
        "fig2_evidence_controls.png",
        "fig2_evidence_controls.pdf",
        "publication_manifest.json",
    }
    assert expected <= {p.name for p in out_dir.iterdir()}
    manifest = json.loads((out_dir / "publication_manifest.json").read_text())
    assert manifest["n_rows"] == 3
    assert "not claim broad average" in manifest["claim_boundary"]
