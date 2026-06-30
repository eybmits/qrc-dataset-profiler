import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def _write_visual_inputs(root):
    atlas_dir = root / "atlas"
    analysis_dir = root / "analysis"
    attribution_dir = root / "attribution"
    features_dir = root / "features"
    for path in (atlas_dir, analysis_dir, attribution_dir, features_dir):
        path.mkdir()

    rows = []
    families = ["chaotic_map", "oscillatory", "real_bridge"]
    labels = ["qrc_useful", "near_tie", "baseline_preferred"]
    advantages = [0.12, 0.02, -0.16, 0.09, -0.01, -0.22, 0.07, 0.00, -0.12]
    for i, adv in enumerate(advantages):
        family = families[i % len(families)]
        label = labels[0] if adv >= 0.05 else labels[1] if adv >= -0.05 else labels[2]
        rows.append(
            {
                "dataset_id": f"d{i}:1:200",
                "name": f"dataset_{i}",
                "family": family,
                "source": "synthetic" if family != "real_bridge" else "real",
                "task_type": "forecast",
                "seed": 100 + i,
                "length": 200,
                "horizon": 3,
                "nrmse_esn_matched": 0.7,
                "nrmse_qrc_spin": 0.7 - adv,
                "qrc_advantage": adv,
                "actual_usefulness_label": label,
                "predicted_qrc_advantage": adv * 0.8,
                "predicted_usefulness_label": label,
                "predicted_prob_qrc_useful": min(max(0.45 + adv, 0.02), 0.98),
                "property_pc1": float(np.cos(i)),
                "property_pc2": float(np.sin(i)),
                "abs_prediction_error": abs(adv * 0.2),
                "prediction_correct_label": True,
            }
        )
    atlas = pd.DataFrame(rows)
    atlas.to_csv(atlas_dir / "qrc_usefulness_map.csv", index=False)

    family = (
        atlas.groupby("family")
        .agg(
            n=("qrc_advantage", "size"),
            mean_qrc_advantage=("qrc_advantage", "mean"),
            median_qrc_advantage=("qrc_advantage", "median"),
            mean_predicted_qrc_advantage=("predicted_qrc_advantage", "mean"),
            qrc_useful_rate=("actual_usefulness_label", lambda s: float((s == "qrc_useful").mean())),
            near_tie_rate=("actual_usefulness_label", lambda s: float((s == "near_tie").mean())),
            baseline_preferred_rate=("actual_usefulness_label", lambda s: float((s == "baseline_preferred").mean())),
            label_accuracy=("prediction_correct_label", "mean"),
            mean_prob_qrc_useful=("predicted_prob_qrc_useful", "mean"),
        )
        .reset_index()
    )
    family["dominant_category"] = "near_tie"
    family.to_csv(atlas_dir / "family_usefulness_summary.csv", index=False)
    pd.DataFrame(
        {
            "model": ["gradient_boosting"],
            "n_samples": [len(atlas)],
            "n_features_used": [4],
            "regression_r2_mean": [0.66],
            "regression_mae_mean": [0.12],
            "classification_roc_auc_mean": [0.84],
            "top_features": ["r2_linear,spectral_entropy,nl_gain,ac_timescale"],
        }
    ).to_csv(atlas_dir / "meta_model_summary.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["r2_linear", "spectral_entropy", "nl_gain", "ac_timescale"],
            "importance_mean": [0.4, 0.3, 0.2, 0.1],
            "direction": ["negative", "positive", "positive", "positive"],
        }
    ).to_csv(atlas_dir / "atlas_importances.csv", index=False)

    family_ci = pd.concat(
        [
            pd.DataFrame(
                {
                    "family": ["overall"],
                    "n": [len(atlas)],
                    "mean_advantage": [atlas["qrc_advantage"].mean()],
                    "mean_ci_low": [-0.12],
                    "mean_ci_high": [0.08],
                }
            ),
            family.assign(
                mean_advantage=family["mean_qrc_advantage"],
                mean_ci_low=family["mean_qrc_advantage"] - 0.05,
                mean_ci_high=family["mean_qrc_advantage"] + 0.05,
            )[["family", "n", "mean_advantage", "mean_ci_low", "mean_ci_high"]],
        ],
        ignore_index=True,
    )
    family_ci.to_csv(analysis_dir / "family_advantage_bootstrap.csv", index=False)
    pd.DataFrame(
        {
            "feature_set": ["all", "without_r2_linear", "without_predictability_proxies", "chaos_nonlinearity_complexity_only"],
            "regression_r2_mean": [0.66, 0.63, 0.61, 0.55],
            "classification_roc_auc_mean": [0.84, 0.82, 0.80, 0.76],
        }
    ).to_csv(analysis_dir / "robustness_summary.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["r2_linear", "spectral_entropy", "nl_gain", "ac_timescale"],
            "importance_mean": [0.4, 0.3, 0.2, 0.1],
            "ci_low": [0.3, 0.2, 0.1, 0.05],
            "ci_high": [0.5, 0.4, 0.3, 0.2],
            "selection_rate": [1.0, 1.0, 0.8, 0.6],
            "direction": ["negative", "positive", "positive", "positive"],
            "corr_with_advantage": [-0.4, 0.3, 0.2, 0.1],
        }
    ).to_csv(analysis_dir / "importance_bootstrap.csv", index=False)

    pd.DataFrame(
        {
            "family": ["overall", "chaotic_map", "oscillatory"],
            "n": [6, 3, 3],
            "mean_delta_J0_minus_J1": [0.0, 0.05, -0.04],
            "delta_ci_low": [-0.05, 0.01, -0.08],
            "delta_ci_high": [0.05, 0.10, -0.01],
            "mechanism_signal": ["null_or_mixed", "positive", "negative"],
        }
    ).to_csv(attribution_dir / "family_attribution_bootstrap.csv", index=False)
    pd.DataFrame(
        {
            "family": ["chaotic_map", "chaotic_map", "oscillatory", "oscillatory"],
            "nrmse_qrc_J0": [0.8, 0.7, 0.6, 0.5],
            "nrmse_qrc_J1": [0.75, 0.72, 0.65, 0.54],
        }
    ).to_csv(attribution_dir / "paired_attribution.csv", index=False)

    features = atlas[["dataset_id", "name", "family", "source", "task_type", "seed", "length"]].copy()
    features["ext_sample_entropy_m2"] = np.linspace(0.1, 0.9, len(features))
    features["ext_lz_complexity"] = np.linspace(0.2, 1.0, len(features))[::-1]
    features["ext_trend_strength"] = np.linspace(0.0, 0.5, len(features))
    features["ext_recurrence_rate"] = np.linspace(0.8, 0.1, len(features))
    features.to_csv(features_dir / "extended_features_sweep.csv", index=False)

    sweep = atlas[["dataset_id", "name", "family"]].copy()
    sweep["params"] = ["{'generator': 'logistic'}", "{'generator': 'mso'}", "{'generator': 'santa_fe_laser'}"] * 3
    sweep["r2_linear"] = np.linspace(0.1, 0.9, len(sweep))
    sweep["spectral_entropy"] = np.linspace(0.9, 0.1, len(sweep))
    sweep["nl_gain"] = np.linspace(-0.2, 0.4, len(sweep))
    sweep["ac_timescale"] = np.arange(2, 2 + len(sweep))
    sweep.to_csv(root / "sweep_catalog.csv", index=False)

    full = pd.DataFrame(
        {
            "name": ["logistic_r4", "mso8", "santa_fe_laser"],
            "family": ["chaotic_map", "oscillatory", "real_bridge"],
            "qrc_advantage": [0.10, -0.20, -0.08],
            "ac_timescale": [4, 10, 12],
            "r2_linear": [0.2, 0.9, 0.8],
            "nl_gain": [0.4, 0.1, 0.2],
            "spectral_entropy": [0.8, 0.2, 0.4],
            "pred_nrmse_gbm": [0.7, 0.2, 0.5],
        }
    )
    full.to_csv(root / "full_catalog.csv", index=False)

    return atlas_dir, analysis_dir, attribution_dir, features_dir, root / "sweep_catalog.csv", root / "full_catalog.csv"


def test_visual_suite_cli_writes_index_manifest_and_figures(tmp_path):
    atlas_dir, analysis_dir, attribution_dir, features_dir, sweep, full = _write_visual_inputs(tmp_path)
    out_dir = tmp_path / "visuals"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qrc_dataset_profiler.run_visual_suite",
            "--atlas-dir",
            str(atlas_dir),
            "--analysis-dir",
            str(analysis_dir),
            "--attribution-dir",
            str(attribution_dir),
            "--features-dir",
            str(features_dir),
            "--sweep-catalog",
            str(sweep),
            "--full-catalog",
            str(full),
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
        "00_visual_abstract.png",
        "01_property_landscape.png",
        "02_sweep_barcode.png",
        "03_family_outcomes.png",
        "04_advantage_distributions.png",
        "05_meta_model_evidence.png",
        "06_extended_feature_map.png",
        "07_quantum_attribution_guardrail.png",
        "08_full_catalog_inventory.png",
        "09_all_points_feature_regressions.png",
        "index.html",
        "VISUAL_SUITE_REPORT.md",
        "visual_suite_manifest.json",
    }
    assert expected <= {p.name for p in out_dir.iterdir()}
    manifest = json.loads((out_dir / "visual_suite_manifest.json").read_text())
    assert manifest["n_rows"] == 9
    assert len(manifest["figures"]) == 10
    assert "does not claim broad average" in manifest["claim_boundary"]
