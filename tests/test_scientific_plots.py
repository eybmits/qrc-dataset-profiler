import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def _write_inputs(root):
    atlas_dir = root / "atlas"
    analysis_dir = root / "analysis"
    attribution_dir = root / "attribution"
    features_dir = root / "features"
    for path in (atlas_dir, analysis_dir, attribution_dir, features_dir):
        path.mkdir()

    rng = np.random.default_rng(3)
    families = ["chaotic_map", "chaotic_flow", "oscillatory", "real_bridge"]
    rows = []
    for i in range(32):
        family = families[i % len(families)]
        adv = float(0.18 * np.sin(i / 3.0) - 0.08 * (family == "oscillatory") - 0.10 * (family == "real_bridge"))
        label = "qrc_useful" if adv >= 0.05 else "near_tie" if adv >= -0.05 else "baseline_preferred"
        pred = float(adv * 0.75 + rng.normal(0, 0.015))
        pred_label = "qrc_useful" if pred >= 0.05 else "near_tie" if pred >= -0.05 else "baseline_preferred"
        rows.append(
            {
                "dataset_id": f"d{i}:1:200",
                "name": f"d{i}",
                "family": family,
                "qrc_advantage": adv,
                "actual_usefulness_label": label,
                "predicted_qrc_advantage": pred,
                "predicted_usefulness_label": pred_label,
                "predicted_prob_qrc_useful": float(np.clip(0.45 + pred, 0.02, 0.98)),
                "property_pc1": float(np.cos(i / 4.0) + rng.normal(0, 0.05)),
                "property_pc2": float(np.sin(i / 4.0) + rng.normal(0, 0.05)),
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
            label_accuracy=("actual_usefulness_label", "size"),
            mean_prob_qrc_useful=("predicted_prob_qrc_useful", "mean"),
        )
        .reset_index()
    )
    family["label_accuracy"] = 0.75
    family["dominant_category"] = "near_tie"
    family.to_csv(atlas_dir / "family_usefulness_summary.csv", index=False)
    pd.DataFrame(
        {
            "model": ["gradient_boosting"],
            "n_samples": [len(atlas)],
            "n_features_used": [8],
            "regression_r2_mean": [0.61],
            "regression_mae_mean": [0.13],
            "classification_roc_auc_mean": [0.82],
            "top_features": ["r2_linear,spectral_entropy,nl_gain,ac_timescale"],
        }
    ).to_csv(atlas_dir / "meta_model_summary.csv", index=False)

    feature_names = ["r2_linear", "spectral_entropy", "nl_gain", "ac_timescale", "zero_one_K", "pred_nrmse_gbm"]
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": [0.4, 0.28, 0.2, 0.15, 0.11, 0.08],
            "ci_low": [0.25, 0.16, 0.09, 0.05, 0.04, 0.02],
            "ci_high": [0.55, 0.40, 0.29, 0.23, 0.19, 0.14],
            "selection_rate": [1.0] * len(feature_names),
            "direction": ["negative", "positive", "positive", "positive", "positive", "negative"],
            "corr_with_advantage": [-0.3, 0.25, 0.2, 0.1, 0.16, -0.1],
        }
    )
    importance.to_csv(analysis_dir / "importance_bootstrap.csv", index=False)
    pd.DataFrame(
        {
            "feature_set": ["all", "without_r2_linear", "without_predictability_proxies", "chaos_nonlinearity_complexity_only"],
            "regression_r2_mean": [0.61, 0.57, 0.55, 0.45],
            "classification_roc_auc_mean": [0.82, 0.80, 0.79, 0.73],
        }
    ).to_csv(analysis_dir / "robustness_summary.csv", index=False)
    family_ci = pd.concat(
        [
            pd.DataFrame({"family": ["overall"], "n": [len(atlas)], "mean_advantage": [atlas["qrc_advantage"].mean()], "mean_ci_low": [-0.1], "mean_ci_high": [0.08]}),
            family.assign(mean_advantage=family["mean_qrc_advantage"], mean_ci_low=family["mean_qrc_advantage"] - 0.04, mean_ci_high=family["mean_qrc_advantage"] + 0.04)[["family", "n", "mean_advantage", "mean_ci_low", "mean_ci_high"]],
        ],
        ignore_index=True,
    )
    family_ci.to_csv(analysis_dir / "family_advantage_bootstrap.csv", index=False)

    sweep = atlas[["dataset_id", "family"]].copy()
    for j, name in enumerate(feature_names):
        sweep[name] = np.linspace(0.05 + j * 0.01, 0.95 - j * 0.01, len(sweep))
    sweep.to_csv(root / "sweep_catalog.csv", index=False)

    extended = atlas[["dataset_id", "name", "family"]].copy()
    extended["ext_sample_entropy_m2"] = np.linspace(0.1, 0.9, len(extended))
    extended["ext_lz_complexity"] = np.linspace(0.9, 0.1, len(extended))
    extended["ext_turning_point_rate"] = rng.uniform(0.1, 0.8, len(extended))
    extended["ext_zero_fraction"] = rng.uniform(0.0, 0.3, len(extended))
    extended.to_csv(features_dir / "extended_features_sweep.csv", index=False)

    pd.DataFrame(
        {
            "family": ["overall", "chaotic_map", "chaotic_flow"],
            "n": [12, 6, 6],
            "mean_delta_J0_minus_J1": [0.0, -0.03, 0.05],
            "delta_ci_low": [-0.05, -0.08, 0.01],
            "delta_ci_high": [0.05, -0.01, 0.09],
            "mechanism_signal": ["null_or_mixed", "negative", "positive"],
        }
    ).to_csv(attribution_dir / "family_attribution_bootstrap.csv", index=False)
    pd.DataFrame(
        {
            "family": ["chaotic_map", "chaotic_flow"] * 8,
            "nrmse_qrc_J0": np.linspace(0.3, 1.1, 16),
            "nrmse_qrc_J1": np.linspace(0.32, 1.0, 16),
            "paired_delta_J0_minus_J1": np.linspace(-0.08, 0.08, 16),
            "advantage_J1_vs_esn": np.linspace(-0.2, 0.25, 16),
            "advantage_J0_vs_esn": np.linspace(-0.15, 0.2, 16),
        }
    ).to_csv(attribution_dir / "paired_attribution.csv", index=False)

    return atlas_dir, analysis_dir, attribution_dir, features_dir, root / "sweep_catalog.csv"


def test_scientific_plots_cli_writes_dense_publication_figures(tmp_path):
    atlas_dir, analysis_dir, attribution_dir, features_dir, sweep = _write_inputs(tmp_path)
    out_dir = tmp_path / "scientific"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qrc_dataset_profiler.run_scientific_plots",
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
        "fig1_dense_atlas_summary.png",
        "fig2_dense_feature_regressions.png",
        "fig3_dense_family_feature_matrix.png",
        "fig4_dense_model_validation.png",
        "fig5_dense_attribution_guardrail.png",
        "SCIENTIFIC_PLOTS_REPORT.md",
        "scientific_plots_manifest.json",
        "index.html",
    }
    assert expected <= {p.name for p in out_dir.iterdir()}
    manifest = json.loads((out_dir / "scientific_plots_manifest.json").read_text())
    assert manifest["n_rows"] == 32
    assert len(manifest["figures"]) == 5
    assert "do not claim broad average" in manifest["claim_boundary"]
