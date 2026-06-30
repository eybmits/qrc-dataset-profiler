import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from qrc_dataset_profiler.analysis import (
    bootstrap_family_advantage,
    feature_sets,
    run_anti_circularity_suite,
    summarize_sweep,
)


def _analysis_catalog(n=72, seed=0):
    rng = np.random.default_rng(seed)
    families = np.array(["chaotic_map", "input_driven", "oscillatory"])
    df = pd.DataFrame(
        {
            "name": [f"ds_{i}" for i in range(n)],
            "family": families[np.arange(n) % len(families)],
            "ac_timescale": rng.normal(size=n),
            "ami_first_min": rng.normal(size=n),
            "mem_capacity": rng.normal(size=n),
            "r2_linear": rng.normal(size=n),
            "nl_gain": rng.normal(size=n),
            "snr_db": rng.normal(size=n),
            "lyapunov": rng.normal(size=n),
            "zero_one_K": rng.normal(size=n),
            "spectral_entropy": rng.normal(size=n),
            "dom_freq": rng.normal(size=n),
            "spectral_flatness": rng.normal(size=n),
            "adf_p": rng.uniform(size=n),
            "kpss_p": rng.uniform(size=n),
            "n_diffs": rng.integers(0, 3, size=n),
            "dfa_alpha": rng.normal(size=n),
            "perm_entropy": rng.normal(size=n),
            "forecastability": rng.normal(size=n),
            "pred_nrmse_gbm": rng.normal(size=n),
        }
    )
    df["qrc_advantage"] = 0.45 * df["nl_gain"] - 0.3 * df["r2_linear"] + rng.normal(scale=0.05, size=n)
    df["nrmse_esn_matched"] = 1.0 + df["qrc_advantage"]
    df["nrmse_qrc_spin"] = 1.0
    return df


def test_feature_sets_encode_anti_circularity_contract():
    sets = {fs.name: fs for fs in feature_sets(_analysis_catalog())}

    assert "r2_linear" in sets["all"].features
    assert "r2_linear" not in sets["without_r2_linear"].features
    for col in ("r2_linear", "forecastability", "pred_nrmse_gbm"):
        assert col not in sets["without_predictability_proxies"].features
    assert set(sets["chaos_nonlinearity_complexity_only"].features) <= {
        "nl_gain",
        "lyapunov",
        "zero_one_K",
        "spectral_entropy",
        "spectral_flatness",
        "dfa_alpha",
        "perm_entropy",
    }


def test_summary_and_bootstrap_are_deterministic():
    df = _analysis_catalog()
    summary = summarize_sweep(df, win_threshold=0.05)
    ci1 = bootstrap_family_advantage(df, n_bootstraps=20, rng=np.random.default_rng(4), win_threshold=0.05)
    ci2 = bootstrap_family_advantage(df, n_bootstraps=20, rng=np.random.default_rng(4), win_threshold=0.05)

    assert set(summary["family"]) == {"overall", "chaotic_map", "input_driven", "oscillatory"}
    pd.testing.assert_frame_equal(ci1, ci2)
    assert {"mean_ci_low", "mean_ci_high", "win_rate_ci_low", "win_rate_ci_high"} <= set(ci1.columns)


def test_anti_circularity_suite_reports_all_feature_sets():
    robustness, importances = run_anti_circularity_suite(_analysis_catalog(), seed=3, win_threshold=0.05)

    assert set(robustness["feature_set"]) == {
        "all",
        "without_r2_linear",
        "without_predictability_proxies",
        "chaos_nonlinearity_complexity_only",
    }
    assert set(importances) == set(robustness["feature_set"])
    assert robustness["n_samples"].min() == 72


def test_run_analysis_cli_writes_reviewer_outputs(tmp_path):
    catalog = tmp_path / "catalog.csv"
    out_dir = tmp_path / "analysis"
    _analysis_catalog(n=48, seed=8).to_csv(catalog, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qrc_dataset_profiler.run_analysis",
            "--catalog",
            str(catalog),
            "--out",
            str(out_dir),
            "--family-bootstraps",
            "8",
            "--importance-bootstraps",
            "4",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    expected = {
        "analysis_manifest.json",
        "sweep_summary.csv",
        "family_advantage_bootstrap.csv",
        "robustness_summary.csv",
        "importance_bootstrap.csv",
        "sweep_summary.png",
        "family_advantage_bootstrap.png",
        "robustness_summary.png",
        "importance_bootstrap.png",
    }
    assert expected <= {p.name for p in out_dir.iterdir()}
    manifest = json.loads((out_dir / "analysis_manifest.json").read_text())
    assert manifest["n_rows"] == 48
    assert "does not establish coupling" in manifest["claim_boundary"]
