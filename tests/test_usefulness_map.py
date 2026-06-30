import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from qrc_dataset_profiler.usefulness_map import build_row_map, run_usefulness_map, summarize_families
from qrc_dataset_profiler.meta_model import fit_meta_model


def _catalog(n=72, seed=0):
    rng = np.random.default_rng(seed)
    families = np.array(["chaotic_map", "input_driven", "oscillatory"])
    nl_gain = rng.normal(size=n)
    r2_linear = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "name": [f"ds_{i}" for i in range(n)],
            "family": families[np.arange(n) % len(families)],
            "source": "synthetic",
            "task_type": "forecast",
            "seed": np.arange(n),
            "length": 400,
            "horizon": 1,
            "ac_timescale": rng.normal(size=n),
            "ami_first_min": rng.normal(size=n),
            "mem_capacity": rng.normal(size=n),
            "r2_linear": r2_linear,
            "nl_gain": nl_gain,
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
    df["qrc_advantage"] = 0.55 * nl_gain - 0.35 * r2_linear + rng.normal(scale=0.05, size=n)
    df["nrmse_esn_matched"] = 1.0 + df["qrc_advantage"]
    df["nrmse_qrc_spin"] = 1.0
    return df


def test_row_map_contains_coordinates_predictions_and_labels():
    df = _catalog()
    result = fit_meta_model(df, seed=1)
    row_map = build_row_map(df, result, win_threshold=0.05, tie_margin=0.05)

    assert len(row_map) == len(df)
    assert {"property_pc1", "property_pc2", "predicted_qrc_advantage", "actual_usefulness_label"} <= set(row_map.columns)
    assert set(row_map["actual_usefulness_label"]) <= {"qrc_useful", "near_tie", "baseline_preferred"}
    assert np.isfinite(row_map["property_pc1"]).all()


def test_family_summary_rates_sum_to_one():
    df = _catalog()
    result = fit_meta_model(df, seed=1)
    summary = summarize_families(build_row_map(df, result, win_threshold=0.05, tie_margin=0.05))

    rates = summary[["qrc_useful_rate", "near_tie_rate", "baseline_preferred_rate"]].sum(axis=1)
    assert np.allclose(rates, 1.0)
    assert set(summary["dominant_category"]) <= {"qrc_useful", "near_tie", "baseline_preferred"}


def test_run_usefulness_map_cli_writes_atlas_outputs(tmp_path):
    catalog = tmp_path / "catalog.csv"
    out_dir = tmp_path / "atlas"
    _catalog(n=48, seed=5).to_csv(catalog, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qrc_dataset_profiler.run_usefulness_map",
            "--catalog",
            str(catalog),
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
        "qrc_usefulness_map.csv",
        "family_usefulness_summary.csv",
        "meta_model_summary.csv",
        "atlas_importances.csv",
        "property_usefulness_map.png",
        "predicted_vs_actual_usefulness.png",
        "family_usefulness_categories.png",
        "atlas_importances.png",
        "usefulness_atlas_manifest.json",
    }
    assert expected <= {p.name for p in out_dir.iterdir()}
    manifest = json.loads((out_dir / "usefulness_atlas_manifest.json").read_text())
    assert manifest["n_rows"] == 48
    assert "not a fundamental quantum-advantage claim" in manifest["claim_boundary"]
