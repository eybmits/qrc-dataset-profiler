import os
import subprocess
import sys

import numpy as np
import pandas as pd

from qrc_dataset_profiler.meta_model import fit_meta_model
from qrc_dataset_profiler.spec import CORE_AXIS_FIELDS


def _synthetic_catalog(n=120, seed=0):
    rng = np.random.default_rng(seed)
    nl_gain = rng.normal(size=n)
    r2_linear = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "nl_gain": nl_gain,
            "r2_linear": r2_linear,
            "ac_timescale": rng.normal(size=n),
            "spectral_entropy": rng.normal(size=n),
            "forecastability": rng.normal(size=n),
            "perm_entropy": rng.normal(size=n),
        }
    )
    for col in CORE_AXIS_FIELDS:
        if col not in df.columns:
            df[col] = rng.normal(size=n)
    noise = rng.normal(scale=0.08, size=n)
    df["qrc_advantage"] = 0.65 * nl_gain - 0.55 * r2_linear + 0.25 * nl_gain * r2_linear + noise
    df["nrmse_esn_matched"] = 1.0 + df["qrc_advantage"]
    df["nrmse_qrc_spin"] = 1.0
    return df


def test_fit_meta_model_recovers_planted_drivers():
    df = _synthetic_catalog()
    result = fit_meta_model(df, seed=123)

    top = set(result.ranked_importances["feature"].head(4))
    assert {"nl_gain", "r2_linear"} <= top
    assert np.isfinite(result.regression_cv["models"]["gradient_boosting"]["r2_mean"])
    assert result.regression_cv["models"]["gradient_boosting"]["r2_mean"] > 0.4
    assert result.n_samples == len(df)


def test_run_meta_cli_writes_outputs(tmp_path):
    df = _synthetic_catalog(n=40, seed=7)
    catalog_path = tmp_path / "catalog.csv"
    out_dir = tmp_path / "meta"
    df.to_csv(catalog_path, index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qrc_dataset_profiler.run_meta",
            "--catalog",
            str(catalog_path),
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
    assert (out_dir / "importances.csv").exists()
    assert (out_dir / "importance_bar.png").exists()
    assert (out_dir / "partial_dependence.png").exists()
    importances = pd.read_csv(out_dir / "importances.csv")
    assert not importances.empty
    assert "n_samples=40" in proc.stdout
