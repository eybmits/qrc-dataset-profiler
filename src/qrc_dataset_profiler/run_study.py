"""Command line runner for Increment 2 Block E targets."""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / "qrc_dataset_profiler_mpl"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from qrc_dataset_profiler.baselines import esn_matched_baseline, gbm_baseline, linear_baseline, qrc_nrmse
from qrc_dataset_profiler.generators import ALL_SPECS, generate
from qrc_dataset_profiler.properties import profile_dataset
from qrc_dataset_profiler.reservoir import StandardSpinV1
from qrc_dataset_profiler.spec import Dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the qrc_dataset_profiler Increment-2 study catalog.")
    parser.add_argument("--smoke", action="store_true", help="Use a short, 6-qubit study for quick validation.")
    parser.add_argument("--out", default="results", help="Output directory.")
    parser.add_argument("--seeds", type=int, default=None, help="Number of ESN/QRC seeds; default is 1 for smoke, 3 otherwise.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forecast horizon override; default is round(ac_timescale) per forecast dataset. Input-driven tasks use horizon 1.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_count = int(args.seeds if args.seeds is not None else (1 if args.smoke else 3))
    seed_values = list(range(seed_count))
    n_qubits = 6 if args.smoke else 8
    qrc_cfg = StandardSpinV1(n_qubits=n_qubits)

    rows: list[dict[str, object]] = []
    for base_spec in ALL_SPECS:
        spec = _study_spec(base_spec, smoke=args.smoke)
        ds = generate(spec)
        if ds.ground_truth.get("_unavailable"):
            warnings.warn(f"skipping unavailable dataset {spec.name}", RuntimeWarning)
            continue

        rec = profile_dataset(ds)
        horizon = _study_horizon(spec, rec.ac_timescale, override=args.horizon)
        if horizon != spec.horizon:
            spec = replace(spec, horizon=horizon)
            ds = Dataset(spec, ds.series, inputs=ds.inputs, ground_truth=ds.ground_truth)
        rec.horizon = horizon
        rec.nrmse_linear = linear_baseline(ds)
        rec.nrmse_gbm = gbm_baseline(ds, seed=spec.seed)
        rec.nrmse_esn_matched = float(np.mean([esn_matched_baseline(ds, qrc_cfg=qrc_cfg, seed=s) for s in seed_values]))
        rec.nrmse_qrc_spin = float(np.mean([qrc_nrmse(ds, qrc_cfg, seed=s) for s in seed_values]))
        rec.qrc_advantage = rec.nrmse_esn_matched - rec.nrmse_qrc_spin
        rows.append(rec.to_row())

    df = pd.DataFrame(rows)
    path = out_dir / ("full_catalog.parquet" if _has_pyarrow() else "full_catalog.csv")
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    cols = ["name", "nrmse_linear", "nrmse_esn_matched", "nrmse_qrc_spin", "qrc_advantage"]
    print(df[cols].to_string(index=False, max_cols=len(cols)))
    print(f"\nfeature_dim={qrc_cfg.feature_dim} n_qubits={qrc_cfg.n_qubits} seeds={seed_count} wrote={path}")
    return 0


def _study_spec(base_spec, *, smoke: bool):
    if not smoke or base_spec.source == "real":
        return base_spec
    gt_full_length_keys = ("true_lyapunov", "true_hurst", "true_n_frequencies", "is_chaotic")
    probe = generate(replace(base_spec, length=min(base_spec.length, 32)))
    if any(key in probe.ground_truth for key in gt_full_length_keys):
        return base_spec
    return replace(base_spec, length=min(base_spec.length, 2000))


def _study_horizon(spec, ac_timescale: float, *, override: int | None) -> int:
    if spec.task_type != "forecast":
        return 1
    if override is not None:
        return max(1, int(override))
    if np.isfinite(ac_timescale):
        return max(1, int(round(float(ac_timescale))))
    return max(1, int(spec.horizon))


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
