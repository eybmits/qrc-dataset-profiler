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
from qrc_dataset_profiler.generators import ALL_SPECS, generate, make_sweep_specs
from qrc_dataset_profiler.properties import profile_dataset
from qrc_dataset_profiler.reservoir import StandardSpinV1
from qrc_dataset_profiler.spec import Dataset, DatasetSpec


FAST_ESN_GRID = {"rho": (0.9, 1.1), "leak": (0.3, 0.9), "input_scale": (1.0,)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the qrc_dataset_profiler Increment-2 study catalog.")
    parser.add_argument("--smoke", action="store_true", help="Use a short, 6-qubit study for quick validation.")
    parser.add_argument("--sweep", action="store_true", help="Profile the parameterized sweep catalog instead of ALL_SPECS.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Scalable approximate mode: 6 qubits, one seed, reduced ESN grid "
            "(rho={0.9,1.1}, leak={0.3,0.9}, input_scale={1.0}), and shorter synthetic series. "
            "Faster for sweep planning, less accurate than the full protocol run."
        ),
    )
    parser.add_argument("--sweep-n-per-family", type=int, default=20, help="Variants per swept generator family when --sweep is set.")
    parser.add_argument("--sweep-seed", type=int, default=0, help="Deterministic seed for --sweep catalog construction.")
    parser.add_argument("--out", default="results", help="Output directory.")
    parser.add_argument("--seeds", type=int, default=None, help="Number of ESN/QRC seeds; default is 1 for smoke/fast, 3 otherwise.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forecast horizon override; default is round(ac_timescale) per forecast dataset. Input-driven tasks use horizon 1.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    specs = make_sweep_specs(args.sweep_n_per_family, seed=args.sweep_seed) if args.sweep else ALL_SPECS
    output_stem = "sweep_catalog" if args.sweep else "full_catalog"
    df, path, qrc_cfg, seed_count = build_catalog(
        specs,
        out_dir=out_dir,
        smoke=args.smoke,
        fast=args.fast,
        seeds=args.seeds,
        horizon_override=args.horizon,
        output_stem=output_stem,
    )

    cols = ["name", "nrmse_linear", "nrmse_esn_matched", "nrmse_qrc_spin", "qrc_advantage"]
    print(df[cols].to_string(index=False, max_cols=len(cols)))
    print(f"\nfeature_dim={qrc_cfg.feature_dim} n_qubits={qrc_cfg.n_qubits} seeds={seed_count} wrote={path}")
    return 0


def build_catalog(
    specs: list[DatasetSpec],
    *,
    out_dir: Path,
    smoke: bool = False,
    fast: bool = False,
    seeds: int | None = None,
    horizon_override: int | None = None,
    output_stem: str = "full_catalog",
) -> tuple[pd.DataFrame, Path, StandardSpinV1, int]:
    """Profile specs and write a schema-v1 catalog with Block E targets."""

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_count = int(seeds if seeds is not None else (1 if smoke or fast else 3))
    seed_values = list(range(seed_count))
    qrc_cfg = StandardSpinV1(n_qubits=6 if smoke or fast else 8)
    esn_grid = FAST_ESN_GRID if fast else None

    rows: list[dict[str, object]] = []
    for base_spec in specs:
        spec = _study_spec(base_spec, smoke=smoke, fast=fast)
        ds = generate(spec)
        if ds.ground_truth.get("_unavailable"):
            warnings.warn(f"skipping unavailable dataset {spec.name}", RuntimeWarning)
            continue

        rec = profile_dataset(ds)
        horizon = _study_horizon(spec, rec.ac_timescale, override=horizon_override)
        if horizon != spec.horizon:
            spec = replace(spec, horizon=horizon)
            ds = Dataset(spec, ds.series, inputs=ds.inputs, ground_truth=ds.ground_truth)
        rec.horizon = horizon
        rec.nrmse_linear = linear_baseline(ds)
        rec.nrmse_gbm = gbm_baseline(ds, seed=spec.seed)
        rec.nrmse_esn_matched = float(np.mean([esn_matched_baseline(ds, qrc_cfg=qrc_cfg, seed=s, esn_grid=esn_grid) for s in seed_values]))
        rec.nrmse_qrc_spin = float(np.mean([qrc_nrmse(ds, qrc_cfg, seed=s) for s in seed_values]))
        rec.qrc_advantage = rec.nrmse_esn_matched - rec.nrmse_qrc_spin
        rows.append(rec.to_row())

    df = pd.DataFrame(rows)
    path = out_dir / (f"{output_stem}.parquet" if _has_pyarrow() else f"{output_stem}.csv")
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return df, path, qrc_cfg, seed_count


def _study_spec(base_spec, *, smoke: bool, fast: bool = False):
    if base_spec.source == "real":
        return base_spec
    if fast:
        return replace(base_spec, length=min(base_spec.length, 800))
    if not smoke:
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
