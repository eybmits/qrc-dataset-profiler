"""Command line runner for Increment 2 Block E targets."""

from __future__ import annotations

import argparse
import json
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

from qrc_dataset_profiler.baselines import (
    esn_matched_baseline,
    esn_sparse_baseline,
    gbm_baseline,
    linear_baseline,
    qrc_nrmse,
    qrc_nrmse_standard,
)
from qrc_dataset_profiler.generators import ALL_SPECS, generate, make_sweep_specs
from qrc_dataset_profiler.properties import profile_dataset
from qrc_dataset_profiler.reservoir import StandardSpinV1
from qrc_dataset_profiler.spec import Dataset, DatasetSpec


FAST_ESN_GRID = {"rho": (0.9, 1.1), "leak": (0.3, 0.9), "input_scale": (1.0,)}
FROZEN_STANDARD_ESN_GRID = {"rho": (0.9,), "leak": (0.3,), "input_scale": (1.0,)}
STANDARD_V2_AMPLITUDE_DAMPING = 0.02
STANDARD_V2_DEPHASING = 0.01


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
        "--comparison-protocol",
        choices=("standard_v3", "standard_v2", "legacy_v1"),
        default="standard_v3",
        help=(
            "Frozen comparison protocol. standard_v3 uses fixed QRC and fixed sparse ESN reservoir hyperparameters; "
            "standard_v2 uses fixed QRC against validation-tuned sparse ESN; legacy_v1 reproduces earlier simple-cycle ESN artifacts."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forecast horizon override; default is round(ac_timescale) per forecast dataset. Input-driven tasks use horizon 1.",
    )
    parser.add_argument(
        "--calibration-config",
        default=None,
        help="Frozen global calibration JSON from run_calibration; intended for --comparison-protocol standard_v3.",
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
        comparison_protocol=args.comparison_protocol,
        calibration_config=Path(args.calibration_config) if args.calibration_config else None,
    )

    cols = ["name", "nrmse_linear", "nrmse_esn_matched", "nrmse_qrc_spin", "qrc_advantage"]
    print(df[cols].to_string(index=False, max_cols=len(cols)))
    print(
        f"\ncomparison_protocol={args.comparison_protocol} "
        f"feature_dim={qrc_cfg.feature_dim} n_qubits={qrc_cfg.n_qubits} "
        f"seeds={seed_count} wrote={path}"
    )
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
    comparison_protocol: str = "standard_v3",
    calibration_config: Path | str | None = None,
) -> tuple[pd.DataFrame, Path, StandardSpinV1, int]:
    """Profile specs and write a schema-v1 catalog with Block E targets."""

    if comparison_protocol not in {"standard_v3", "standard_v2", "legacy_v1"}:
        raise ValueError("comparison_protocol must be 'standard_v3', 'standard_v2', or 'legacy_v1'")
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_manifest = _load_calibration_config(calibration_config) if calibration_config is not None else None
    if frozen_manifest is not None and comparison_protocol != "standard_v3":
        raise ValueError("calibration_config is only supported for comparison_protocol='standard_v3'")
    seed_count = int(seeds if seeds is not None else (1 if smoke or fast else 3))
    seed_values = list(range(seed_count))
    if comparison_protocol == "standard_v3" and frozen_manifest is not None:
        qrc_cfg = _qrc_from_calibration_config(frozen_manifest)
    elif comparison_protocol in {"standard_v3", "standard_v2"}:
        qrc_cfg = StandardSpinV1(
            n_qubits=6 if smoke or fast else 8,
            reupload=False,
            amplitude_damping=STANDARD_V2_AMPLITUDE_DAMPING,
            dephasing=STANDARD_V2_DEPHASING,
            dissipation_method="trajectory",
        )
    else:
        qrc_cfg = StandardSpinV1(n_qubits=6 if smoke or fast else 8, reupload=True)
    if comparison_protocol == "standard_v3":
        esn_grid = _esn_grid_from_calibration_config(frozen_manifest) if frozen_manifest is not None else FROZEN_STANDARD_ESN_GRID
    elif comparison_protocol == "standard_v2":
        esn_grid = FAST_ESN_GRID if fast else None
    else:
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
        if comparison_protocol in {"standard_v3", "standard_v2"}:
            rec.nrmse_esn_matched = float(np.mean([esn_sparse_baseline(ds, qrc_cfg=qrc_cfg, seed=s, esn_grid=esn_grid) for s in seed_values]))
            rec.nrmse_qrc_spin = float(np.mean([qrc_nrmse_standard(ds, qrc_cfg, seed=s) for s in seed_values]))
        else:
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
    _write_catalog_manifest(
        out_dir,
        output_stem,
        comparison_protocol,
        qrc_cfg,
        seed_count,
        fast=fast,
        smoke=smoke,
        esn_grid=esn_grid,
        calibration_config=calibration_config,
    )
    return df, path, qrc_cfg, seed_count


def _write_catalog_manifest(
    out_dir: Path,
    output_stem: str,
    comparison_protocol: str,
    qrc_cfg: StandardSpinV1,
    seed_count: int,
    *,
    fast: bool,
    smoke: bool,
    esn_grid: dict[str, tuple[float, ...]] | None = None,
    calibration_config: Path | str | None = None,
) -> None:
    if comparison_protocol == "standard_v3":
        primary_esn = "frozen_sparse_random_leaky_esn"
        esn_selection = "selected_once_on_held_out_calibration_set_then_frozen" if calibration_config is not None else "none_reservoir_hyperparameters_frozen_globally"
        esn_grid = esn_grid or FROZEN_STANDARD_ESN_GRID
        qrc_encoding = "train_split_scaled_input_qubit_injection_no_rz_reupload_with_fixed_local_dissipation"
    elif comparison_protocol == "standard_v2":
        primary_esn = "validation_tuned_sparse_random_leaky_esn"
        esn_selection = "per_dataset_validation_selection_over_fixed_grid"
        esn_grid = FAST_ESN_GRID if fast else {
            "rho": (0.7, 0.9, 1.0, 1.1, 1.3),
            "leak": (0.1, 0.3, 0.6, 1.0),
            "input_scale": (0.3, 1.0, 2.0),
        }
        qrc_encoding = "train_split_scaled_input_qubit_injection_no_rz_reupload_with_fixed_local_dissipation"
    else:
        primary_esn = "simple_cycle_leaky_esn"
        esn_selection = "per_dataset_validation_selection_over_fixed_grid"
        esn_grid = FAST_ESN_GRID if fast else {
            "rho": (0.7, 0.9, 1.0, 1.1, 1.3),
            "leak": (0.1, 0.3, 0.6, 1.0),
            "input_scale": (0.3, 1.0, 2.0),
        }
        qrc_encoding = "full_sequence_scaled_input_injection_plus_rz_reupload"
    manifest = {
        "comparison_protocol": comparison_protocol,
        "output_stem": output_stem,
        "qrc": {
            "class": "StandardSpinV1",
            "n_qubits": qrc_cfg.n_qubits,
            "J": qrc_cfg.J,
            "h": qrc_cfg.h,
            "dt": qrc_cfg.dt,
            "depth": qrc_cfg.depth,
            "topology": qrc_cfg.topology,
            "virtual_nodes": qrc_cfg.virtual_nodes,
            "reupload": qrc_cfg.reupload,
            "amplitude_damping": qrc_cfg.amplitude_damping,
            "dephasing": qrc_cfg.dephasing,
            "dissipation_method": qrc_cfg.dissipation_method,
            "coupling_mode": qrc_cfg.coupling_mode,
            "coupling_seed": qrc_cfg.coupling_seed,
            "feature_dim": qrc_cfg.feature_dim,
            "encoding": qrc_encoding,
            "hyperparameter_selection": "selected_once_on_held_out_calibration_set_then_frozen"
            if calibration_config is not None and comparison_protocol == "standard_v3"
            else "fixed_protocol_default",
        },
        "primary_esn": primary_esn,
        "esn": {
            "class": primary_esn,
            "reservoir_size": qrc_cfg.feature_dim,
            "density": 0.1 if comparison_protocol in {"standard_v3", "standard_v2"} else None,
            "bias_scale": 0.2 if comparison_protocol in {"standard_v3", "standard_v2"} else None,
            "hyperparameter_selection": esn_selection,
            "grid": {key: list(value) for key, value in esn_grid.items()},
            "readout": "ridge alpha selected on validation split; same protocol as QRC",
        },
        "calibration_config": str(calibration_config) if calibration_config is not None else None,
        "seed_count": seed_count,
        "fast": bool(fast),
        "smoke": bool(smoke),
        "claim_boundary": "Dataset categorization only; not a broad quantum-advantage or mechanism claim.",
    }
    (out_dir / f"{output_stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _load_calibration_config(path: Path | str | None) -> dict[str, object] | None:
    if path is None:
        return None
    cfg_path = Path(path)
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if data.get("comparison_protocol") != "standard_v3":
        raise ValueError("calibration config must have comparison_protocol='standard_v3'")
    return data


def _qrc_from_calibration_config(manifest: dict[str, object]) -> StandardSpinV1:
    qrc = dict(manifest.get("qrc", {}))
    allowed = {
        "n_qubits",
        "J",
        "h",
        "dt",
        "depth",
        "topology",
        "virtual_nodes",
        "reupload",
        "amplitude_damping",
        "dephasing",
        "dissipation_method",
        "coupling_mode",
        "coupling_seed",
        "shots",
        "seed",
    }
    kwargs = {key: qrc[key] for key in allowed if key in qrc}
    return StandardSpinV1(**kwargs)


def _esn_grid_from_calibration_config(manifest: dict[str, object] | None) -> dict[str, tuple[float, ...]]:
    if manifest is None:
        return FROZEN_STANDARD_ESN_GRID
    esn = dict(manifest.get("esn", {}))
    return {
        "rho": (float(esn["rho"]),),
        "leak": (float(esn["leak"]),),
        "input_scale": (float(esn["input_scale"]),),
    }


def _study_spec(base_spec, *, smoke: bool, fast: bool = False):
    if fast:
        return replace(base_spec, length=min(base_spec.length, 800))
    if base_spec.source == "real":
        return base_spec
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
