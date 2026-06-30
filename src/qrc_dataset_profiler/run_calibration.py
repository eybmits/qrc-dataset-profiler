"""Command line runner for held-out global reservoir calibration."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / "qrc_dataset_profiler_mpl"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

from qrc_dataset_profiler.calibration import run_global_calibration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate global fixed QRC and ESN reservoirs on held-out synthetic datasets.")
    parser.add_argument("--out", default="results_calibration_v3", help="Output calibration directory.")
    parser.add_argument("--sweep-seed", type=int, default=917, help="Held-out synthetic sweep seed for calibration data.")
    parser.add_argument("--sweep-n-per-family", type=int, default=2, help="Pool density passed to make_sweep_specs for calibration.")
    parser.add_argument("--rows-per-family", type=int, default=3, help="Balanced calibration rows selected per family.")
    parser.add_argument("--fast", action="store_true", help="Use scalable 6-qubit/length-capped calibration mode.")
    parser.add_argument("--seeds", type=int, default=1, help="Reservoir seeds averaged per candidate during calibration.")
    parser.add_argument("--n-qubits", type=int, default=None, help="Override QRC qubit count; default 6 with --fast, else 8.")
    parser.add_argument("--n-qubits-options", default=None, help="Comma-separated QRC qubit counts for a multi-size calibration grid.")
    parser.add_argument("--depth-options", default=None, help="Comma-separated QRC depths for a multi-depth calibration grid.")
    parser.add_argument("--virtual-nodes-options", default=None, help="Comma-separated QRC virtual-node counts for calibration.")
    parser.add_argument("--taxonomy", choices=("v3", "v4"), default="v3", help="Synthetic taxonomy for held-out calibration rows.")
    parser.add_argument("--small-grid", action="store_true", help="Use a one-point smoke grid; not for final calibration.")
    parser.add_argument("--v4-protocol-grid", action="store_true", help="Use the paper-facing v4 QRC/ESN calibration grid.")
    parser.add_argument("--selection-tolerance", type=float, default=0.0, help="Validation-NRMSE tolerance for choosing the smallest QRC among near-ties.")
    args = parser.parse_args(argv)
    qrc_grid = None
    esn_grid = None
    n_qubits_options = _parse_int_list(args.n_qubits_options)
    depth_options = _parse_int_list(args.depth_options)
    virtual_nodes_options = _parse_int_list(args.virtual_nodes_options)
    if args.small_grid:
        qrc_grid = {"J": (1.0,), "h": (1.0,), "dt": (0.2,), "amplitude_damping": (0.0,), "dephasing": (0.0,)}
        esn_grid = {"rho": (0.9,), "leak": (0.3,), "input_scale": (1.0,)}
    if args.v4_protocol_grid:
        qrc_grid = {
            "J": (0.8, 1.0, 1.2, 1.5),
            "h": (1.0,),
            "dt": (0.15, 0.20, 0.25),
            "amplitude_damping": (0.0, 0.005),
            "dephasing": (0.0, 0.005),
        }
        esn_grid = {
            "rho": (0.7, 0.9, 1.0, 1.1, 1.3),
            "leak": (0.1, 0.3, 0.6, 1.0),
            "input_scale": (0.3, 1.0, 2.0),
        }
        n_qubits_options = n_qubits_options or (6, 8, 10)
        depth_options = depth_options or (5,)
        virtual_nodes_options = virtual_nodes_options or (5,)

    manifest = run_global_calibration(
        out_dir=Path(args.out),
        sweep_seed=args.sweep_seed,
        sweep_n_per_family=args.sweep_n_per_family,
        calibration_rows_per_family=args.rows_per_family,
        fast=args.fast,
        seeds=args.seeds,
        n_qubits=args.n_qubits,
        n_qubits_options=n_qubits_options,
        depth_options=depth_options,
        virtual_nodes_options=virtual_nodes_options,
        taxonomy=args.taxonomy,
        selection_tolerance=args.selection_tolerance,
        qrc_grid=qrc_grid,
        esn_grid=esn_grid,
    )
    print(f"wrote={args.out}")
    print(f"n_rows={manifest['calibration_data']['n_rows']}")
    print(f"qrc_J={manifest['qrc']['J']} qrc_dt={manifest['qrc']['dt']}")
    print(f"qrc_amp={manifest['qrc']['amplitude_damping']} qrc_dephasing={manifest['qrc']['dephasing']}")
    print(f"esn_rho={manifest['esn']['rho']} esn_leak={manifest['esn']['leak']} esn_input_scale={manifest['esn']['input_scale']}")
    print("claim_boundary=" + manifest["claim_boundary"])
    return 0


def _parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None or not str(value).strip():
        return None
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


if __name__ == "__main__":
    raise SystemExit(main())
