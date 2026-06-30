"""Command line runner for corrected paired quantum attribution."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / "qrc_dataset_profiler_mpl"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")

from qrc_dataset_profiler.quantum_attribution import DEFAULT_ATTRIBUTION_FAMILIES, run_quantum_attribution_from_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run corrected paired J=1 vs J=0 attribution.")
    parser.add_argument("--catalog", default="results_sweep/sweep_catalog.csv", help="Input sweep catalog CSV or parquet file.")
    parser.add_argument("--out", default="results_quantum_attribution", help="Output attribution directory.")
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_ATTRIBUTION_FAMILIES),
        help="Comma-separated families to rerun; use 'all' for every family.",
    )
    parser.add_argument("--seeds", type=int, default=1, help="Number of matched QRC seeds.")
    parser.add_argument("--n-qubits", type=int, default=6, help="Qubit count, matching fast sweep by default.")
    parser.add_argument("--depth", type=int, default=5, help="Reservoir Trotter depth.")
    parser.add_argument("--virtual-nodes", type=int, default=5, help="Number of virtual nodes.")
    parser.add_argument("--amplitude-damping", type=float, default=0.02, help="Fixed local amplitude-damping probability per layer.")
    parser.add_argument("--dephasing", type=float, default=0.01, help="Fixed local dephasing probability per layer.")
    parser.add_argument("--dissipation-method", default="trajectory", choices=("trajectory", "density"), help="Dissipative simulation method.")
    parser.add_argument("--calibration-config", default=None, help="Frozen standard_v3 calibration JSON; overrides QRC reservoir settings for the coupled branch.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000, help="Bootstrap replicates for paired-effect intervals.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic bootstrap seed.")
    args = parser.parse_args(argv)

    families = () if args.families.strip().lower() == "all" else tuple(f.strip() for f in args.families.split(",") if f.strip())
    manifest = run_quantum_attribution_from_path(
        Path(args.catalog),
        out_dir=Path(args.out),
        families=families,
        seeds=args.seeds,
        n_qubits=args.n_qubits,
        depth=args.depth,
        virtual_nodes=args.virtual_nodes,
        amplitude_damping=args.amplitude_damping,
        dephasing=args.dephasing,
        dissipation_method=args.dissipation_method,
        calibration_config=Path(args.calibration_config) if args.calibration_config else None,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(f"wrote={args.out}")
    print(f"rows_written={manifest['rows_written']}")
    print(f"families={','.join(manifest['families'])}")
    print(f"feature_dim_J1={manifest['reservoir']['feature_dim_J1']}")
    print(f"feature_dim_J0={manifest['reservoir']['feature_dim_J0']}")
    print("claim_boundary=" + manifest["claim_boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
