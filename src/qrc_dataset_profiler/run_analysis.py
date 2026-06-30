"""Command line runner for the Increment-4 reproducible analysis suite."""

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

from qrc_dataset_profiler.analysis import load_catalog, run_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Increment-4 paper-grade sweep analysis.")
    parser.add_argument("--catalog", default="results_sweep/sweep_catalog.csv", help="Input sweep catalog CSV or parquet file.")
    parser.add_argument("--out", default="results_analysis", help="Output analysis directory.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for bootstraps and meta-models.")
    parser.add_argument("--family-bootstraps", type=int, default=1000, help="Bootstrap replicates for family advantage intervals.")
    parser.add_argument("--importance-bootstraps", type=int, default=100, help="Bootstrap replicates for all-feature importances.")
    parser.add_argument("--win-threshold", type=float, default=0.05, help="Threshold for QRC-win classification.")
    args = parser.parse_args(argv)

    manifest = run_analysis(
        load_catalog(Path(args.catalog)),
        out_dir=Path(args.out),
        seed=args.seed,
        family_bootstraps=args.family_bootstraps,
        importance_bootstraps=args.importance_bootstraps,
        win_threshold=args.win_threshold,
    )
    print(f"wrote={args.out}")
    print(f"n_rows={manifest['n_rows']}")
    print(f"family_bootstraps={manifest['family_bootstraps']}")
    print(f"importance_bootstraps={manifest['importance_bootstraps']}")
    print("claim_boundary=" + manifest["claim_boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
