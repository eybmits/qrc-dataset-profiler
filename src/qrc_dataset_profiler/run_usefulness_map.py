"""Command line runner for the QRC usefulness atlas."""

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

from qrc_dataset_profiler.usefulness_map import run_usefulness_map_from_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the QRC dataset usefulness atlas.")
    parser.add_argument("--catalog", default="results_sweep/sweep_catalog.csv", help="Input sweep catalog CSV or parquet file.")
    parser.add_argument("--out", default="results_atlas", help="Output atlas directory.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic meta-model seed.")
    parser.add_argument("--win-threshold", type=float, default=0.05, help="Advantage threshold for qrc_useful label.")
    parser.add_argument("--tie-margin", type=float, default=0.05, help="Near-tie margin below zero advantage.")
    args = parser.parse_args(argv)

    manifest = run_usefulness_map_from_path(
        Path(args.catalog),
        out_dir=Path(args.out),
        seed=args.seed,
        win_threshold=args.win_threshold,
        tie_margin=args.tie_margin,
    )
    print(f"wrote={args.out}")
    print(f"n_rows={manifest['n_rows']}")
    print(f"features_used={len(manifest['features_used'])}")
    print("claim_boundary=" + manifest["claim_boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
