"""Command line runner for the QRC atlas visual suite."""

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

from qrc_dataset_profiler.visual_suite import run_visual_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the state-of-the-art QRC usefulness visual suite.")
    parser.add_argument("--atlas-dir", default="results_atlas", help="Directory containing atlas CSV outputs.")
    parser.add_argument("--analysis-dir", default="results_analysis", help="Directory containing analysis CSV outputs.")
    parser.add_argument("--attribution-dir", default="results_quantum_attribution", help="Directory containing attribution CSV outputs.")
    parser.add_argument("--features-dir", default="results_features", help="Directory containing extended feature CSV outputs.")
    parser.add_argument("--sweep-catalog", default="results_sweep/sweep_catalog.csv", help="Sweep catalog CSV for generator barcode labels.")
    parser.add_argument("--full-catalog", default="results_full/full_catalog.csv", help="Full 50-row catalog CSV for benchmark inventory.")
    parser.add_argument("--out", default="results_visuals", help="Output visual suite directory.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats: png,pdf,svg.")
    args = parser.parse_args(argv)

    manifest = run_visual_suite(
        atlas_dir=Path(args.atlas_dir),
        analysis_dir=Path(args.analysis_dir),
        attribution_dir=Path(args.attribution_dir),
        features_dir=Path(args.features_dir),
        sweep_catalog=Path(args.sweep_catalog) if args.sweep_catalog else None,
        full_catalog=Path(args.full_catalog) if args.full_catalog else None,
        out_dir=Path(args.out),
        formats=args.formats.split(","),
    )
    print(f"wrote={args.out}")
    print(f"n_rows={manifest['n_rows']}")
    print(f"n_families={manifest['n_families']}")
    print(f"figures={len(manifest['figures'])}")
    print("claim_boundary=" + manifest["claim_boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
