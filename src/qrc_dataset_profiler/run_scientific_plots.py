"""Command line runner for dense publication-style QRC atlas figures."""

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

from qrc_dataset_profiler.scientific_plots import run_scientific_plots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build dense scientific QRC atlas figures.")
    parser.add_argument("--atlas-dir", default="results_atlas", help="Directory containing atlas outputs.")
    parser.add_argument("--analysis-dir", default="results_analysis", help="Directory containing analysis outputs.")
    parser.add_argument("--attribution-dir", default="results_quantum_attribution", help="Directory containing attribution outputs.")
    parser.add_argument("--features-dir", default="results_features", help="Directory containing extended features.")
    parser.add_argument("--sweep-catalog", default="results_sweep/sweep_catalog.csv", help="Sweep catalog CSV.")
    parser.add_argument("--out", default="results_scientific_plots", help="Output directory.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats: png,pdf,svg.")
    args = parser.parse_args(argv)
    manifest = run_scientific_plots(
        atlas_dir=Path(args.atlas_dir),
        analysis_dir=Path(args.analysis_dir),
        attribution_dir=Path(args.attribution_dir),
        features_dir=Path(args.features_dir),
        sweep_catalog=Path(args.sweep_catalog),
        out_dir=Path(args.out),
        formats=args.formats.split(","),
    )
    print(f"wrote={args.out}")
    print(f"n_rows={manifest['n_rows']}")
    print(f"figures={len(manifest['figures'])}")
    print("claim_boundary=" + manifest["claim_boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
