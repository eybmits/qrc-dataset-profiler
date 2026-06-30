"""Command line runner for publication-facing atlas figures and report."""

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

from qrc_dataset_profiler.publication import run_publication_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build publication-facing QRC atlas figures and report.")
    parser.add_argument("--atlas-dir", default="results_atlas", help="Directory containing atlas CSV outputs.")
    parser.add_argument("--analysis-dir", default="results_analysis", help="Directory containing analysis CSV outputs.")
    parser.add_argument("--attribution-dir", default="results_quantum_attribution", help="Directory containing attribution CSV outputs.")
    parser.add_argument("--out", default="results_publication", help="Output publication package directory.")
    args = parser.parse_args(argv)

    manifest = run_publication_package(
        atlas_dir=Path(args.atlas_dir),
        analysis_dir=Path(args.analysis_dir),
        attribution_dir=Path(args.attribution_dir),
        out_dir=Path(args.out),
    )
    print(f"wrote={args.out}")
    print(f"n_rows={manifest['n_rows']}")
    print(f"n_families={manifest['n_families']}")
    print("claim_boundary=" + manifest["claim_boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
