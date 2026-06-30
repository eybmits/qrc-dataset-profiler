"""CLI for the v5 publication-facing analysis package."""

from __future__ import annotations

import argparse
from pathlib import Path

from qrc_dataset_profiler.v5_publication import run_v5_publication_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build v5 paper-facing tables, plots, and HTML report.")
    parser.add_argument("--discovery-table", default="results_frontier_v5_discovery/frontier_discovery_evaluated_v5_multi_qrc.csv")
    parser.add_argument("--validation-table", default="results_frontier_v5_validation/frontier_validation_evaluated_v5_multi_qrc.csv")
    parser.add_argument("--out", default="results_v5_publication")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--win-threshold", type=float, default=0.05)
    parser.add_argument("--family-bootstraps", type=int, default=1000)
    parser.add_argument("--importance-bootstraps", type=int, default=80)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args(argv)

    manifest = run_v5_publication_analysis(
        discovery_table=Path(args.discovery_table),
        validation_table=Path(args.validation_table),
        out_dir=Path(args.out),
        seed=args.seed,
        win_threshold=args.win_threshold,
        family_bootstraps=args.family_bootstraps,
        importance_bootstraps=args.importance_bootstraps,
        formats=args.formats,
    )
    print(f"wrote={Path(args.out) / 'index.html'}")
    print(f"n_discovery={manifest['n_discovery']}")
    print(f"n_validation={manifest['n_validation']}")
    print(f"n_features_available={manifest['n_features_available']}")
    metrics = manifest["prospective_validation_metrics"]
    print(f"validation_r2={metrics['regression_r2']:.6g}")
    print(f"validation_roc_auc={metrics['classification_roc_auc']:.6g}")
    print(f"validation_pr_auc={metrics['classification_pr_auc']:.6g}")
    print("claim_boundary=" + manifest["claim_boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
