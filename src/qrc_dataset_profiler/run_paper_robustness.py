"""CLI for the paper robustness and external-probe suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from qrc_dataset_profiler.paper_robustness import run_paper_robustness_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paper-facing robustness checks for the v4 frontier atlas.")
    parser.add_argument("--discovery-table", default="results_frontier_v4_discovery/frontier_discovery_evaluated_30_features.csv")
    parser.add_argument("--validation-table", default="results_frontier_v4_validation/frontier_validation_evaluated_30_features.csv")
    parser.add_argument("--out", default="results_frontier_v4_paper_robustness")
    parser.add_argument("--calibration-config", default="results_calibration_v4/frozen_config.json")
    parser.add_argument("--thresholds", default="0,0.025,0.05,0.1", help="Comma-separated useful thresholds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric-subset-n", type=int, default=120, help="Property-defined subset rows for NMAE/NVAR reruns; 0 disables.")
    parser.add_argument("--mechanism-rows", type=int, default=60, help="Property-defined subset rows for paired J=0/J* guardrail; 0 disables.")
    parser.add_argument("--mechanism-seeds", type=int, default=1)
    parser.add_argument("--real-probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--formats", default="png", help="Comma-separated figure formats.")
    args = parser.parse_args(argv)

    thresholds = tuple(float(v.strip()) for v in args.thresholds.split(",") if v.strip())
    calibration_config = Path(args.calibration_config) if args.calibration_config else None
    manifest = run_paper_robustness_suite(
        discovery_table=Path(args.discovery_table),
        validation_table=Path(args.validation_table),
        out_dir=Path(args.out),
        calibration_config=calibration_config,
        thresholds=thresholds,
        seed=args.seed,
        metric_subset_n=args.metric_subset_n,
        mechanism_rows=args.mechanism_rows,
        mechanism_seeds=args.mechanism_seeds,
        real_probes=args.real_probes,
        formats=tuple(v.strip() for v in args.formats.split(",") if v.strip()),
    )
    print(f"wrote={args.out}")
    print(f"n_discovery={manifest['n_discovery']} n_validation={manifest['n_validation']}")
    print(f"real_probes={manifest['real_probes_written']}")
    print(f"metric_subset_rows={manifest['metric_subset_n_written']}")
    print(f"mechanism_rows={manifest['mechanism_rows_written']}")
    print("claim_boundary=" + manifest["claim_boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

