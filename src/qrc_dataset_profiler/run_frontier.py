"""CLI for the frontier conditional-QRC-advantage atlas."""

from __future__ import annotations

import argparse
from pathlib import Path

from qrc_dataset_profiler.frontier import (
    DEFAULT_DISCOVERY_EVALUATED_ROWS,
    DEFAULT_PROPERTY_N_PER_TEMPLATE,
    DEFAULT_VALIDATION_EVALUATED_ROWS,
    write_evaluation_selection,
    write_evaluated_selection,
    write_frontier_30_table,
    write_frontier_regime_analysis,
    write_property_atlas,
    write_support_scores,
)
from qrc_dataset_profiler.frontier_plots import run_frontier_publication_plots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build frontier atlas artifacts for the QRC regime map.")
    sub = parser.add_subparsers(dest="command", required=True)

    prop = sub.add_parser("property-atlas", help="Generate the 20k property-only synthetic candidate atlas.")
    prop.add_argument("--out", default="results_frontier_property", help="Output directory.")
    prop.add_argument("--n-per-template", type=int, default=DEFAULT_PROPERTY_N_PER_TEMPLATE, help="Sweep density; 400 gives 20000 rows.")
    prop.add_argument("--seed", type=int, default=0, help="Synthetic sweep seed.")
    prop.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True, help="Use fast series lengths; default true.")
    prop.add_argument("--smoke", action="store_true", help="Use smoke-compatible lengths.")
    prop.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke/debug runs.")
    prop.add_argument("--taxonomy", choices=("v3", "v4"), default="v3", help="Synthetic taxonomy to generate. v4 uses the 16-family NPJ protocol.")
    prop.add_argument("--checkpoint-every", type=int, default=0, help="Write/resume property-atlas partial CSV every N processed specs.")

    join = sub.add_parser("join-30", help="Join existing evaluated sweep + extended features into the 30-feature table.")
    join.add_argument("--catalog", required=True, help="Sweep/full catalog CSV/parquet.")
    join.add_argument("--extended-features", default=None, help="Extended feature CSV/parquet.")
    join.add_argument("--out", default="results_frontier_features", help="Output directory.")

    select = sub.add_parser("select", help="Freeze the target-free discovery/validation evaluation selection.")
    select.add_argument("--property-atlas", required=True, help="Property atlas CSV/parquet.")
    select.add_argument("--out", default="results_frontier_selection", help="Output directory.")
    select.add_argument("--n-discovery", type=int, default=DEFAULT_DISCOVERY_EVALUATED_ROWS, help="Discovery labels to select.")
    select.add_argument("--n-validation", type=int, default=DEFAULT_VALIDATION_EVALUATED_ROWS, help="Prospective validation labels to select.")
    select.add_argument("--seed", type=int, default=0, help="Selection seed.")
    select.add_argument("--selection-protocol", choices=("v3", "v4"), default="v3", help="Target-free selection recipe.")

    analyze = sub.add_parser("analyze", help="Run the 30-feature frontier regime-map analysis on evaluated rows.")
    analyze.add_argument("--evaluated-table", required=True, help="Evaluated 30-feature table CSV/parquet.")
    analyze.add_argument("--out", default="results_frontier_regime", help="Output directory.")
    analyze.add_argument("--seed", type=int, default=0, help="Analysis seed.")
    analyze.add_argument("--win-threshold", type=float, default=0.05, help="QRC-useful threshold.")

    support = sub.add_parser("support", help="Fit discovery-only atlas-support/OOD scores for target rows.")
    support.add_argument("--discovery-table", required=True, help="Discovery table used to fit support distances.")
    support.add_argument("--target-table", required=True, help="Validation/real-probe table to score.")
    support.add_argument("--out", default="results_frontier_support", help="Output directory.")
    support.add_argument("--k-values", default="15,30,50", help="Comma-separated k values for support scoring.")

    plot = sub.add_parser("plot", help="Write publication figures for the discovery/validation frontier atlas.")
    plot.add_argument("--discovery-table", default="results_frontier_discovery/frontier_discovery_evaluated_30_features.csv")
    plot.add_argument("--validation-table", default="results_frontier_validation/frontier_validation_evaluated_30_features.csv")
    plot.add_argument("--discovery-analysis-dir", default="results_frontier_regime_discovery")
    plot.add_argument("--validation-analysis-dir", default="results_frontier_regime_validation")
    plot.add_argument("--out", default="results_frontier_publication")
    plot.add_argument("--seed", type=int, default=0, help="Meta-model and plotting seed.")
    plot.add_argument("--win-threshold", type=float, default=0.05, help="QRC-useful threshold.")
    plot.add_argument("--formats", default="png,pdf", help="Comma-separated output formats: png,pdf,svg.")

    evaluate = sub.add_parser("evaluate-selection", help="Run frozen QRC/ESN labels on selected frontier rows.")
    evaluate.add_argument("--selection", required=True, help="Frozen frontier_evaluation_selection CSV/parquet.")
    evaluate.add_argument("--out", default="results_frontier_evaluated", help="Output directory.")
    evaluate.add_argument("--split", choices=("discovery", "validation", "all"), default="discovery", help="Selection split to evaluate.")
    evaluate.add_argument(
        "--comparison-protocol",
        choices=("standard_v3", "standard_v2", "legacy_v1"),
        default="standard_v3",
        help="Model-comparison protocol passed to run_study.",
    )
    evaluate.add_argument("--calibration-config", default=None, help="Frozen standard_v3 calibration JSON.")
    evaluate.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True, help="Use fast study lengths; default true.")
    evaluate.add_argument("--smoke", action="store_true", help="Use smoke study mode.")
    evaluate.add_argument("--seeds", type=int, default=None, help="Number of reservoir seeds.")
    evaluate.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write resumable chunk checkpoints every N selected rows. Use 100 for long frontier runs.",
    )

    args = parser.parse_args(argv)
    if args.command == "property-atlas":
        df, path = write_property_atlas(
            out_dir=Path(args.out),
            n_per_template=args.n_per_template,
            seed=args.seed,
            fast=args.fast,
            smoke=args.smoke,
            max_rows=args.max_rows,
            taxonomy=args.taxonomy,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"wrote={path}")
        print(f"n_rows={len(df)}")
        return 0
    if args.command == "join-30":
        df, path = write_frontier_30_table(
            catalog_path=Path(args.catalog),
            extended_features_path=Path(args.extended_features) if args.extended_features else None,
            out_dir=Path(args.out),
        )
        print(f"wrote={path}")
        print(f"n_rows={len(df)}")
        return 0
    if args.command == "select":
        df, path = write_evaluation_selection(
            property_atlas_path=Path(args.property_atlas),
            out_dir=Path(args.out),
            n_discovery=args.n_discovery,
            n_validation=args.n_validation,
            seed=args.seed,
            selection_protocol=args.selection_protocol,
        )
        print(f"wrote={path}")
        print(f"n_rows={len(df)}")
        print(f"n_selected={(df['evaluation_split'] != 'unselected').sum()}")
        return 0
    if args.command == "analyze":
        manifest = write_frontier_regime_analysis(
            evaluated_table_path=Path(args.evaluated_table),
            out_dir=Path(args.out),
            seed=args.seed,
            win_threshold=args.win_threshold,
        )
        print(f"wrote={Path(args.out) / 'frontier_regime_manifest.json'}")
        print(f"n_rows={manifest['n_rows']}")
        return 0
    if args.command == "support":
        k_values = tuple(int(v.strip()) for v in args.k_values.split(",") if v.strip())
        df, path = write_support_scores(
            discovery_table_path=Path(args.discovery_table),
            target_table_path=Path(args.target_table),
            out_dir=Path(args.out),
            k_values=k_values,
        )
        print(f"wrote={path}")
        print(f"n_rows={len(df)}")
        print(f"n_ood={int(df['ood_flag'].sum())}")
        return 0
    if args.command == "plot":
        manifest = run_frontier_publication_plots(
            discovery_table=Path(args.discovery_table),
            validation_table=Path(args.validation_table),
            discovery_analysis_dir=Path(args.discovery_analysis_dir),
            validation_analysis_dir=Path(args.validation_analysis_dir),
            out_dir=Path(args.out),
            seed=args.seed,
            win_threshold=args.win_threshold,
            formats=args.formats.split(","),
        )
        print(f"wrote={args.out}")
        print(f"n_validation={manifest['n_validation']}")
        print(f"figures={len(manifest['figures'])}")
        print("claim_boundary=" + manifest["claim_boundary"])
        return 0
    if args.command == "evaluate-selection":
        df, path = write_evaluated_selection(
            selection_path=Path(args.selection),
            out_dir=Path(args.out),
            split=args.split,
            comparison_protocol=args.comparison_protocol,
            calibration_config=Path(args.calibration_config) if args.calibration_config else None,
            fast=args.fast,
            smoke=args.smoke,
            seeds=args.seeds,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"wrote={path}")
        print(f"n_rows={len(df)}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
