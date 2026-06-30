"""CLI for synthetic-trained, real-world external validation probes."""

from __future__ import annotations

import argparse
from pathlib import Path

from qrc_dataset_profiler.real_validation import (
    build_real_probe_atlas,
    evaluate_real_label_subset,
    score_real_probe_atlas,
    select_real_label_subset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run external real-world validation for the synthetic QRC regime atlas.")
    sub = parser.add_subparsers(dest="command", required=True)

    prop = sub.add_parser("property-atlas", help="Fetch real benchmark series and compute real-window properties.")
    prop.add_argument("--out", default="results_real_validation")
    prop.add_argument("--window-length", type=int, default=800)
    prop.add_argument("--min-length", type=int, default=120)
    prop.add_argument("--max-windows", type=int, default=120)
    prop.add_argument("--max-windows-per-series", type=int, default=3)
    prop.add_argument("--seed", type=int, default=0)
    prop.add_argument("--include-m4", action=argparse.BooleanOptionalAction, default=True)
    prop.add_argument("--include-nab", action=argparse.BooleanOptionalAction, default=True)

    score = sub.add_parser("score", help="Score real windows with the synthetic-trained regime map.")
    score.add_argument("--property-atlas", default="results_real_validation/real_window_property_atlas.csv")
    score.add_argument("--discovery-table", default="results_frontier_v4_discovery/frontier_discovery_evaluated_30_features.csv")
    score.add_argument("--out", default="results_real_validation")
    score.add_argument("--seed", type=int, default=0)

    select = sub.add_parser("select-labels", help="Select a target-free real subset for QRC/ESN labels.")
    select.add_argument("--predictions", default="results_real_validation/real_window_predictions.csv")
    select.add_argument("--out", default="results_real_validation")
    select.add_argument("--n-rows", type=int, default=32)
    select.add_argument("--seed", type=int, default=0)

    evalp = sub.add_parser("evaluate-labels", help="Evaluate frozen QRC/ESN on the selected real windows.")
    evalp.add_argument("--selection", default="results_real_validation/real_label_selection.csv")
    evalp.add_argument("--windows", default="results_real_validation/real_windows.npz")
    evalp.add_argument("--calibration-config", default="results_calibration_v4/frozen_config.json")
    evalp.add_argument("--out", default="results_real_validation")
    evalp.add_argument("--seeds", type=int, default=1)

    allp = sub.add_parser("all", help="Run property-atlas, score, select, and optional label evaluation.")
    allp.add_argument("--out", default="results_real_validation")
    allp.add_argument("--discovery-table", default="results_frontier_v4_discovery/frontier_discovery_evaluated_30_features.csv")
    allp.add_argument("--calibration-config", default="results_calibration_v4/frozen_config.json")
    allp.add_argument("--window-length", type=int, default=800)
    allp.add_argument("--min-length", type=int, default=120)
    allp.add_argument("--max-windows", type=int, default=120)
    allp.add_argument("--max-windows-per-series", type=int, default=3)
    allp.add_argument("--n-labels", type=int, default=32)
    allp.add_argument("--label-seeds", type=int, default=1)
    allp.add_argument("--seed", type=int, default=0)
    allp.add_argument("--include-m4", action=argparse.BooleanOptionalAction, default=True)
    allp.add_argument("--include-nab", action=argparse.BooleanOptionalAction, default=True)
    allp.add_argument("--evaluate-labels", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(argv)
    if args.command == "property-atlas":
        df, path = build_real_probe_atlas(
            out_dir=Path(args.out),
            window_length=args.window_length,
            min_length=args.min_length,
            max_windows=args.max_windows,
            max_windows_per_series=args.max_windows_per_series,
            seed=args.seed,
            include_m4=args.include_m4,
            include_nab=args.include_nab,
        )
        print(f"wrote={path}")
        print(f"n_rows={len(df)}")
        return 0
    if args.command == "score":
        df, path = score_real_probe_atlas(
            property_atlas=Path(args.property_atlas),
            discovery_table=Path(args.discovery_table),
            out_dir=Path(args.out),
            seed=args.seed,
        )
        print(f"wrote={path}")
        print(f"n_rows={len(df)} n_ood={int(df['ood_flag'].sum())}")
        return 0
    if args.command == "select-labels":
        df, path = select_real_label_subset(
            predictions_path=Path(args.predictions),
            out_dir=Path(args.out),
            n_rows=args.n_rows,
            seed=args.seed,
        )
        print(f"wrote={path}")
        print(f"n_rows={len(df)}")
        return 0
    if args.command == "evaluate-labels":
        df, path = evaluate_real_label_subset(
            selection_path=Path(args.selection),
            windows_path=Path(args.windows),
            calibration_config=Path(args.calibration_config),
            out_dir=Path(args.out),
            seeds=args.seeds,
        )
        print(f"wrote={path}")
        print(f"n_rows={len(df)}")
        return 0
    if args.command == "all":
        out = Path(args.out)
        atlas, atlas_path = build_real_probe_atlas(
            out_dir=out,
            window_length=args.window_length,
            min_length=args.min_length,
            max_windows=args.max_windows,
            max_windows_per_series=args.max_windows_per_series,
            seed=args.seed,
            include_m4=args.include_m4,
            include_nab=args.include_nab,
        )
        scored, scored_path = score_real_probe_atlas(
            property_atlas=atlas_path,
            discovery_table=Path(args.discovery_table),
            out_dir=out,
            seed=args.seed,
        )
        selected, selection_path = select_real_label_subset(
            predictions_path=scored_path,
            out_dir=out,
            n_rows=args.n_labels,
            seed=args.seed,
        )
        labeled_n = 0
        if args.evaluate_labels:
            labeled, _ = evaluate_real_label_subset(
                selection_path=selection_path,
                windows_path=out / "real_windows.npz",
                calibration_config=Path(args.calibration_config),
                out_dir=out,
                seeds=args.label_seeds,
            )
            labeled_n = len(labeled)
        print(f"wrote={out}")
        print(f"property_rows={len(atlas)} scored_rows={len(scored)} selected_rows={len(selected)} labeled_rows={labeled_n}")
        print("claim_boundary=synthetic-trained regime map; real labels are external validation only")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
