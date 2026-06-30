"""CLI for the v5 multi-QRC atlas protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from qrc_dataset_profiler.analysis import load_catalog
from qrc_dataset_profiler.v5_protocol import run_v5_calibration, summarize_v5_evaluation, write_v5_evaluated_selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run v5 globally frozen multi-QRC-vs-ESN protocol steps.")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="Calibrate QRC-M/QRC-E/QRC-D and canonical ESN once globally.")
    cal.add_argument("--out", default="results_calibration_v5")
    cal.add_argument("--sweep-seed", type=int, default=1501)
    cal.add_argument("--sweep-n-per-template", type=int, default=20)
    cal.add_argument("--rows-per-family", type=int, default=20)
    cal.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    cal.add_argument("--seeds", type=int, default=1)
    cal.add_argument("--small-grid", action="store_true")
    cal.add_argument("--selection-tolerance", type=float, default=0.005)

    evalp = sub.add_parser("evaluate-selection", help="Evaluate selected synthetic rows under frozen v5 protocol.")
    evalp.add_argument("--selection", required=True)
    evalp.add_argument("--calibration-config", default="results_calibration_v5/frozen_v5_config.json")
    evalp.add_argument("--out", default="results_frontier_v5_discovery")
    evalp.add_argument("--split", choices=("discovery", "validation", "all"), default="discovery")
    evalp.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    evalp.add_argument("--smoke", action="store_true")
    evalp.add_argument("--seeds", type=int, default=1)
    evalp.add_argument("--include-nvar", action=argparse.BooleanOptionalAction, default=True)
    evalp.add_argument("--checkpoint-every", type=int, default=0)

    summ = sub.add_parser("summarize", help="Write a compact v5 summary table.")
    summ.add_argument("--evaluated-table", required=True)
    summ.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        manifest = run_v5_calibration(
            out_dir=Path(args.out),
            sweep_seed=args.sweep_seed,
            sweep_n_per_template=args.sweep_n_per_template,
            calibration_rows_per_family=args.rows_per_family,
            fast=args.fast,
            seeds=args.seeds,
            small_grid=args.small_grid,
            selection_tolerance=args.selection_tolerance,
        )
        print(f"wrote={Path(args.out) / 'frozen_v5_config.json'}")
        print(f"n_calibration_rows={manifest['calibration_data']['n_rows']}")
        print(f"feature_dim={manifest['esn']['reservoir_size']}")
        for variant, qrc in manifest["qrc_variants"].items():
            print(f"{variant}: J={qrc['J']} dt={qrc['dt']} reupload={qrc['reupload']} amp={qrc['amplitude_damping']} deph={qrc['dephasing']}")
        print(f"esn: rho={manifest['esn']['rho']} leak={manifest['esn']['leak']} input_scale={manifest['esn']['input_scale']}")
        print("claim_boundary=" + manifest["claim_boundary"])
        return 0
    if args.command == "evaluate-selection":
        df, path = write_v5_evaluated_selection(
            selection_path=Path(args.selection),
            calibration_config=Path(args.calibration_config),
            out_dir=Path(args.out),
            split=args.split,
            fast=args.fast,
            smoke=args.smoke,
            seeds=args.seeds,
            include_nvar=args.include_nvar,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"wrote={path}")
        print(f"n_rows={len(df)}")
        print(f"n_labeled={df['best_qrc_advantage_vs_esn'].notna().sum()}")
        return 0
    if args.command == "summarize":
        df = load_catalog(Path(args.evaluated_table))
        summary = summarize_v5_evaluation(df)
        out = Path(args.out) if args.out else Path(args.evaluated_table).with_name("v5_evaluation_summary.csv")
        summary.to_csv(out, index=False)
        print(f"wrote={out}")
        print(summary.head(10).to_string(index=False))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
