"""CLI for screening whether a new time series is likely to benefit from QRC."""

from __future__ import annotations

import argparse
from pathlib import Path

from qrc_dataset_profiler.triage import (
    DEFAULT_DISCOVERY_TABLE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_WIN_THRESHOLD,
    load_discovery_table,
    read_series_csv,
    report_to_json,
    report_to_text,
    triage_series,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Triage a univariate time series for likely QRC usefulness.")
    parser.add_argument("--series", required=True, help="CSV/TSV file containing one univariate time series.")
    parser.add_argument("--column", default=None, help="Column name or integer index. Required if multiple numeric columns exist.")
    parser.add_argument("--sep", default=",", help="CSV separator; use '\\t' for TSV.")
    parser.add_argument("--no-header", action="store_true", help="Treat the file as headerless.")
    parser.add_argument("--name", default="user_series", help="Dataset name shown in the report.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon metadata for descriptor computation.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Use at most this many samples for atlas-comparable descriptors; <=0 uses the full series.",
    )
    parser.add_argument("--window", choices=("tail", "head", "full"), default="tail", help="Which window to analyze when the series is long.")
    parser.add_argument("--discovery-table", default=str(DEFAULT_DISCOVERY_TABLE), help="Frozen discovery table used to fit the triage model.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic meta-model seed.")
    parser.add_argument("--win-threshold", type=float, default=DEFAULT_WIN_THRESHOLD, help="QRC-useful advantage threshold.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    parser.add_argument("--out", default=None, help="Optional path to write the report.")
    args = parser.parse_args(argv)

    column = _parse_column(args.column)
    sep = "\t" if args.sep == "\\t" else args.sep
    series = read_series_csv(Path(args.series), column=column, sep=sep, header=not args.no_header)
    discovery = load_discovery_table(Path(args.discovery_table))
    report = triage_series(
        series,
        discovery_table=discovery,
        name=args.name,
        horizon=args.horizon,
        max_length=args.max_length,
        window=args.window,
        seed=args.seed,
        win_threshold=args.win_threshold,
    )
    text = report_to_json(report) if args.format == "json" else report_to_text(report)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


def _parse_column(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
