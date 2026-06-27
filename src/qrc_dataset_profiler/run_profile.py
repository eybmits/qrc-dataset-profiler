"""Command line runner for Increment 1."""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / "qrc_dataset_profiler_mpl"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qrc_dataset_profiler.generators import ALL_SPECS, generate
from qrc_dataset_profiler.properties import count_spectral_peaks, profile_dataset
from qrc_dataset_profiler.spec import CORE_AXIS_FIELDS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the qrc_dataset_profiler Increment-1 catalog.")
    parser.add_argument("--smoke", action="store_true", help="Use length=2000 for synthetic datasets.")
    parser.add_argument("--out", default="results", help="Output directory.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets needing FULL length even under --smoke: ground-truth validation AND
    # chaos estimators (Lyapunov / 0-1 test) are only reliable at length >= 4000.
    _GT_KEYS = ("true_lyapunov", "true_hurst", "true_n_frequencies", "is_chaotic")
    rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    full_length_names: list[str] = []
    for base_spec in ALL_SPECS:
        smoke_target = args.smoke and base_spec.source != "real"
        spec = replace(base_spec, length=min(base_spec.length, 2000)) if smoke_target else base_spec
        ds = generate(spec)
        if ds.ground_truth.get("_unavailable"):
            warnings.warn(f"skipping unavailable dataset {spec.name}", RuntimeWarning)
            continue
        # Ground-truth validation must use FULL length even under --smoke: Lyapunov/DFA
        # are only accurate at length >= 4000, so shortening them gives misleading errors.
        if smoke_target and spec.length != base_spec.length and any(k in ds.ground_truth for k in _GT_KEYS):
            spec = base_spec
            ds = generate(spec)
            full_length_names.append(spec.name)
        rec = profile_dataset(ds)
        rows.append(rec.to_row())
        validation_rows.extend(_validation_rows(ds, rec))

    df = pd.DataFrame(rows)
    if _has_pyarrow():
        df.to_parquet(out_dir / "catalog.parquet", index=False)
    else:
        df.to_csv(out_dir / "catalog.csv", index=False)

    _write_heatmap(df, out_dir / "correlation_heatmap.png")
    _write_coverage(df, out_dir / "coverage.png")
    pd.DataFrame(validation_rows).to_csv(out_dir / "gt_validation.csv", index=False)

    summary_cols = ["name", "family", "task_type", "ac_timescale", "r2_linear", "nl_gain", "lyapunov", "spectral_entropy", "pred_nrmse_gbm"]
    print(df[summary_cols].to_string(index=False, max_cols=len(summary_cols)))
    if args.smoke and full_length_names:
        print(f"\n[smoke] ground-truth datasets profiled at FULL length: {', '.join(full_length_names)}")
    return 0


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


def _write_heatmap(df: pd.DataFrame, path: Path) -> None:
    cols = [c for c in CORE_AXIS_FIELDS if c in df.columns]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_coverage(df: pd.DataFrame, path: Path) -> None:
    cols = ["lyapunov", "ac_timescale", "nl_gain", "spectral_entropy"]
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(9, 9), constrained_layout=True)
    for i, yi in enumerate(cols):
        for j, xj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[xj].dropna(), bins=12, color="#4c78a8", alpha=0.85)
            else:
                ax.scatter(data[xj], data[yi], s=16, alpha=0.75, color="#2f4b7c")
            if i == n - 1:
                ax.set_xlabel(xj, fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(yi, fontsize=7)
            else:
                ax.set_yticklabels([])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _validation_rows(ds, rec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    truth = ds.ground_truth
    if "true_lyapunov" in truth:
        rows.append({"name": rec.name, "quantity": "lyapunov", "estimated": rec.lyapunov, "true": truth["true_lyapunov"], "abs_error": abs(rec.lyapunov - truth["true_lyapunov"])})
    if "true_hurst" in truth:
        rows.append({"name": rec.name, "quantity": "hurst_from_dfa", "estimated": rec.dfa_alpha, "true": truth["true_hurst"], "abs_error": abs(rec.dfa_alpha - truth["true_hurst"])})
    if "true_n_frequencies" in truth:
        estimated = count_spectral_peaks(ds.series)
        rows.append({"name": rec.name, "quantity": "n_frequencies", "estimated": estimated, "true": truth["true_n_frequencies"], "abs_error": abs(estimated - truth["true_n_frequencies"])})
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
