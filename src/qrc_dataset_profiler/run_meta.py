"""Command line runner for the explanatory QRC-advantage meta-model."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

from qrc_dataset_profiler.meta_model import MetaModelResult, fit_meta_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the explanatory meta-model for QRC advantage.")
    parser.add_argument("--catalog", required=True, help="Input catalog CSV or parquet file.")
    parser.add_argument("--out", default="results_meta", help="Output directory.")
    parser.add_argument("--target", default="qrc_advantage", help="Regression target column.")
    parser.add_argument("--win-threshold", type=float, default=0.05, help="Threshold for QRC-win classification.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = _load_catalog(Path(args.catalog))
    result = fit_meta_model(catalog, target=args.target, win_threshold=args.win_threshold, seed=args.seed)

    result.ranked_importances.to_csv(out_dir / "importances.csv", index=False)
    _write_importance_bar(result, out_dir / "importance_bar.png")
    _write_partial_dependence(result, out_dir / "partial_dependence.png")
    _print_summary(result)
    return 0


def _load_catalog(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_importance_bar(result: MetaModelResult, path: Path) -> None:
    imp = result.ranked_importances.head(12)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    if imp.empty:
        ax.text(0.5, 0.5, "No usable importances", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = imp["feature"].astype(str).to_numpy()[::-1]
        vals = imp["importance_mean"].astype(float).to_numpy()[::-1]
        colors = ["#2f6f6f" if d == "positive" else "#9a4f4f" if d == "negative" else "#6b7280" for d in imp["direction"].to_numpy()[::-1]]
        ax.barh(np.arange(len(labels)), vals, color=colors)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Held-out permutation importance")
        ax.set_title("Properties explaining QRC advantage")
        ax.axvline(0.0, color="#333333", linewidth=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_partial_dependence(result: MetaModelResult, path: Path) -> None:
    estimator = result.estimators.get("regression_gradient_boosting")
    top_features = [f for f in result.ranked_importances["feature"].head(4).tolist() if f in result.features_used]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if estimator is None or result.X.size == 0 or not top_features:
        ax.text(0.5, 0.5, "No usable partial dependence", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    plt.close(fig)
    feature_indices = [result.features_used.index(f) for f in top_features]
    rows = 2 if len(feature_indices) > 2 else 1
    cols = min(2, len(feature_indices))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.4 * rows), constrained_layout=True)
    PartialDependenceDisplay.from_estimator(
        estimator,
        result.X,
        features=feature_indices,
        feature_names=result.features_used,
        ax=np.asarray(axes).ravel()[: len(feature_indices)],
    )
    fig.suptitle("Regression partial dependence")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _print_summary(result: MetaModelResult) -> None:
    reg_r2 = result.regression_cv.get("models", {}).get("gradient_boosting", {}).get("r2_mean", np.nan)
    clf_auc = result.classification_cv.get("models", {}).get("gradient_boosting", {}).get("roc_auc_mean", np.nan)
    print(f"n_samples={result.n_samples}")
    print(f"regression_cv_r2_gradient_boosting={_fmt(reg_r2)}")
    print(f"classification_cv_roc_auc_gradient_boosting={_fmt(clf_auc)}")
    if result.notes:
        print("notes=" + "; ".join(result.notes))
    if result.ranked_importances.empty:
        print("top_properties=none")
        return
    print("top_properties:")
    for row in result.ranked_importances.head(8).itertuples(index=False):
        sign = "+" if row.direction == "positive" else "-" if row.direction == "negative" else "0"
        print(f"  {row.feature}: importance={row.importance_mean:.4g}, direction={sign}, corr={_fmt(row.corr_with_advantage)}")


def _fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
