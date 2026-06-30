"""Dataset atlas for QRC usefulness.

The atlas turns a completed sweep catalog into a row-level and family-level map
of where the fixed Spin-QRC is useful relative to the matched ESN baseline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from qrc_dataset_profiler.analysis import load_catalog
from qrc_dataset_profiler.meta_model import fit_meta_model


DEFAULT_WIN_THRESHOLD = 0.05
DEFAULT_TIE_MARGIN = 0.05
USEFULNESS_ORDER = ("baseline_preferred", "near_tie", "qrc_useful")
USEFULNESS_COLORS = {
    "baseline_preferred": "#9a4f4f",
    "near_tie": "#6b7280",
    "qrc_useful": "#2f6f6f",
}


def run_usefulness_map(
    catalog: pd.DataFrame,
    *,
    out_dir: Path,
    seed: int = 0,
    win_threshold: float = DEFAULT_WIN_THRESHOLD,
    tie_margin: float = DEFAULT_TIE_MARGIN,
) -> dict[str, Any]:
    """Fit the atlas meta-model and write row-level maps, summaries, and figures."""

    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned = _validated_catalog(catalog)
    result = fit_meta_model(cleaned, seed=seed, win_threshold=win_threshold)
    if result.X.size == 0 or len(result.features_used) == 0:
        raise ValueError("no usable feature matrix for usefulness map")

    row_map = build_row_map(cleaned, result, win_threshold=win_threshold, tie_margin=tie_margin)
    family_summary = summarize_families(row_map)
    meta_summary = summarize_meta_model(result)

    row_map.to_csv(out_dir / "qrc_usefulness_map.csv", index=False)
    family_summary.to_csv(out_dir / "family_usefulness_summary.csv", index=False)
    meta_summary.to_csv(out_dir / "meta_model_summary.csv", index=False)
    result.ranked_importances.to_csv(out_dir / "atlas_importances.csv", index=False)

    _write_property_map(row_map, out_dir / "property_usefulness_map.png")
    _write_prediction_map(row_map, out_dir / "predicted_vs_actual_usefulness.png")
    _write_family_category_figure(family_summary, out_dir / "family_usefulness_categories.png")
    _write_importance_figure(result.ranked_importances, out_dir / "atlas_importances.png")

    manifest = {
        "analysis_version": "usefulness-atlas-v1",
        "seed": int(seed),
        "win_threshold": float(win_threshold),
        "tie_margin": float(tie_margin),
        "n_rows": int(len(row_map)),
        "families": sorted(str(f) for f in row_map["family"].unique()),
        "features_used": list(result.features_used),
        "label_definition": {
            "qrc_useful": f"qrc_advantage >= {win_threshold}",
            "near_tie": f"-{tie_margin} <= qrc_advantage < {win_threshold}",
            "baseline_preferred": f"qrc_advantage < -{tie_margin}",
        },
        "claim_boundary": (
            "This atlas categorizes dataset regimes where the fixed Spin-QRC is useful "
            "relative to matched baselines. It is not a fundamental quantum-advantage claim."
        ),
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    manifest["outputs"] = sorted(set(manifest["outputs"] + ["usefulness_atlas_manifest.json"]))
    (out_dir / "usefulness_atlas_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def run_usefulness_map_from_path(catalog_path: Path, **kwargs: Any) -> dict[str, Any]:
    return run_usefulness_map(load_catalog(catalog_path), **kwargs)


def build_row_map(
    catalog: pd.DataFrame,
    result,
    *,
    win_threshold: float,
    tie_margin: float,
) -> pd.DataFrame:
    identity_cols = [
        c
        for c in (
            "dataset_id",
            "name",
            "family",
            "source",
            "task_type",
            "seed",
            "length",
            "horizon",
            "nrmse_esn_matched",
            "nrmse_qrc_spin",
            "qrc_advantage",
        )
        if c in catalog.columns
    ]
    out = catalog.loc[:, identity_cols].reset_index(drop=True).copy()
    actual = pd.to_numeric(out["qrc_advantage"], errors="coerce").to_numpy(dtype=float)

    reg = result.estimators.get("regression_gradient_boosting")
    pred = np.asarray(reg.predict(result.X), dtype=float) if reg is not None else np.full(len(out), np.nan)
    clf = result.estimators.get("classification_gradient_boosting")
    if clf is not None and hasattr(clf, "predict_proba"):
        p_useful = np.asarray(clf.predict_proba(result.X)[:, 1], dtype=float)
    else:
        p_useful = np.full(len(out), np.nan)

    coords = _pca_coordinates(result.X)
    out["actual_usefulness_label"] = [_usefulness_label(v, win_threshold=win_threshold, tie_margin=tie_margin) for v in actual]
    out["predicted_qrc_advantage"] = pred
    out["predicted_usefulness_label"] = [_usefulness_label(v, win_threshold=win_threshold, tie_margin=tie_margin) for v in pred]
    out["predicted_prob_qrc_useful"] = p_useful
    out["property_pc1"] = coords[:, 0]
    out["property_pc2"] = coords[:, 1]
    out["abs_prediction_error"] = np.abs(pred - actual)
    out["prediction_correct_label"] = out["actual_usefulness_label"] == out["predicted_usefulness_label"]
    return out


def summarize_families(row_map: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in row_map.groupby("family", sort=True):
        labels = group["actual_usefulness_label"].astype(str)
        n = len(group)
        rows.append(
            {
                "family": str(family),
                "n": int(n),
                "mean_qrc_advantage": _safe_mean(group["qrc_advantage"]),
                "median_qrc_advantage": _safe_median(group["qrc_advantage"]),
                "mean_predicted_qrc_advantage": _safe_mean(group["predicted_qrc_advantage"]),
                "qrc_useful_rate": float((labels == "qrc_useful").mean()),
                "near_tie_rate": float((labels == "near_tie").mean()),
                "baseline_preferred_rate": float((labels == "baseline_preferred").mean()),
                "label_accuracy": float(group["prediction_correct_label"].mean()),
                "mean_prob_qrc_useful": _safe_mean(group["predicted_prob_qrc_useful"]),
                "dominant_category": _dominant_label(labels),
            }
        )
    return pd.DataFrame(rows).sort_values(["qrc_useful_rate", "mean_qrc_advantage"], ascending=[False, False]).reset_index(drop=True)


def summarize_meta_model(result) -> pd.DataFrame:
    reg = result.regression_cv.get("models", {}).get("gradient_boosting", {})
    clf = result.classification_cv.get("models", {}).get("gradient_boosting", {})
    return pd.DataFrame(
        [
            {
                "model": "gradient_boosting",
                "n_samples": int(result.n_samples),
                "n_features_used": int(len(result.features_used)),
                "regression_r2_mean": _float_or_nan(reg.get("r2_mean")),
                "regression_mae_mean": _float_or_nan(reg.get("mae_mean")),
                "classification_roc_auc_mean": _float_or_nan(clf.get("roc_auc_mean")),
                "top_features": ",".join(result.ranked_importances["feature"].head(8).astype(str).tolist()),
            }
        ]
    )


def _validated_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    required = ("name", "family", "qrc_advantage", "nrmse_esn_matched", "nrmse_qrc_spin")
    missing = [c for c in required if c not in catalog.columns]
    if missing:
        raise ValueError(f"catalog is missing required columns: {', '.join(missing)}")
    out = catalog.copy()
    for col in ("qrc_advantage", "nrmse_esn_matched", "nrmse_qrc_spin"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[np.isfinite(out["qrc_advantage"].to_numpy(dtype=float))].reset_index(drop=True)
    if out.empty:
        raise ValueError("catalog has no finite qrc_advantage rows")
    return out


def _pca_coordinates(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.shape[1] >= 2:
        return PCA(n_components=2, random_state=0).fit_transform(X)
    if X.shape[1] == 1:
        return np.column_stack([X[:, 0], np.zeros(X.shape[0])])
    return np.zeros((X.shape[0], 2), dtype=float)


def _usefulness_label(value: float, *, win_threshold: float, tie_margin: float) -> str:
    if not math.isfinite(float(value)):
        return "unknown"
    if float(value) >= win_threshold:
        return "qrc_useful"
    if float(value) >= -tie_margin:
        return "near_tie"
    return "baseline_preferred"


def _dominant_label(labels: pd.Series) -> str:
    counts = labels.value_counts()
    for label in USEFULNESS_ORDER:
        if label in counts.index and int(counts[label]) == int(counts.max()):
            return label
    return str(counts.index[0]) if not counts.empty else "unknown"


def _write_property_map(row_map: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 8}):
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        for label in USEFULNESS_ORDER:
            sub = row_map[row_map["actual_usefulness_label"] == label]
            if sub.empty:
                continue
            ax.scatter(
                sub["property_pc1"],
                sub["property_pc2"],
                s=24,
                alpha=0.82,
                color=USEFULNESS_COLORS[label],
                label=label.replace("_", " "),
                edgecolors="white",
                linewidths=0.3,
            )
        centroids = row_map.groupby("family", sort=True)[["property_pc1", "property_pc2"]].mean()
        label_offsets = {
            "input_driven": (0.62, 0.5),
            "nonstationary": (-0.52, 0.55),
            "linear_stochastic": (-0.2, -0.28),
            "nonlinear_stochastic": (0.18, -0.34),
            "long_range": (0.0, -0.3),
        }
        for family, coords in centroids.iterrows():
            dx, dy = label_offsets.get(str(family), (0.0, 0.0))
            ax.text(
                float(coords["property_pc1"]) + dx,
                float(coords["property_pc2"]) + dy,
                str(family).replace("_", " "),
                fontsize=7,
                color="#222222",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.78, "linewidth": 0.4},
            )
        ax.set_title("QRC usefulness map from dataset properties")
        ax.set_xlabel("Property map PC1")
        ax.set_ylabel("Property map PC2")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
        fig.savefig(path, dpi=200)
        plt.close(fig)


def _write_prediction_map(row_map: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(5.6, 5.1), constrained_layout=True)
        colors = [USEFULNESS_COLORS.get(v, "#6b7280") for v in row_map["actual_usefulness_label"]]
        ax.scatter(row_map["qrc_advantage"], row_map["predicted_qrc_advantage"], s=24, alpha=0.82, c=colors, edgecolors="white", linewidths=0.3)
        lo = float(np.nanmin([row_map["qrc_advantage"].min(), row_map["predicted_qrc_advantage"].min()]))
        hi = float(np.nanmax([row_map["qrc_advantage"].max(), row_map["predicted_qrc_advantage"].max()]))
        ax.plot([lo, hi], [lo, hi], color="#222222", linewidth=0.8)
        ax.axhline(0.05, color="#6b7280", linewidth=0.8, linestyle=":")
        ax.axvline(0.05, color="#6b7280", linewidth=0.8, linestyle=":")
        ax.set_title("Meta-model predicted vs observed usefulness")
        ax.set_xlabel("Observed QRC advantage")
        ax.set_ylabel("Predicted QRC advantage")
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(path, dpi=200)
        plt.close(fig)


def _write_family_category_figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = summary.sort_values("qrc_useful_rate", ascending=True)
    y = np.arange(len(data))
    left = np.zeros(len(data), dtype=float)
    with plt.rc_context({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 8}):
        fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
        for label, col in (
            ("baseline_preferred", "baseline_preferred_rate"),
            ("near_tie", "near_tie_rate"),
            ("qrc_useful", "qrc_useful_rate"),
        ):
            vals = data[col].to_numpy(dtype=float)
            ax.barh(y, vals, left=left, color=USEFULNESS_COLORS[label], label=label.replace("_", " "), height=0.72)
            left += vals
        ax.set_yticks(y)
        ax.set_yticklabels(data["family"].str.replace("_", " "))
        ax.set_xlabel("Share of family")
        ax.set_title("Dataset families by QRC usefulness category")
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3)
        fig.savefig(path, dpi=200)
        plt.close(fig)


def _write_importance_figure(importances: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = importances.head(12).iloc[::-1]
    with plt.rc_context({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        ax.barh(data["feature"], data["importance_mean"], color="#4c6f7f", height=0.72)
        ax.set_title("Meta-model drivers of QRC usefulness")
        ax.set_xlabel("Permutation importance")
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(path, dpi=200)
        plt.close(fig)


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else np.nan


def _safe_median(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if math.isfinite(out) else np.nan


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
