"""Paper-grade analysis helpers for the qrc_dataset_profiler sweep.

The analysis layer is intentionally downstream-only: it reads committed sweep
catalogs, reuses the frozen schema fields, and writes deterministic tables and
figures without changing the profiling or reservoir protocol.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrc_dataset_profiler.meta_model import fit_meta_model
from qrc_dataset_profiler.spec import CORE_AXIS_FIELDS


DIRECT_PREDICTABILITY_PROXIES = ("r2_linear", "forecastability", "pred_nrmse_gbm")
CHAOS_NONLINEARITY_COMPLEXITY_FEATURES = (
    "nl_gain",
    "lyapunov",
    "zero_one_K",
    "spectral_entropy",
    "spectral_flatness",
    "dfa_alpha",
    "perm_entropy",
    "sample_entropy",
    "hurst_rs",
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    description: str
    features: tuple[str, ...]


def feature_sets(catalog: pd.DataFrame) -> list[FeatureSet]:
    """Return the formal anti-circularity feature sets available in a catalog."""

    available = tuple(c for c in CORE_AXIS_FIELDS if c in catalog.columns)
    return [
        FeatureSet("all", "All schema-v1 core axis fields.", available),
        FeatureSet(
            "without_r2_linear",
            "All core axis fields except the direct linear predictability score.",
            tuple(c for c in available if c != "r2_linear"),
        ),
        FeatureSet(
            "without_predictability_proxies",
            "Core axis fields excluding r2_linear, forecastability, and pred_nrmse_gbm.",
            tuple(c for c in available if c not in DIRECT_PREDICTABILITY_PROXIES),
        ),
        FeatureSet(
            "chaos_nonlinearity_complexity_only",
            "Restricted to chaos, nonlinearity, spectral complexity, and entropy features.",
            tuple(c for c in CHAOS_NONLINEARITY_COMPLEXITY_FEATURES if c in available),
        ),
    ]


def run_analysis(
    catalog: pd.DataFrame,
    *,
    out_dir: Path,
    seed: int = 0,
    family_bootstraps: int = 1000,
    importance_bootstraps: int = 100,
    win_threshold: float = 0.05,
) -> dict[str, Any]:
    """Write deterministic Increment-4 analysis outputs and return a manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned = _validated_catalog(catalog)
    rng = np.random.default_rng(seed)

    sweep_summary = summarize_sweep(cleaned, win_threshold=win_threshold)
    family_ci = bootstrap_family_advantage(
        cleaned,
        n_bootstraps=family_bootstraps,
        rng=rng,
        win_threshold=win_threshold,
    )
    robustness, importance_tables = run_anti_circularity_suite(cleaned, seed=seed, win_threshold=win_threshold)
    importance_ci = bootstrap_feature_importances(
        cleaned,
        n_bootstraps=importance_bootstraps,
        seed=seed,
        win_threshold=win_threshold,
    )

    sweep_summary.to_csv(out_dir / "sweep_summary.csv", index=False)
    family_ci.to_csv(out_dir / "family_advantage_bootstrap.csv", index=False)
    robustness.to_csv(out_dir / "robustness_summary.csv", index=False)
    importance_ci.to_csv(out_dir / "importance_bootstrap.csv", index=False)
    for name, table in importance_tables.items():
        table.to_csv(out_dir / f"importances_{name}.csv", index=False)

    _write_sweep_summary_figure(sweep_summary, out_dir / "sweep_summary.png")
    _write_family_advantage_figure(family_ci, out_dir / "family_advantage_bootstrap.png")
    _write_robustness_figure(robustness, out_dir / "robustness_summary.png")
    _write_importance_figure(importance_ci, out_dir / "importance_bootstrap.png")

    output_names = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    output_names = sorted(set(output_names + ["analysis_manifest.json"]))
    manifest = {
        "analysis_version": "increment4-v1",
        "seed": int(seed),
        "family_bootstraps": int(family_bootstraps),
        "importance_bootstraps": int(importance_bootstraps),
        "win_threshold": float(win_threshold),
        "n_rows": int(len(cleaned)),
        "families": sorted(str(f) for f in cleaned["family"].dropna().unique()),
        "feature_sets": [
            {"name": fs.name, "description": fs.description, "features": list(fs.features)}
            for fs in feature_sets(cleaned)
        ],
        "claim_boundary": (
            "This analysis can support property-to-advantage robustness claims. "
            "It does not establish coupling, entanglement, or a quantum mechanism."
        ),
        "outputs": output_names,
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def load_catalog(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def summarize_sweep(catalog: pd.DataFrame, *, win_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in _iter_groups_with_overall(catalog):
        adv = _finite(group["qrc_advantage"])
        rows.append(
            {
                "family": family,
                "n": int(adv.size),
                "mean_advantage": _safe_mean(adv),
                "median_advantage": _safe_median(adv),
                "std_advantage": _safe_std(adv),
                "win_rate_gt0": _safe_mean(adv > 0.0),
                "win_rate_gt_threshold": _safe_mean(adv > win_threshold),
                "mean_nrmse_esn_matched": _safe_mean(_finite(group["nrmse_esn_matched"])),
                "mean_nrmse_qrc_spin": _safe_mean(_finite(group["nrmse_qrc_spin"])),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_family_advantage(
    catalog: pd.DataFrame,
    *,
    n_bootstraps: int,
    rng: np.random.Generator,
    win_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, group in _iter_groups_with_overall(catalog):
        adv = _finite(group["qrc_advantage"])
        means = _bootstrap_stat(adv, n_bootstraps=n_bootstraps, rng=rng, stat=np.mean)
        medians = _bootstrap_stat(adv, n_bootstraps=n_bootstraps, rng=rng, stat=np.median)
        win_rates = _bootstrap_stat((adv > win_threshold).astype(float), n_bootstraps=n_bootstraps, rng=rng, stat=np.mean)
        rows.append(
            {
                "family": family,
                "n": int(adv.size),
                "mean_advantage": _safe_mean(adv),
                "mean_ci_low": _percentile(means, 2.5),
                "mean_ci_high": _percentile(means, 97.5),
                "median_advantage": _safe_median(adv),
                "median_ci_low": _percentile(medians, 2.5),
                "median_ci_high": _percentile(medians, 97.5),
                "win_rate_gt_threshold": _safe_mean(adv > win_threshold),
                "win_rate_ci_low": _percentile(win_rates, 2.5),
                "win_rate_ci_high": _percentile(win_rates, 97.5),
            }
        )
    return pd.DataFrame(rows)


def run_anti_circularity_suite(
    catalog: pd.DataFrame,
    *,
    seed: int,
    win_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    importance_tables: dict[str, pd.DataFrame] = {}
    identity_cols = [c for c in ("family", "name", "dataset_id") if c in catalog.columns]
    target_cols = [c for c in ("qrc_advantage", "nrmse_esn_matched", "nrmse_qrc_spin") if c in catalog.columns]

    for fs in feature_sets(catalog):
        subset_cols = identity_cols + target_cols + list(fs.features)
        result = fit_meta_model(catalog.loc[:, list(dict.fromkeys(subset_cols))], seed=seed, win_threshold=win_threshold)
        imp = result.ranked_importances.copy()
        imp.insert(0, "feature_set", fs.name)
        importance_tables[fs.name] = imp
        gb_reg = result.regression_cv.get("models", {}).get("gradient_boosting", {})
        gb_clf = result.classification_cv.get("models", {}).get("gradient_boosting", {})
        rows.append(
            {
                "feature_set": fs.name,
                "description": fs.description,
                "n_samples": int(result.n_samples),
                "n_candidate_features": int(len(fs.features)),
                "n_features_used": int(len(result.features_used)),
                "features_used": ",".join(result.features_used),
                "regression_r2_mean": _float_or_nan(gb_reg.get("r2_mean")),
                "regression_mae_mean": _float_or_nan(gb_reg.get("mae_mean")),
                "classification_roc_auc_mean": _float_or_nan(gb_clf.get("roc_auc_mean")),
                "top_features": ",".join(imp["feature"].head(4).astype(str).tolist()) if not imp.empty else "",
                "notes": "; ".join(result.notes),
            }
        )
    return pd.DataFrame(rows), importance_tables


def bootstrap_feature_importances(
    catalog: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
    win_threshold: float,
) -> pd.DataFrame:
    """Bootstrap all-feature meta-model importances from row resamples."""

    base_result = fit_meta_model(catalog, seed=seed, win_threshold=win_threshold)
    base = base_result.ranked_importances[["feature", "importance_mean", "direction", "corr_with_advantage"]].copy()
    if base.empty:
        return pd.DataFrame(columns=["feature", "importance_mean", "ci_low", "ci_high", "selection_rate", "direction", "corr_with_advantage"])

    rng = np.random.default_rng(seed)
    traces: dict[str, list[float]] = {str(f): [] for f in base["feature"]}
    n = len(catalog)
    for b in range(max(0, int(n_bootstraps))):
        sample = catalog.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
        result = fit_meta_model(sample, seed=seed + b + 1, win_threshold=win_threshold)
        observed = {str(row.feature): float(row.importance_mean) for row in result.ranked_importances.itertuples(index=False)}
        for feature in traces:
            traces[feature].append(observed.get(feature, np.nan))

    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        vals = np.asarray(traces[str(row.feature)], dtype=float)
        finite = vals[np.isfinite(vals)]
        rows.append(
            {
                "feature": str(row.feature),
                "importance_mean": float(row.importance_mean),
                "ci_low": _percentile(finite, 2.5),
                "ci_high": _percentile(finite, 97.5),
                "selection_rate": float(finite.size / max(1, int(n_bootstraps))),
                "direction": str(row.direction),
                "corr_with_advantage": _float_or_nan(row.corr_with_advantage),
            }
        )
    return pd.DataFrame(rows).sort_values(["importance_mean", "selection_rate"], ascending=[False, False]).reset_index(drop=True)


def _validated_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    required = ("family", "qrc_advantage", "nrmse_esn_matched", "nrmse_qrc_spin")
    missing = [c for c in required if c not in catalog.columns]
    if missing:
        raise ValueError(f"catalog is missing required columns: {', '.join(missing)}")
    out = catalog.copy()
    for col in set(required).union(CORE_AXIS_FIELDS):
        if col in out.columns and col != "family":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[np.isfinite(out["qrc_advantage"].to_numpy(dtype=float))].reset_index(drop=True)
    if out.empty:
        raise ValueError("catalog has no finite qrc_advantage rows")
    return out


def _iter_groups_with_overall(catalog: pd.DataFrame):
    yield "overall", catalog
    for family, group in catalog.groupby("family", sort=True):
        yield str(family), group


def _bootstrap_stat(values: np.ndarray, *, n_bootstraps: int, rng: np.random.Generator, stat) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or n_bootstraps <= 0:
        return np.asarray([], dtype=float)
    idx = rng.integers(0, values.size, size=(int(n_bootstraps), values.size))
    return np.asarray([float(stat(values[row])) for row in idx], dtype=float)


def _write_sweep_summary_figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    families = summary[summary["family"] != "overall"].sort_values("n", ascending=True)
    axes[0].barh(families["family"], families["n"], color="#4c6f7f")
    axes[0].set_title("Sweep rows by family")
    axes[0].set_xlabel("Rows")
    ordered = families.sort_values("mean_advantage", ascending=True)
    colors = ["#9a4f4f" if v < 0 else "#2f6f6f" for v in ordered["mean_advantage"]]
    axes[1].barh(ordered["family"], ordered["mean_advantage"], color=colors)
    axes[1].axvline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_title("Mean QRC advantage")
    axes[1].set_xlabel("ESN NRMSE - QRC NRMSE")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_family_advantage_figure(ci: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = ci[ci["family"] != "overall"].sort_values("mean_advantage", ascending=True)
    y = np.arange(len(data))
    mean = data["mean_advantage"].to_numpy(dtype=float)
    low = data["mean_ci_low"].to_numpy(dtype=float)
    high = data["mean_ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
    ax.errorbar(mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="o", color="#2f6f6f", ecolor="#6b7280", capsize=3)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["family"])
    ax.set_xlabel("Mean QRC advantage with bootstrap 95% CI")
    ax.set_title("Family-level advantage intervals")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_robustness_figure(robustness: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    x = np.arange(len(robustness))
    ax.plot(x, robustness["regression_r2_mean"], marker="o", label="CV R2", color="#2f6f6f")
    ax.plot(x, robustness["classification_roc_auc_mean"], marker="s", label="ROC-AUC", color="#7b5e2f")
    ax.set_xticks(x)
    ax.set_xticklabels(robustness["feature_set"], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Anti-circularity robustness")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_importance_figure(importance: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.0), constrained_layout=True)
    if data.empty:
        ax.text(0.5, 0.5, "No usable importances", ha="center", va="center")
        ax.set_axis_off()
    else:
        y = np.arange(len(data))
        mean = data["importance_mean"].to_numpy(dtype=float)
        low = data["ci_low"].to_numpy(dtype=float)
        high = data["ci_high"].to_numpy(dtype=float)
        xerr = np.vstack([np.maximum(mean - low, 0.0), np.maximum(high - mean, 0.0)])
        ax.barh(y, mean, color="#4c6f7f")
        if np.isfinite(xerr).all():
            ax.errorbar(mean, y, xerr=xerr, fmt="none", ecolor="#222222", capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels(data["feature"])
        ax.set_xlabel("Permutation importance")
        ax.set_title("All-feature importance bootstrap")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _finite(series: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)) if arr.size else np.nan


def _safe_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)) if arr.size else np.nan


def _safe_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan


def _percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else np.nan


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
