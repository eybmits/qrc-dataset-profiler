"""Publication-facing analysis suite for the v5 multi-QRC atlas.

The v5 protocol labels each selected synthetic row with three globally frozen
QRC variants and one globally frozen feature-matched ESN.  This module is a
downstream-only layer: it reads the completed discovery/validation tables and
writes deterministic tables, static figures, and a portable HTML report.
"""

from __future__ import annotations

import html
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree

from qrc_dataset_profiler.analysis import load_catalog
from qrc_dataset_profiler.spec import CORE_AXIS_FIELDS, FRONTIER_TIER_A_FIELDS


V5_PUBLICATION_VERSION = "v5-publication-analysis-v1"
V5_TARGET = "best_qrc_advantage_vs_esn"
V5_VARIANTS = ("qrc_m", "qrc_e", "qrc_d")
DEFAULT_THRESHOLDS = (0.0, 0.025, 0.05, 0.10)
DIRECT_PREDICTABILITY_PROXIES = (
    "r2_linear",
    "forecastability",
    "pred_nrmse_linear",
    "pred_nrmse_gbm",
    "predictability_gap_linear_gbm",
)
CHAOS_NONLINEARITY_COMPLEXITY_FIELDS = (
    "nl_gain",
    "lyapunov",
    "zero_one_K",
    "spectral_entropy",
    "spectral_flatness",
    "perm_entropy",
    "sample_entropy",
    "ext_lz_complexity",
    "ext_recurrence_rate",
    "ext_recurrence_determinism",
    "ext_fnn_fraction",
    "ext_corr_dim_approx",
    "ext_bds_like",
)
PERSISTENCE_SPECTRAL_NONSTATIONARITY_FIELDS = (
    "ac_timescale",
    "ami_first_min",
    "mem_capacity",
    "dfa_alpha",
    "hurst_rs",
    "dom_freq",
    "spectral_entropy",
    "spectral_flatness",
    "ext_psd_slope",
    "ext_spectral_centroid",
    "ext_trend_strength",
    "ext_changepoint_count",
    "adf_p",
    "kpss_p",
    "n_diffs",
    "ext_volatility_ac1",
    "ext_arch_lm5",
)


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
COLORS = {
    "blue": "#5477C4",
    "blue_light": "#CEDFFE",
    "blue_xlight": "#EAF1FE",
    "gold": "#B8A037",
    "gold_light": "#FFEA8F",
    "orange": "#CC6F47",
    "orange_light": "#FFBDA1",
    "olive": "#71B436",
    "olive_light": "#BEEB96",
    "pink": "#BD569B",
    "neutral": "#A7ADBD",
    "neutral_dark": "#464C55",
}
LABEL_COLORS = {
    "esn_preferred": COLORS["orange"],
    "qrc_small_win": COLORS["neutral"],
    "qrc_useful": COLORS["blue"],
}
VARIANT_LABELS = {
    "qrc_m": "QRC-M dynamics",
    "qrc_e": "QRC-E reupload",
    "qrc_d": "QRC-D dissipative",
}
MODEL_COLUMNS = {
    "Linear ridge": "nrmse_linear",
    "GBM": "nrmse_gbm",
    "NVAR": "nrmse_nvar",
    "Frozen ESN": "nrmse_esn_v5",
    "QRC-M": "nrmse_qrc_m",
    "QRC-E": "nrmse_qrc_e",
    "QRC-D": "nrmse_qrc_d",
    "Best QRC": "nrmse_best_qrc",
}


@dataclass(frozen=True)
class FeatureSet:
    name: str
    description: str
    features: tuple[str, ...]


@dataclass
class FastMetaResult:
    features_used: list[str]
    X: np.ndarray
    y: np.ndarray
    preprocessing: dict[str, Any]
    ranked_importances: pd.DataFrame
    regression_cv: dict[str, Any]
    classification_cv: dict[str, Any]
    notes: list[str]


def run_v5_publication_analysis(
    *,
    discovery_table: Path,
    validation_table: Path,
    out_dir: Path,
    seed: int = 0,
    win_threshold: float = 0.05,
    family_bootstraps: int = 1000,
    importance_bootstraps: int = 80,
    formats: Iterable[str] = ("png", "pdf"),
) -> dict[str, Any]:
    """Write the v5 paper-facing analysis package."""

    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    table_dir = out_dir / "tables"
    table_dir.mkdir(exist_ok=True)

    discovery = _prepare_v5_table(load_catalog(discovery_table), split="discovery", win_threshold=win_threshold)
    validation = _prepare_v5_table(load_catalog(validation_table), split="validation", win_threshold=win_threshold)
    available_features = tuple(f for f in FRONTIER_TIER_A_FIELDS if f in discovery.columns and f in validation.columns)
    if not available_features:
        raise ValueError("no v5 Tier-A features available in both splits")

    split_summary = _split_summary(discovery, validation, win_threshold=win_threshold)
    family_summary = _family_summary(discovery, validation, win_threshold=win_threshold, n_bootstraps=family_bootstraps, seed=seed)
    variant_summary = _variant_summary(discovery, validation, win_threshold=win_threshold)
    threshold_robustness = _threshold_robustness(discovery, validation, thresholds=DEFAULT_THRESHOLDS)
    baseline_summary = _baseline_summary(discovery, validation)

    print("v5 publication: fit discovery meta-model", flush=True)
    meta_result = _fit_fast_meta_model(discovery, feature_fields=available_features, seed=seed, win_threshold=win_threshold)
    print("v5 publication: score validation", flush=True)
    validation_pred, prospective_metrics = _predict_validation(discovery, validation, meta_result, seed=seed, win_threshold=win_threshold)
    feature_importances = meta_result.ranked_importances.copy()
    print("v5 publication: bootstrap feature stability", flush=True)
    feature_stability = _feature_importance_stability(discovery, meta_result.features_used, n_bootstraps=importance_bootstraps, seed=seed)
    print("v5 publication: feature-set robustness", flush=True)
    feature_set_robustness = _feature_set_robustness(discovery, validation, available_features=available_features, seed=seed, win_threshold=win_threshold)
    rule_pockets = _rule_pockets(discovery, validation, feature_importances, win_threshold=win_threshold, seed=seed)

    print("v5 publication: write tables", flush=True)
    split_summary.to_csv(table_dir / "v5_split_summary.csv", index=False)
    family_summary.to_csv(table_dir / "v5_family_summary.csv", index=False)
    variant_summary.to_csv(table_dir / "v5_variant_summary.csv", index=False)
    threshold_robustness.to_csv(table_dir / "v5_threshold_robustness.csv", index=False)
    baseline_summary.to_csv(table_dir / "v5_baseline_summary.csv", index=False)
    validation_pred.to_csv(table_dir / "v5_validation_predictions.csv", index=False)
    feature_importances.to_csv(table_dir / "v5_feature_importances.csv", index=False)
    feature_stability.to_csv(table_dir / "v5_feature_importance_stability.csv", index=False)
    feature_set_robustness.to_csv(table_dir / "v5_feature_set_robustness.csv", index=False)
    rule_pockets.to_csv(table_dir / "v5_rule_pockets.csv", index=False)

    figures: list[dict[str, str]] = []
    _use_chart_theme()
    fmt = _normalize_formats(formats)
    print("v5 publication: write figures", flush=True)
    figures.append(_figure_outcome_overview(discovery, validation, variant_summary, baseline_summary, figure_dir, fmt, win_threshold=win_threshold))
    figures.append(_figure_regime_map(validation_pred, family_summary, available_features, figure_dir, fmt, win_threshold=win_threshold, seed=seed))
    figures.append(_figure_family_effects(family_summary, figure_dir, fmt))
    figures.append(_figure_feature_regressions(validation_pred, feature_importances, figure_dir, fmt, win_threshold=win_threshold, seed=seed))
    figures.append(_figure_meta_model(validation_pred, feature_importances, prospective_metrics, figure_dir, fmt, win_threshold=win_threshold))
    figures.append(_figure_robustness(feature_set_robustness, threshold_robustness, split_summary, figure_dir, fmt))
    figures.append(_figure_family_feature_matrix(validation_pred, feature_importances, family_summary, available_features, figure_dir, fmt))

    claims = _claim_table(split_summary, family_summary, feature_set_robustness, prospective_metrics, rule_pockets)
    claims.to_csv(table_dir / "v5_claims_and_guardrails.csv", index=False)
    print("v5 publication: write reports", flush=True)
    report_path = _write_markdown_report(
        out_dir=out_dir,
        split_summary=split_summary,
        family_summary=family_summary,
        variant_summary=variant_summary,
        feature_set_robustness=feature_set_robustness,
        rule_pockets=rule_pockets,
        prospective_metrics=prospective_metrics,
        claims=claims,
    )
    html_path = _write_html_report(
        out_dir=out_dir,
        figures=figures,
        split_summary=split_summary,
        family_summary=family_summary,
        variant_summary=variant_summary,
        feature_set_robustness=feature_set_robustness,
        rule_pockets=rule_pockets,
        claims=claims,
        prospective_metrics=prospective_metrics,
        win_threshold=win_threshold,
    )

    manifest = {
        "analysis_version": V5_PUBLICATION_VERSION,
        "inputs": {
            "discovery_table": str(discovery_table),
            "validation_table": str(validation_table),
        },
        "outputs": {
            "tables_dir": str(table_dir),
            "figures_dir": str(figure_dir),
            "html": str(html_path),
            "markdown_report": str(report_path),
        },
        "n_discovery": int(len(discovery)),
        "n_validation": int(len(validation)),
        "n_features_declared": int(len(FRONTIER_TIER_A_FIELDS)),
        "n_features_available": int(len(available_features)),
        "features_available": list(available_features),
        "win_threshold": float(win_threshold),
        "family_bootstraps": int(family_bootstraps),
        "importance_bootstraps": int(importance_bootstraps),
        "prospective_validation_metrics": prospective_metrics,
        "top_features": feature_importances["feature"].head(12).astype(str).tolist() if not feature_importances.empty else [],
        "figures": figures,
        "claim_boundary": (
            "The v5 publication package supports a protocol-local regime-atlas claim for globally frozen "
            "canonical QRC variants against a globally frozen feature-matched ESN. It does not establish "
            "broad average QRC superiority, quantum advantage, or an entanglement/coupling mechanism."
        ),
    }
    (out_dir / "v5_publication_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def feature_sets(available_features: Iterable[str]) -> list[FeatureSet]:
    available = tuple(dict.fromkeys(str(f) for f in available_features))
    return [
        FeatureSet("tier_a_30", "All Tier-A dataset descriptors used for the v5 atlas map.", available),
        FeatureSet("core_20", "Original 20 core measured dataset descriptors.", tuple(f for f in CORE_AXIS_FIELDS if f in available)),
        FeatureSet(
            "without_predictability_proxies",
            "Tier-A descriptors excluding direct predictability proxies.",
            tuple(f for f in available if f not in DIRECT_PREDICTABILITY_PROXIES),
        ),
        FeatureSet(
            "without_chaos_nonlinearity_complexity",
            "Tier-A descriptors excluding chaos, nonlinear, and complexity descriptors.",
            tuple(f for f in available if f not in CHAOS_NONLINEARITY_COMPLEXITY_FIELDS),
        ),
        FeatureSet(
            "persistence_spectral_nonstationarity",
            "Persistence, memory, spectrum, volatility, and nonstationarity descriptors only.",
            tuple(f for f in PERSISTENCE_SPECTRAL_NONSTATIONARITY_FIELDS if f in available),
        ),
        FeatureSet(
            "chaos_nonlinearity_complexity_only",
            "Chaos, nonlinearity, entropy, recurrence, and complexity descriptors only.",
            tuple(f for f in CHAOS_NONLINEARITY_COMPLEXITY_FIELDS if f in available),
        ),
    ]


def _prepare_v5_table(df: pd.DataFrame, *, split: str, win_threshold: float) -> pd.DataFrame:
    required = {"dataset_id", "family", V5_TARGET, "best_qrc_variant", "nrmse_esn_v5"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"v5 table missing required columns: {sorted(missing)}")
    out = df.copy()
    out["evaluation_split"] = split
    out["qrc_advantage"] = pd.to_numeric(out[V5_TARGET], errors="coerce")
    out = out[np.isfinite(out["qrc_advantage"].to_numpy(dtype=float))].reset_index(drop=True)
    out["actual_usefulness_label"] = np.where(
        out["qrc_advantage"] >= float(win_threshold),
        "qrc_useful",
        np.where(out["qrc_advantage"] > 0.0, "qrc_small_win", "esn_preferred"),
    )
    out["nrmse_best_qrc"] = _best_qrc_nrmse(out)
    if "nrmse_nvar" in out.columns:
        out["advantage_best_qrc_vs_nvar"] = pd.to_numeric(out["nrmse_nvar"], errors="coerce") - out["nrmse_best_qrc"]
        out["advantage_esn_vs_nvar"] = pd.to_numeric(out["nrmse_nvar"], errors="coerce") - pd.to_numeric(out["nrmse_esn_v5"], errors="coerce")
    return out


def _best_qrc_nrmse(df: pd.DataFrame) -> pd.Series:
    values = pd.Series(np.nan, index=df.index, dtype=float)
    for variant in V5_VARIANTS:
        mask = df["best_qrc_variant"].astype(str).eq(variant)
        col = f"nrmse_{variant}"
        if col in df.columns:
            values.loc[mask] = pd.to_numeric(df.loc[mask, col], errors="coerce")
    fallback_cols = [f"nrmse_{variant}" for variant in V5_VARIANTS if f"nrmse_{variant}" in df.columns]
    if fallback_cols:
        fallback = df[fallback_cols].apply(pd.to_numeric, errors="coerce").min(axis=1)
        values = values.fillna(fallback)
    return values


def _split_summary(discovery: pd.DataFrame, validation: pd.DataFrame, *, win_threshold: float) -> pd.DataFrame:
    rows = [_split_summary_row("discovery", discovery, win_threshold), _split_summary_row("validation", validation, win_threshold)]
    return pd.DataFrame(rows)


def _split_summary_row(split: str, df: pd.DataFrame, win_threshold: float) -> dict[str, Any]:
    adv = _finite(df["qrc_advantage"])
    row: dict[str, Any] = {
        "split": split,
        "n": int(len(df)),
        "n_families": int(df["family"].nunique()),
        "n_base_generators": int(df["base_generator"].nunique()) if "base_generator" in df.columns else math.nan,
        "mean_best_qrc_advantage": _safe_mean(adv),
        "median_best_qrc_advantage": _safe_median(adv),
        "best_qrc_win_rate": _safe_mean(adv > 0.0),
        "best_qrc_useful_rate": _safe_mean(adv >= float(win_threshold)),
        "mean_nrmse_esn": _safe_mean(_finite(df["nrmse_esn_v5"])),
        "mean_nrmse_best_qrc": _safe_mean(_finite(df["nrmse_best_qrc"])),
        "mean_nrmse_nvar": _safe_mean(_finite(df["nrmse_nvar"])) if "nrmse_nvar" in df.columns else math.nan,
        "best_variant_mode": str(df["best_qrc_variant"].mode(dropna=True).iloc[0]) if df["best_qrc_variant"].notna().any() else "",
    }
    for variant in V5_VARIANTS:
        col = f"advantage_{variant}_vs_esn"
        vals = _finite(df[col]) if col in df.columns else np.asarray([], dtype=float)
        row[f"mean_advantage_{variant}"] = _safe_mean(vals)
        row[f"win_rate_{variant}"] = _safe_mean(vals > 0.0)
        row[f"useful_rate_{variant}"] = _safe_mean(vals >= float(win_threshold))
    return row


def _family_summary(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    win_threshold: float,
    n_bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, df in (("discovery", discovery), ("validation", validation)):
        for i, (family, group) in enumerate(df.groupby("family", sort=True)):
            adv = _finite(group["qrc_advantage"])
            means = _bootstrap_stat(adv, n_bootstraps=n_bootstraps, seed=seed + 1000 * (split == "validation") + i, stat=np.mean)
            useful = _bootstrap_stat((adv >= float(win_threshold)).astype(float), n_bootstraps=n_bootstraps, seed=seed + 37 + i, stat=np.mean)
            rows.append(
                {
                    "split": split,
                    "family": str(family),
                    "n": int(len(group)),
                    "mean_best_qrc_advantage": _safe_mean(adv),
                    "mean_ci_low": _percentile(means, 2.5),
                    "mean_ci_high": _percentile(means, 97.5),
                    "median_best_qrc_advantage": _safe_median(adv),
                    "best_qrc_win_rate": _safe_mean(adv > 0.0),
                    "best_qrc_useful_rate": _safe_mean(adv >= float(win_threshold)),
                    "useful_rate_ci_low": _percentile(useful, 2.5),
                    "useful_rate_ci_high": _percentile(useful, 97.5),
                    "most_common_best_variant": str(group["best_qrc_variant"].mode(dropna=True).iloc[0]) if group["best_qrc_variant"].notna().any() else "",
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "best_qrc_useful_rate", "mean_best_qrc_advantage"], ascending=[True, False, False]).reset_index(drop=True)


def _variant_summary(discovery: pd.DataFrame, validation: pd.DataFrame, *, win_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, df in (("discovery", discovery), ("validation", validation)):
        for variant in V5_VARIANTS:
            adv = _finite(df.get(f"advantage_{variant}_vs_esn", pd.Series(dtype=float)))
            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "label": VARIANT_LABELS[variant],
                    "mean_advantage": _safe_mean(adv),
                    "median_advantage": _safe_median(adv),
                    "win_rate": _safe_mean(adv > 0.0),
                    "useful_rate": _safe_mean(adv >= float(win_threshold)),
                    "best_variant_share": float((df["best_qrc_variant"].astype(str) == variant).mean()),
                }
            )
    return pd.DataFrame(rows)


def _threshold_robustness(discovery: pd.DataFrame, validation: pd.DataFrame, *, thresholds: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, df in (("discovery", discovery), ("validation", validation)):
        adv = _finite(df["qrc_advantage"])
        for threshold in thresholds:
            rows.append(
                {
                    "split": split,
                    "threshold": float(threshold),
                    "best_qrc_useful_rate": _safe_mean(adv >= float(threshold)),
                    "n_useful": int((adv >= float(threshold)).sum()),
                    "mean_advantage_above_threshold": _safe_mean(adv[adv >= float(threshold)]),
                }
            )
    return pd.DataFrame(rows)


def _baseline_summary(discovery: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, df in (("discovery", discovery), ("validation", validation)):
        esn = pd.to_numeric(df["nrmse_esn_v5"], errors="coerce")
        for model, col in MODEL_COLUMNS.items():
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "mean_nrmse": _safe_mean(_finite(vals)),
                    "median_nrmse": _safe_median(_finite(vals)),
                    "win_rate_vs_esn": _safe_mean(_finite(esn - vals) > 0.0) if model != "Frozen ESN" else 0.0,
                    "mean_advantage_vs_esn": _safe_mean(_finite(esn - vals)) if model != "Frozen ESN" else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _fit_fast_meta_model(
    df: pd.DataFrame,
    *,
    feature_fields: tuple[str, ...],
    seed: int,
    win_threshold: float,
) -> FastMetaResult:
    features = [f for f in feature_fields if f in df.columns]
    X_raw = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_all = pd.to_numeric(df["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    valid_y = np.isfinite(y_all)
    X_raw = X_raw.loc[valid_y].reset_index(drop=True)
    y = y_all[valid_y]
    notes: list[str] = []

    dropped_all_nan = [c for c in X_raw.columns if X_raw[c].isna().all()]
    X_raw = X_raw.drop(columns=dropped_all_nan)
    dropped_constant = []
    for col in X_raw.columns:
        vals = X_raw[col].dropna().to_numpy(dtype=float)
        if vals.size == 0 or np.nanstd(vals) <= 1e-12 or pd.Series(vals).nunique(dropna=True) <= 1:
            dropped_constant.append(col)
    X_raw = X_raw.drop(columns=dropped_constant)
    if dropped_all_nan:
        notes.append(f"dropped all-NaN features: {', '.join(dropped_all_nan)}")
    if dropped_constant:
        notes.append(f"dropped near-constant features: {', '.join(dropped_constant)}")
    if X_raw.empty:
        raise ValueError("no usable features for v5 meta-model")

    medians = X_raw.median(numeric_only=True).to_dict()
    X_imp = X_raw.fillna(medians)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_imp.to_numpy(dtype=float))
    features_used = list(X_imp.columns)

    reg_model = GradientBoostingRegressor(random_state=seed)
    clf_model = GradientBoostingClassifier(random_state=seed)
    regression_cv = _fast_regression_cv(X, y, reg_model, seed=seed)
    classification_cv = _fast_classification_cv(X, y >= float(win_threshold), clf_model, seed=seed)

    reg = clone(reg_model).fit(X, y)
    importances = np.asarray(getattr(reg, "feature_importances_", np.zeros(len(features_used))), dtype=float)
    rows = []
    for feature, importance in zip(features_used, importances):
        raw = X_imp[feature].to_numpy(dtype=float)
        corr = _safe_corr(raw, y)
        rows.append(
            {
                "feature": feature,
                "importance_mean": float(importance),
                "importance_std": math.nan,
                "corr_with_advantage": corr,
                "direction": "positive" if corr > 1e-12 else "negative" if corr < -1e-12 else "flat",
            }
        )
    ranked = pd.DataFrame(rows).sort_values(["importance_mean", "feature"], ascending=[False, True]).reset_index(drop=True)
    return FastMetaResult(
        features_used=features_used,
        X=X,
        y=y,
        preprocessing={
            "medians": {str(k): float(v) for k, v in medians.items()},
            "scaler_mean": dict(zip(features_used, scaler.mean_.astype(float))),
            "scaler_scale": dict(zip(features_used, scaler.scale_.astype(float))),
            "dropped_all_nan": dropped_all_nan,
            "dropped_constant": dropped_constant,
        },
        ranked_importances=ranked,
        regression_cv=regression_cv,
        classification_cv=classification_cv,
        notes=notes,
    )


def _fast_regression_cv(X: np.ndarray, y: np.ndarray, model: Any, *, seed: int) -> dict[str, Any]:
    k = min(5, max(2, len(y) // 1000))
    k = min(k, len(y))
    if len(y) < 8 or k < 2:
        return {"models": {"gradient_boosting": {"r2_mean": math.nan, "mae_mean": math.nan}}, "r2_mean": math.nan, "mae_mean": math.nan}
    r2_vals: list[float] = []
    mae_vals: list[float] = []
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X):
        fitted = clone(model).fit(X[train_idx], y[train_idx])
        pred = fitted.predict(X[test_idx])
        r2_vals.append(float(r2_score(y[test_idx], pred)) if np.var(y[test_idx]) > 1e-12 else math.nan)
        mae_vals.append(float(mean_absolute_error(y[test_idx], pred)))
    row = {"r2": r2_vals, "mae": mae_vals, "r2_mean": _safe_mean(r2_vals), "mae_mean": _safe_mean(mae_vals), "n_splits": k}
    return {"models": {"gradient_boosting": row}, "r2_mean": row["r2_mean"], "mae_mean": row["mae_mean"], "n_splits": k}


def _fast_classification_cv(X: np.ndarray, y_bin: np.ndarray, model: Any, *, seed: int) -> dict[str, Any]:
    classes, counts = np.unique(y_bin, return_counts=True)
    if len(classes) < 2:
        row = {"roc_auc_mean": math.nan, "average_precision_mean": math.nan, "brier_mean": math.nan, "n_splits": 0}
        return {"models": {"gradient_boosting": row}, **row}
    k = min(5, int(counts.min()))
    if k < 2:
        row = {"roc_auc_mean": math.nan, "average_precision_mean": math.nan, "brier_mean": math.nan, "n_splits": 0}
        return {"models": {"gradient_boosting": row}, **row}
    auc_vals: list[float] = []
    ap_vals: list[float] = []
    brier_vals: list[float] = []
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y_bin):
        fitted = clone(model).fit(X[train_idx], y_bin[train_idx])
        prob = fitted.predict_proba(X[test_idx])[:, 1]
        auc_vals.append(float(roc_auc_score(y_bin[test_idx], prob)))
        ap_vals.append(float(average_precision_score(y_bin[test_idx], prob)))
        brier_vals.append(float(brier_score_loss(y_bin[test_idx], prob)))
    row = {
        "roc_auc": auc_vals,
        "average_precision": ap_vals,
        "brier": brier_vals,
        "roc_auc_mean": _safe_mean(auc_vals),
        "average_precision_mean": _safe_mean(ap_vals),
        "brier_mean": _safe_mean(brier_vals),
        "n_splits": k,
    }
    return {"models": {"gradient_boosting": row}, "roc_auc_mean": row["roc_auc_mean"], "average_precision_mean": row["average_precision_mean"], "brier_mean": row["brier_mean"], "n_splits": k}


def _predict_validation(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    result: Any,
    *,
    seed: int,
    win_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    X_train = np.asarray(result.X, dtype=float)
    y_train = np.asarray(result.y, dtype=float)
    X_val = _transform_with_result(validation, result)
    y_val = pd.to_numeric(validation["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    useful_val = y_val >= float(win_threshold)
    if X_train.size == 0 or X_val.size == 0:
        raise ValueError("no usable features for v5 meta-model validation")

    reg = GradientBoostingRegressor(random_state=seed).fit(X_train, y_train)
    pred = reg.predict(X_val)
    out = validation.copy()
    out["predicted_qrc_advantage"] = pred

    metrics: dict[str, float] = {
        "n": float(len(out)),
        "regression_r2": _finite_or_nan(r2_score(y_val, pred) if np.var(y_val) > 1e-12 else math.nan),
        "regression_mae": _finite_or_nan(mean_absolute_error(y_val, pred)),
        "classification_roc_auc": math.nan,
        "classification_pr_auc": math.nan,
        "classification_brier": math.nan,
    }
    y_train_bin = y_train >= float(win_threshold)
    if len(np.unique(y_train_bin)) == 2 and len(np.unique(useful_val)) == 2:
        clf = GradientBoostingClassifier(random_state=seed).fit(X_train, y_train_bin)
        prob = clf.predict_proba(X_val)[:, 1]
        out["predicted_prob_qrc_useful"] = prob
        out["predicted_usefulness_label"] = np.where(prob >= 0.5, "qrc_useful", "not_useful")
        metrics["classification_roc_auc"] = _finite_or_nan(roc_auc_score(useful_val, prob))
        metrics["classification_pr_auc"] = _finite_or_nan(average_precision_score(useful_val, prob))
        metrics["classification_brier"] = _finite_or_nan(brier_score_loss(useful_val, prob))
    else:
        out["predicted_prob_qrc_useful"] = np.nan
        out["predicted_usefulness_label"] = "unavailable"
    out["abs_prediction_error"] = (out["qrc_advantage"] - out["predicted_qrc_advantage"]).abs()
    return out, metrics


def _transform_with_result(df: pd.DataFrame, result: Any) -> np.ndarray:
    features = list(result.features_used)
    prep = result.preprocessing or {}
    medians = prep.get("medians", {})
    means = prep.get("scaler_mean", {})
    scales = prep.get("scaler_scale", {})
    if not features:
        return np.empty((len(df), 0), dtype=float)
    X = pd.DataFrame(index=df.index)
    for feature in features:
        fallback = medians.get(feature, 0.0)
        X[feature] = pd.to_numeric(df[feature], errors="coerce") if feature in df.columns else np.nan
        X[feature] = X[feature].replace([np.inf, -np.inf], np.nan).fillna(fallback)
    arr = X.to_numpy(dtype=float)
    mean = np.asarray([means.get(feature, float(np.nanmean(arr[:, i])) if arr.shape[0] else 0.0) for i, feature in enumerate(features)], dtype=float)
    scale = np.asarray([scales.get(feature, 1.0) for feature in features], dtype=float)
    scale[~np.isfinite(scale) | (np.abs(scale) <= 1e-12)] = 1.0
    return (arr - mean) / scale


def _feature_set_robustness(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    available_features: tuple[str, ...],
    seed: int,
    win_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fs in feature_sets(available_features):
        if not fs.features:
            continue
        result = _fit_fast_meta_model(discovery, feature_fields=fs.features, seed=seed, win_threshold=win_threshold)
        try:
            _, prospective = _predict_validation(discovery, validation, result, seed=seed, win_threshold=win_threshold)
        except ValueError:
            prospective = {"regression_r2": math.nan, "regression_mae": math.nan, "classification_roc_auc": math.nan, "classification_pr_auc": math.nan, "classification_brier": math.nan}
        gb_reg = result.regression_cv.get("models", {}).get("gradient_boosting", {})
        gb_clf = result.classification_cv.get("models", {}).get("gradient_boosting", {})
        rows.append(
            {
                "feature_set": fs.name,
                "description": fs.description,
                "n_candidate_features": int(len(fs.features)),
                "n_features_used": int(len(result.features_used)),
                "features_used": ",".join(result.features_used),
                "discovery_cv_r2": _float_or_nan(gb_reg.get("r2_mean")),
                "discovery_cv_mae": _float_or_nan(gb_reg.get("mae_mean")),
                "discovery_cv_roc_auc": _float_or_nan(gb_clf.get("roc_auc_mean")),
                "discovery_cv_pr_auc": _float_or_nan(gb_clf.get("average_precision_mean")),
                "validation_r2": prospective["regression_r2"],
                "validation_mae": prospective["regression_mae"],
                "validation_roc_auc": prospective["classification_roc_auc"],
                "validation_pr_auc": prospective["classification_pr_auc"],
                "validation_brier": prospective["classification_brier"],
                "top_features": ",".join(result.ranked_importances["feature"].head(8).astype(str).tolist()) if not result.ranked_importances.empty else "",
                "notes": "; ".join(result.notes),
            }
        )
    return pd.DataFrame(rows)


def _feature_importance_stability(
    discovery: pd.DataFrame,
    features: list[str],
    *,
    n_bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(columns=["feature", "gbr_importance_mean", "ci_low", "ci_high", "selection_rate"])
    X = discovery[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = pd.to_numeric(discovery["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)
    X = X.loc[mask]
    y = y[mask]
    scaler = StandardScaler()
    X_arr = scaler.fit_transform(X.to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    traces = {feature: [] for feature in features}
    n = len(y)
    sample_size = min(n, 3000)
    for b in range(max(0, int(n_bootstraps))):
        idx = rng.integers(0, n, size=sample_size)
        model = GradientBoostingRegressor(random_state=seed + b + 1, n_estimators=60).fit(X_arr[idx], y[idx])
        for feature, val in zip(features, model.feature_importances_):
            traces[feature].append(float(val))
    rows = []
    for feature in features:
        vals = np.asarray(traces[feature], dtype=float)
        finite = vals[np.isfinite(vals)]
        rows.append(
            {
                "feature": feature,
                "gbr_importance_mean": _safe_mean(finite),
                "ci_low": _percentile(finite, 2.5),
                "ci_high": _percentile(finite, 97.5),
                "selection_rate": float((finite > 0.0).mean()) if finite.size else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("gbr_importance_mean", ascending=False).reset_index(drop=True)


def _rule_pockets(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    importances: pd.DataFrame,
    *,
    win_threshold: float,
    seed: int,
) -> pd.DataFrame:
    features = [f for f in importances["feature"].head(8).astype(str).tolist() if f in discovery.columns and f in validation.columns] if not importances.empty else []
    if len(features) < 2:
        return pd.DataFrame(columns=["rule", "split", "n", "win_rate", "useful_rate", "mean_advantage"])
    train_x = _rank_features(discovery, features)
    val_x = _rank_features(validation, features)
    y = (pd.to_numeric(discovery["qrc_advantage"], errors="coerce") >= float(win_threshold)).to_numpy(dtype=bool)
    if len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["rule", "split", "n", "win_rate", "useful_rate", "mean_advantage"])
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(50, int(0.015 * len(discovery))), random_state=seed)
    clf.fit(train_x.to_numpy(dtype=float), y)
    train_leaf = clf.apply(train_x.to_numpy(dtype=float))
    val_leaf = clf.apply(val_x.to_numpy(dtype=float))
    rules = _extract_tree_rules(clf, features)
    rows: list[dict[str, Any]] = []
    for leaf_id, rule in rules.items():
        for split, df, leaves in (("discovery", discovery, train_leaf), ("validation", validation, val_leaf)):
            mask = leaves == leaf_id
            if int(mask.sum()) < 20:
                continue
            adv = _finite(df.loc[mask, "qrc_advantage"])
            rows.append(
                {
                    "rule": rule,
                    "split": split,
                    "n": int(mask.sum()),
                    "win_rate": _safe_mean(adv > 0.0),
                    "useful_rate": _safe_mean(adv >= float(win_threshold)),
                    "mean_advantage": _safe_mean(adv),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["split", "useful_rate", "n"], ascending=[True, False, False]).reset_index(drop=True)


def _rank_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feature in features:
        vals = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[feature] = vals.rank(method="average", pct=True).fillna(0.5)
    return out


def _extract_tree_rules(clf: DecisionTreeClassifier, features: list[str]) -> dict[int, str]:
    tree = clf.tree_
    rules: dict[int, str] = {}

    def walk(node: int, parts: list[str]) -> None:
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            rules[node] = " and ".join(parts) if parts else "all rows"
            return
        feature = features[int(tree.feature[node])]
        threshold = float(tree.threshold[node])
        walk(tree.children_left[node], parts + [f"{feature} percentile <= {threshold:.2f}"])
        walk(tree.children_right[node], parts + [f"{feature} percentile > {threshold:.2f}"])

    walk(0, [])
    return rules


def _figure_outcome_overview(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    variant_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
    *,
    win_threshold: float,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4), facecolor=TOKENS["surface"])
    ax_var, ax_hist, ax_base, ax_share = axes.ravel()

    val_variants = variant_summary[variant_summary["split"] == "validation"].copy()
    x = np.arange(len(val_variants))
    width = 0.34
    ax_var.bar(x - width / 2, val_variants["win_rate"], width=width, color=COLORS["blue_light"], edgecolor=COLORS["blue"], label="Any win")
    ax_var.bar(x + width / 2, val_variants["useful_rate"], width=width, color=COLORS["blue"], edgecolor="#2E4780", label=f"Useful >= {win_threshold:.3g}")
    ax_var.set_xticks(x)
    ax_var.set_xticklabels(val_variants["label"], rotation=10, ha="right")
    ax_var.set_ylim(0, max(0.6, float(val_variants[["win_rate", "useful_rate"]].max().max()) + 0.08))
    ax_var.set_ylabel("Validation share")
    ax_var.set_title("a  Frozen QRC variants versus ESN", loc="left")
    ax_var.legend(frameon=False, loc="upper right")
    _clean_axes(ax_var)

    bins = np.linspace(
        min(discovery["qrc_advantage"].quantile(0.005), validation["qrc_advantage"].quantile(0.005)),
        max(discovery["qrc_advantage"].quantile(0.995), validation["qrc_advantage"].quantile(0.995)),
        44,
    )
    ax_hist.hist(discovery["qrc_advantage"], bins=bins, color=COLORS["neutral"], alpha=0.52, label="Discovery")
    ax_hist.hist(validation["qrc_advantage"], bins=bins, color=COLORS["blue"], alpha=0.58, label="Validation")
    ax_hist.axvline(0.0, color=TOKENS["ink"], linewidth=1.0)
    ax_hist.axvline(win_threshold, color=COLORS["blue"], linewidth=1.0, linestyle="--")
    ax_hist.set_xlabel("Best QRC advantage vs ESN")
    ax_hist.set_ylabel("Rows")
    ax_hist.set_title("b  Best-QRC advantage distribution", loc="left")
    ax_hist.legend(frameon=False)
    _clean_axes(ax_hist)

    base = baseline_summary[baseline_summary["split"] == "validation"].dropna(subset=["mean_nrmse"]).copy()
    order = ["Linear ridge", "GBM", "NVAR", "Frozen ESN", "QRC-M", "QRC-E", "QRC-D", "Best QRC"]
    base["model"] = pd.Categorical(base["model"], categories=order, ordered=True)
    base = base.sort_values("model")
    ax_base.barh(base["model"].astype(str), base["mean_nrmse"], color=[COLORS["blue"] if m == "Frozen ESN" else COLORS["gold"] if "QRC" in m else COLORS["neutral"] for m in base["model"].astype(str)], edgecolor=TOKENS["ink"], linewidth=0.6)
    ax_base.invert_yaxis()
    ax_base.set_xlabel("Mean NRMSE, validation")
    ax_base.set_title("c  Baselines and QRC variants", loc="left")
    _clean_axes(ax_base)

    shares = validation["best_qrc_variant"].value_counts(normalize=True).reindex(V5_VARIANTS, fill_value=0.0)
    ax_share.bar([VARIANT_LABELS[v] for v in shares.index], shares.to_numpy(), color=[COLORS["blue"], COLORS["gold"], COLORS["olive"]], edgecolor=TOKENS["ink"], linewidth=0.6)
    ax_share.set_ylim(0, max(0.75, float(shares.max()) + 0.08))
    ax_share.set_ylabel("Share of validation rows")
    ax_share.set_title("d  Which QRC variant wins inside QRC?", loc="left")
    ax_share.tick_params(axis="x", rotation=10)
    _clean_axes(ax_share)

    _add_header(fig, "V5 outcome overview", "20,000 labeled rows; globally frozen QRC-M/QRC-E/QRC-D compared with a globally frozen feature-matched ESN.")
    return _save_figure(fig, out_dir, "v5_01_outcome_overview", formats, "Variant outcomes, advantage distribution, baseline NRMSE, and best-QRC composition.")


def _figure_regime_map(
    validation: pd.DataFrame,
    family_summary: pd.DataFrame,
    features: tuple[str, ...],
    out_dir: Path,
    formats: tuple[str, ...],
    *,
    win_threshold: float,
    seed: int,
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    coords = _feature_coordinates(validation, features, seed=seed)
    fig = plt.figure(figsize=(14.4, 8.2), facecolor=TOKENS["surface"])
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.45, 0.95], hspace=0.42, wspace=0.32)
    ax_map = fig.add_subplot(gs[:, :2])
    ax_family = fig.add_subplot(gs[0, 2])
    ax_variant = fig.add_subplot(gs[1, 2])

    adv = validation["qrc_advantage"].to_numpy(dtype=float)
    lim = max(0.12, float(np.nanpercentile(np.abs(adv), 98)))
    sc = ax_map.scatter(
        coords[:, 0],
        coords[:, 1],
        c=adv,
        s=13,
        cmap=_advantage_cmap(),
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
        alpha=0.78,
        edgecolors="none",
        rasterized=True,
    )
    useful = validation["qrc_advantage"] >= float(win_threshold)
    ax_map.scatter(coords[useful, 0], coords[useful, 1], s=18, facecolors="none", edgecolors=TOKENS["ink"], linewidths=0.35, rasterized=True)
    cb = fig.colorbar(sc, ax=ax_map, orientation="horizontal", fraction=0.048, pad=0.075)
    cb.set_label("Best QRC advantage vs ESN")
    cb.outline.set_edgecolor(TOKENS["axis"])
    ax_map.set_xlabel("Dataset-property PC1")
    ax_map.set_ylabel("Dataset-property PC2")
    ax_map.set_title("a  All validation rows in property space", loc="left")
    _clean_axes(ax_map)

    val_fam = family_summary[family_summary["split"] == "validation"].sort_values("best_qrc_useful_rate", ascending=True).tail(10)
    ax_family.barh(val_fam["family"].str.replace("_", " "), val_fam["best_qrc_useful_rate"], color=COLORS["blue"], edgecolor="#2E4780", linewidth=0.6)
    ax_family.set_xlabel("Useful rate")
    ax_family.set_title("b  Top validation families", loc="left")
    _clean_axes(ax_family)

    cross = pd.crosstab(validation["family"], validation["best_qrc_variant"], normalize="index").reindex(columns=list(V5_VARIANTS), fill_value=0.0)
    order = family_summary[family_summary["split"] == "validation"].sort_values("best_qrc_useful_rate", ascending=False)["family"].head(10).tolist()
    cross = cross.reindex(order).iloc[::-1]
    left = np.zeros(len(cross))
    colors = [COLORS["blue"], COLORS["gold"], COLORS["olive"]]
    for variant, color in zip(V5_VARIANTS, colors):
        vals = cross[variant].to_numpy(dtype=float)
        ax_variant.barh(cross.index.str.replace("_", " "), vals, left=left, color=color, edgecolor=TOKENS["panel"], linewidth=0.6, label=variant)
        left += vals
    ax_variant.set_xlim(0, 1)
    ax_variant.set_xlabel("Best-variant share")
    ax_variant.set_title("c  Variant composition", loc="left")
    ax_variant.legend(frameon=False, loc="lower right", fontsize=8)
    _clean_axes(ax_variant)

    _add_header(fig, "V5 regime map", "All validation datapoints projected from the 30 Tier-A descriptors; outlined points are useful QRC cases.")
    return _save_figure(fig, out_dir, "v5_02_regime_map", formats, "Validation regime map with all rows, family rates, and best-variant composition.")


def _figure_family_effects(family_summary: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    val = family_summary[family_summary["split"] == "validation"].sort_values("best_qrc_useful_rate", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.6), facecolor=TOKENS["surface"], gridspec_kw={"width_ratios": [0.95, 1.05]})
    ax_rate, ax_ci = axes
    y = np.arange(len(val))
    ax_rate.barh(y, val["best_qrc_useful_rate"], color=COLORS["blue"], edgecolor="#2E4780", linewidth=0.6)
    ax_rate.set_yticks(y)
    ax_rate.set_yticklabels(val["family"].str.replace("_", " "), fontsize=8)
    ax_rate.set_xlabel("Useful rate")
    ax_rate.set_title("a  Validation useful rate by family", loc="left")
    _clean_axes(ax_rate)

    mean = val["mean_best_qrc_advantage"].to_numpy(dtype=float)
    low = val["mean_ci_low"].to_numpy(dtype=float)
    high = val["mean_ci_high"].to_numpy(dtype=float)
    ax_ci.errorbar(mean, y, xerr=_xerr(mean, low, high), fmt="none", ecolor=COLORS["neutral_dark"], linewidth=0.9, capsize=2)
    ax_ci.scatter(mean, y, color=[COLORS["blue"] if lo > 0 else COLORS["orange"] if hi < 0 else COLORS["neutral"] for lo, hi in zip(low, high)], edgecolors=TOKENS["panel"], linewidths=0.5, s=34)
    ax_ci.axvline(0.0, color=TOKENS["ink"], linewidth=0.9)
    ax_ci.axvline(0.05, color=COLORS["blue"], linewidth=0.9, linestyle="--")
    ax_ci.set_yticks(y)
    ax_ci.set_yticklabels([])
    ax_ci.set_xlabel("Mean best-QRC advantage, 95% bootstrap CI")
    ax_ci.set_title("b  Mean effect by family", loc="left")
    _clean_axes(ax_ci)
    _add_header(fig, "Family-level v5 effects", "Family summaries are validation-only; intervals bootstrap rows within each family.")
    return _save_figure(fig, out_dir, "v5_03_family_effects", formats, "Validation family useful rates and mean-advantage confidence intervals.")


def _figure_feature_regressions(
    validation: pd.DataFrame,
    importances: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
    *,
    win_threshold: float,
    seed: int,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    features = [f for f in importances["feature"].head(8).astype(str).tolist() if f in validation.columns]
    if not features:
        raise ValueError("no feature importances available for regression figure")
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.3), facecolor=TOKENS["surface"], sharey=True)
    rng = np.random.default_rng(seed)
    for ax, feature in zip(axes.ravel(), features):
        _feature_regression_panel(ax, validation, feature, win_threshold=win_threshold, rng=rng)
    for ax in axes.ravel()[len(features) :]:
        ax.set_visible(False)
    _add_header(fig, "All-point feature regressions", "Each panel shows all validation rows; x-axes are percentile ranks of top discovery-trained meta-model features.")
    return _save_figure(fig, out_dir, "v5_04_feature_regressions", formats, "All validation datapoints plotted against top features with regression lines and confidence bands.")


def _feature_regression_panel(ax: Any, df: pd.DataFrame, feature: str, *, win_threshold: float, rng: np.random.Generator) -> None:
    raw_x = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(raw_x) & np.isfinite(y)
    x_rank = pd.Series(raw_x[mask]).rank(method="average", pct=True).to_numpy(dtype=float)
    if np.unique(x_rank).size < 80:
        x_rank = np.clip(x_rank + rng.normal(0, 0.003, size=x_rank.size), 0.0, 1.0)
    y_plot = y[mask]
    colors = np.where(y_plot >= float(win_threshold), COLORS["blue"], np.where(y_plot > 0, COLORS["neutral"], COLORS["orange"]))
    ax.scatter(x_rank, y_plot, s=9, c=colors, alpha=0.44, edgecolors="none", rasterized=True)
    fit = _linear_fit_with_ci(x_rank, y_plot)
    if fit is not None:
        x_line, y_hat, low, high, r2 = fit
        ax.fill_between(x_line, low, high, color=COLORS["blue_xlight"], alpha=0.86, linewidth=0)
        ax.plot(x_line, y_hat, color=TOKENS["ink"], linewidth=1.05)
    else:
        r2 = math.nan
    rho = float(pd.Series(raw_x[mask]).corr(pd.Series(y_plot), method="spearman")) if mask.sum() > 3 else math.nan
    ax.axhline(0.0, color=TOKENS["ink"], linewidth=0.8)
    ax.axhline(win_threshold, color=COLORS["blue"], linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Feature percentile")
    ax.set_ylabel("Best-QRC advantage")
    ax.set_title(_feature_label(feature), loc="left", fontsize=10)
    ax.text(0.03, 0.96, f"n={int(mask.sum())}\nrho={rho:.2f}\nR2={r2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7.2, bbox=_textbox())
    _clean_axes(ax)


def _figure_meta_model(
    validation: pd.DataFrame,
    importances: pd.DataFrame,
    metrics: dict[str, float],
    out_dir: Path,
    formats: tuple[str, ...],
    *,
    win_threshold: float,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4), facecolor=TOKENS["surface"])
    ax_pred, ax_resid, ax_cal, ax_imp = axes.ravel()
    y = validation["qrc_advantage"].to_numpy(dtype=float)
    pred = validation["predicted_qrc_advantage"].to_numpy(dtype=float)
    colors = [LABEL_COLORS.get(v, COLORS["neutral"]) for v in validation["actual_usefulness_label"]]
    ax_pred.scatter(y, pred, s=11, c=colors, alpha=0.62, edgecolors="none", rasterized=True)
    lo = float(np.nanmin([np.nanmin(y), np.nanmin(pred)]))
    hi = float(np.nanmax([np.nanmax(y), np.nanmax(pred)]))
    ax_pred.plot([lo, hi], [lo, hi], color=TOKENS["ink"], linewidth=0.9)
    ax_pred.axhline(win_threshold, color=COLORS["blue"], linewidth=0.8, linestyle="--")
    ax_pred.axvline(win_threshold, color=COLORS["blue"], linewidth=0.8, linestyle="--")
    ax_pred.set_xlabel("Observed advantage")
    ax_pred.set_ylabel("Predicted advantage")
    ax_pred.set_title("a  Discovery-trained prediction on validation", loc="left")
    ax_pred.text(0.04, 0.96, f"R2={metrics['regression_r2']:.2f}\nMAE={metrics['regression_mae']:.3f}\nAUC={metrics['classification_roc_auc']:.2f}", transform=ax_pred.transAxes, ha="left", va="top", fontsize=8, bbox=_textbox())
    _clean_axes(ax_pred)

    resid = y - pred
    ax_resid.scatter(pred, resid, s=11, c=colors, alpha=0.55, edgecolors="none", rasterized=True)
    ax_resid.axhline(0.0, color=TOKENS["ink"], linewidth=0.9)
    ax_resid.set_xlabel("Predicted advantage")
    ax_resid.set_ylabel("Residual")
    ax_resid.set_title("b  Validation residuals", loc="left")
    _clean_axes(ax_resid)

    _calibration_panel(ax_cal, validation, win_threshold=win_threshold)
    _importance_panel(ax_imp, importances)

    _add_header(fig, "V5 meta-model validation", "The meta-model is trained on discovery rows and evaluated once on the held-out validation split.")
    return _save_figure(fig, out_dir, "v5_05_meta_model_validation", formats, "Discovery-trained meta-model prediction, calibration, residuals, and feature importances.")


def _calibration_panel(ax: Any, validation: pd.DataFrame, *, win_threshold: float) -> None:
    prob = pd.to_numeric(validation["predicted_prob_qrc_useful"], errors="coerce").to_numpy(dtype=float)
    actual = (pd.to_numeric(validation["qrc_advantage"], errors="coerce").to_numpy(dtype=float) >= float(win_threshold)).astype(float)
    mask = np.isfinite(prob)
    ax.plot([0, 1], [0, 1], color=TOKENS["ink"], linestyle=":", linewidth=0.9)
    if mask.sum() > 10:
        edges = np.unique(np.quantile(prob[mask], np.linspace(0, 1, 11)))
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = mask & (prob >= lo) & (prob <= hi if hi == edges[-1] else prob < hi)
            if idx.sum():
                rows.append((float(prob[idx].mean()), float(actual[idx].mean()), int(idx.sum())))
        if rows:
            x, y, n = zip(*rows)
            ax.scatter(x, y, s=np.clip(np.asarray(n) * 0.6, 24, 130), color=COLORS["blue"], edgecolors=TOKENS["panel"], linewidths=0.5)
            ax.plot(x, y, color=COLORS["blue"], linewidth=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted useful probability")
    ax.set_ylabel("Observed useful rate")
    ax.set_title("c  Probability calibration", loc="left")
    _clean_axes(ax)


def _importance_panel(ax: Any, importances: pd.DataFrame) -> None:
    top = importances.head(12).iloc[::-1]
    colors = [COLORS["blue"] if d == "positive" else COLORS["orange"] if d == "negative" else COLORS["neutral"] for d in top["direction"].astype(str)]
    ax.barh(_feature_labels(top["feature"]), top["importance_mean"], color=colors, edgecolor=TOKENS["ink"], linewidth=0.5)
    ax.set_xlabel("Gradient boosting importance")
    ax.set_title("d  Top discovery importances", loc="left")
    _clean_axes(ax)


def _figure_robustness(
    feature_set_robustness: pd.DataFrame,
    threshold_robustness: pd.DataFrame,
    split_summary: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.6), facecolor=TOKENS["surface"], gridspec_kw={"width_ratios": [1.25, 0.95, 0.8]})
    ax_fs, ax_thr, ax_split = axes
    fs = feature_set_robustness.iloc[::-1].copy()
    y = np.arange(len(fs))
    ax_fs.barh(y - 0.16, fs["validation_roc_auc"], height=0.30, color=COLORS["blue"], label="ROC-AUC")
    ax_fs.barh(y + 0.16, fs["validation_pr_auc"], height=0.30, color=COLORS["gold"], label="PR-AUC")
    ax_fs.set_yticks(y)
    ax_fs.set_yticklabels(fs["feature_set"].str.replace("_", " "), fontsize=8)
    ax_fs.set_xlim(0, max(0.95, float(fs[["validation_roc_auc", "validation_pr_auc"]].max().max()) + 0.05))
    ax_fs.set_xlabel("Held-out validation score")
    ax_fs.set_title("a  Feature-set anti-circularity", loc="left")
    ax_fs.legend(frameon=False, loc="lower right")
    _clean_axes(ax_fs)

    for split, color in (("discovery", COLORS["neutral_dark"]), ("validation", COLORS["blue"])):
        sub = threshold_robustness[threshold_robustness["split"] == split]
        ax_thr.plot(sub["threshold"], sub["best_qrc_useful_rate"], marker="o", color=color, label=split.title(), linewidth=1.1)
    ax_thr.set_xlabel("Useful threshold")
    ax_thr.set_ylabel("Share above threshold")
    ax_thr.set_title("b  Threshold robustness", loc="left")
    ax_thr.legend(frameon=False)
    _clean_axes(ax_thr)

    split = split_summary.set_index("split").loc[["discovery", "validation"]]
    metrics = ["best_qrc_win_rate", "best_qrc_useful_rate"]
    x = np.arange(len(metrics))
    width = 0.34
    ax_split.bar(x - width / 2, split.loc["discovery", metrics], width=width, color=COLORS["neutral"], label="Discovery")
    ax_split.bar(x + width / 2, split.loc["validation", metrics], width=width, color=COLORS["blue"], label="Validation")
    ax_split.set_xticks(x)
    ax_split.set_xticklabels(["Any win", "Useful"])
    ax_split.set_ylabel("Share")
    ax_split.set_title("c  Split stability", loc="left")
    ax_split.legend(frameon=False)
    _clean_axes(ax_split)

    _add_header(fig, "Robustness checks", "The v5 regime signal is tested against feature removals, threshold choices, and discovery-validation split drift.")
    return _save_figure(fig, out_dir, "v5_06_robustness", formats, "Feature-set robustness, threshold robustness, and split stability.")


def _figure_family_feature_matrix(
    validation: pd.DataFrame,
    importances: pd.DataFrame,
    family_summary: pd.DataFrame,
    features: tuple[str, ...],
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    top_features = [f for f in importances["feature"].head(14).astype(str).tolist() if f in features and f in validation.columns]
    if not top_features:
        top_features = list(features[:14])
    fam = family_summary[family_summary["split"] == "validation"].sort_values("best_qrc_useful_rate", ascending=False)
    order = fam["family"].tolist()
    mat = validation.groupby("family")[top_features].mean(numeric_only=True).reindex(order)
    z = mat.apply(lambda s: (s - s.median()) / _robust_scale(s), axis=0).clip(-2.5, 2.5)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 7.4), facecolor=TOKENS["surface"], gridspec_kw={"width_ratios": [2.2, 0.6, 0.85]})
    ax_heat, ax_rate, ax_ci = axes
    image = ax_heat.imshow(z.to_numpy(dtype=float), aspect="auto", cmap=_feature_cmap(), norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5), interpolation="nearest")
    ax_heat.set_yticks(np.arange(len(z.index)))
    ax_heat.set_yticklabels([str(v).replace("_", " ") for v in z.index], fontsize=7.5)
    ax_heat.set_xticks(np.arange(len(z.columns)))
    ax_heat.set_xticklabels(_feature_labels(z.columns), rotation=45, ha="right", fontsize=7)
    ax_heat.set_title("a  Family-average top descriptors", loc="left")
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_color(TOKENS["axis"])
    cb = fig.colorbar(image, ax=ax_heat, fraction=0.028, pad=0.012)
    cb.set_label("Robust z-score")
    cb.outline.set_edgecolor(TOKENS["axis"])

    y = np.arange(len(fam))
    ax_rate.barh(y, fam["best_qrc_useful_rate"], color=COLORS["blue"], height=0.72)
    ax_rate.set_yticks(y)
    ax_rate.set_yticklabels([])
    ax_rate.invert_yaxis()
    ax_rate.set_xlabel("Useful")
    ax_rate.set_title("b  Rate", loc="left")
    _clean_axes(ax_rate)

    mean = fam["mean_best_qrc_advantage"].to_numpy(dtype=float)
    low = fam["mean_ci_low"].to_numpy(dtype=float)
    high = fam["mean_ci_high"].to_numpy(dtype=float)
    ax_ci.errorbar(mean, y, xerr=_xerr(mean, low, high), fmt="none", ecolor=COLORS["neutral_dark"], linewidth=0.8, capsize=2)
    ax_ci.scatter(mean, y, color=[COLORS["blue"] if lo > 0 else COLORS["orange"] if hi < 0 else COLORS["neutral"] for lo, hi in zip(low, high)], s=25, edgecolors=TOKENS["panel"], linewidths=0.4)
    ax_ci.axvline(0, color=TOKENS["ink"], linewidth=0.8)
    ax_ci.set_yticks(y)
    ax_ci.set_yticklabels([])
    ax_ci.invert_yaxis()
    ax_ci.set_xlabel("Mean adv.")
    ax_ci.set_title("c  Effect", loc="left")
    _clean_axes(ax_ci)

    _add_header(fig, "Family-property structure", "Top discovery-trained descriptors summarized by validation family, with validation useful rate and bootstrap effect interval.")
    return _save_figure(fig, out_dir, "v5_07_family_feature_matrix", formats, "Family-level feature heatmap with useful rates and effect intervals.")


def _feature_coordinates(df: pd.DataFrame, features: tuple[str, ...], *, seed: int) -> np.ndarray:
    X = df[list(features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def _claim_table(
    split_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    feature_set_robustness: pd.DataFrame,
    prospective_metrics: dict[str, float],
    rule_pockets: pd.DataFrame,
) -> pd.DataFrame:
    val = split_summary[split_summary["split"] == "validation"].iloc[0]
    top_family = family_summary[family_summary["split"] == "validation"].sort_values(["best_qrc_useful_rate", "mean_best_qrc_advantage"], ascending=False).iloc[0]
    no_proxy = _row_by_name(feature_set_robustness, "feature_set", "without_predictability_proxies")
    no_chaos = _row_by_name(feature_set_robustness, "feature_set", "without_chaos_nonlinearity_complexity")
    best_rule = rule_pockets[rule_pockets["split"] == "validation"].head(1)
    rule_text = "No stable rule pocket extracted."
    if not best_rule.empty:
        row = best_rule.iloc[0]
        rule_text = f"{row['rule']} (n={int(row['n'])}, useful={float(row['useful_rate']):.1%}, mean_adv={float(row['mean_advantage']):+.3f})"
    return pd.DataFrame(
        [
            {
                "claim_type": "main_result",
                "statement": (
                    "Globally frozen canonical QRC variants do not beat the globally frozen ESN on average, "
                    "but useful QRC regimes are non-random and measurable."
                ),
                "evidence": (
                    f"Validation win rate {float(val['best_qrc_win_rate']):.1%}; useful rate "
                    f"{float(val['best_qrc_useful_rate']):.1%}; mean best-QRC advantage {float(val['mean_best_qrc_advantage']):+.3f}."
                ),
            },
            {
                "claim_type": "regime_map",
                "statement": "QRC usefulness is concentrated in specific dataset-property regimes rather than broad chaos/nonlinearity classes.",
                "evidence": (
                    f"Top validation family {top_family['family']} has useful rate {float(top_family['best_qrc_useful_rate']):.1%}; "
                    f"discovery-trained validation AUC {prospective_metrics['classification_roc_auc']:.3f}."
                ),
            },
            {
                "claim_type": "anti_circularity",
                "statement": "The signal is not only a direct predictability-proxy artifact.",
                "evidence": (
                    f"Without predictability proxies: validation AUC {float(no_proxy.get('validation_roc_auc', math.nan)):.3f}, "
                    f"PR-AUC {float(no_proxy.get('validation_pr_auc', math.nan)):.3f}. Without chaos/nonlinearity/complexity: "
                    f"validation AUC {float(no_chaos.get('validation_roc_auc', math.nan)):.3f}."
                ),
            },
            {
                "claim_type": "rule_pocket",
                "statement": "The map yields readable high-usefulness pockets that can guide practitioner triage.",
                "evidence": rule_text,
            },
            {
                "claim_type": "guardrail",
                "statement": "Do not call this broad quantum advantage or a mechanism proof.",
                "evidence": "The primary comparison is protocol-local: globally frozen QRC mechanisms versus a globally frozen feature-matched ESN.",
            },
        ]
    )


def _write_markdown_report(
    *,
    out_dir: Path,
    split_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    variant_summary: pd.DataFrame,
    feature_set_robustness: pd.DataFrame,
    rule_pockets: pd.DataFrame,
    prospective_metrics: dict[str, float],
    claims: pd.DataFrame,
) -> Path:
    val = split_summary[split_summary["split"] == "validation"].iloc[0]
    top_families = family_summary[family_summary["split"] == "validation"].head(8)
    lines = [
        "# V5 Publication Analysis",
        "",
        "## Scope",
        "",
        "This package analyzes the completed v5 multi-QRC atlas. The target is `best_qrc_advantage_vs_esn`, where positive means the best globally frozen QRC variant has lower NRMSE than the globally frozen feature-matched ESN.",
        "",
        "## Validation Headline",
        "",
        f"- Rows: {int(val['n']):,}",
        f"- Best-QRC win rate: {float(val['best_qrc_win_rate']):.2%}",
        f"- Best-QRC useful rate: {float(val['best_qrc_useful_rate']):.2%}",
        f"- Mean best-QRC advantage: {float(val['mean_best_qrc_advantage']):+.4f}",
        f"- Discovery-trained validation R2: {prospective_metrics['regression_r2']:.4f}",
        f"- Discovery-trained validation ROC-AUC: {prospective_metrics['classification_roc_auc']:.4f}",
        f"- Discovery-trained validation PR-AUC: {prospective_metrics['classification_pr_auc']:.4f}",
        "",
        "## Claims And Guardrails",
        "",
    ]
    for row in claims.itertuples(index=False):
        lines.append(f"- **{row.claim_type}**: {row.statement} Evidence: {row.evidence}")
    lines.extend(["", "## Top Validation Families", "", top_families.to_markdown(index=False), "", "## Variant Summary", "", variant_summary[variant_summary["split"] == "validation"].to_markdown(index=False), "", "## Feature-Set Robustness", "", feature_set_robustness.to_markdown(index=False), "", "## Rule Pockets", "", rule_pockets.head(12).to_markdown(index=False) if not rule_pockets.empty else "No rule pockets extracted.", ""])
    path = out_dir / "V5_PUBLICATION_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_html_report(
    *,
    out_dir: Path,
    figures: list[dict[str, str]],
    split_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    variant_summary: pd.DataFrame,
    feature_set_robustness: pd.DataFrame,
    rule_pockets: pd.DataFrame,
    claims: pd.DataFrame,
    prospective_metrics: dict[str, float],
    win_threshold: float,
) -> Path:
    val = split_summary[split_summary["split"] == "validation"].iloc[0]
    cards = [
        ("Validation rows", f"{int(val['n']):,}", f"{int(val['n_families'])} families"),
        ("Best-QRC wins", f"{float(val['best_qrc_win_rate']):.1%}", "Advantage > 0"),
        ("Useful QRC", f"{float(val['best_qrc_useful_rate']):.1%}", f"Advantage >= {win_threshold:.3g}"),
        ("Meta-model", f"AUC {prospective_metrics['classification_roc_auc']:.2f}", f"R2 {prospective_metrics['regression_r2']:.2f}"),
    ]
    figure_html = "\n".join(
        f'<section class="figure"><h2>{html.escape(fig["title"])}</h2><p>{html.escape(fig["caption"])}</p><img src="{html.escape("figures/" + fig["png"])}" alt="{html.escape(fig["title"])}"></section>'
        for fig in figures
        if "png" in fig
    )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V5 QRC-vs-ESN Publication Analysis</title>
  <style>
    :root {{
      --surface: {TOKENS['surface']};
      --panel: {TOKENS['panel']};
      --ink: {TOKENS['ink']};
      --muted: {TOKENS['muted']};
      --axis: {TOKENS['axis']};
      --blue: {COLORS['blue']};
      --orange: {COLORS['orange']};
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--surface); color: var(--ink); font-family: Inter, Aptos, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.45; }}
    header {{ padding: 42px min(5vw, 64px) 26px; border-bottom: 1px solid var(--axis); background: var(--panel); }}
    main {{ padding: 28px min(5vw, 64px) 56px; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 46px); letter-spacing: 0; }}
    h2 {{ margin: 0 0 6px; font-size: 20px; letter-spacing: 0; }}
    p {{ margin: 0 0 12px; color: var(--muted); max-width: 960px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; margin-top: 24px; }}
    .card {{ border: 1px solid var(--axis); background: var(--panel); padding: 16px; border-radius: 8px; }}
    .card span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .card strong {{ display: block; font-size: 28px; margin: 4px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; align-items: start; }}
    .figure {{ background: var(--panel); border: 1px solid var(--axis); border-radius: 8px; padding: 16px; }}
    .figure img {{ display: block; width: 100%; height: auto; border: 1px solid #EEF0F5; background: white; }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0 26px; font-size: 13px; background: var(--panel); }}
    th, td {{ border-bottom: 1px solid #E6E8F0; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .claim {{ border-left: 3px solid var(--blue); padding: 10px 14px; background: #F7F9FE; margin-bottom: 10px; }}
    .guardrail {{ border-left-color: var(--orange); }}
    @media (max-width: 900px) {{
      .cards, .grid {{ grid-template-columns: 1fr; }}
      header, main {{ padding-left: 18px; padding-right: 18px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>V5 QRC-vs-ESN Regime Atlas</h1>
    <p>Publication-facing analysis for globally frozen QRC-M, QRC-E, and QRC-D variants against a globally frozen feature-matched sparse leaky ESN. This report supports conditional regime-map claims, not broad quantum-advantage claims.</p>
    <div class="cards">
      {''.join(f'<div class="card"><span>{html.escape(k)}</span><strong>{html.escape(v)}</strong><p>{html.escape(s)}</p></div>' for k, v, s in cards)}
    </div>
  </header>
  <main>
    <section>
      <h2>Claims and guardrails</h2>
      {''.join(_claim_html(row) for row in claims.to_dict(orient='records'))}
    </section>
    <section class="grid">
      {figure_html}
    </section>
    <section>
      <h2>Validation family summary</h2>
      {_html_table(family_summary[family_summary['split'] == 'validation'].head(12))}
      <h2>Validation variant summary</h2>
      {_html_table(variant_summary[variant_summary['split'] == 'validation'])}
      <h2>Feature-set robustness</h2>
      {_html_table(feature_set_robustness)}
      <h2>Rule pockets</h2>
      {_html_table(rule_pockets.head(12)) if not rule_pockets.empty else '<p>No stable rule pockets extracted.</p>'}
    </section>
  </main>
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(content, encoding="utf-8")
    return path


def _claim_html(row: dict[str, Any]) -> str:
    cls = "claim guardrail" if row.get("claim_type") == "guardrail" else "claim"
    return f'<div class="{cls}"><strong>{html.escape(str(row.get("claim_type", ""))).replace("_", " ").title()}</strong><p>{html.escape(str(row.get("statement", "")))} {html.escape(str(row.get("evidence", "")))}</p></div>'


def _html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4g}")
    return display.to_html(index=False, escape=True)


def _use_chart_theme() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
        },
    )
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def _add_header(fig: Any, title: str, subtitle: str) -> None:
    title = textwrap.fill(title, width=78, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=118, break_long_words=False)
    fig.subplots_adjust(top=0.86)
    fig.text(0.02, 0.985, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.02, 0.948, subtitle, ha="left", va="top", fontsize=9.2, color=TOKENS["muted"])


def _save_figure(fig: Any, out_dir: Path, stem: str, formats: tuple[str, ...], caption: str) -> dict[str, str]:
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
    paths: dict[str, str] = {"title": stem.replace("_", " ").title(), "caption": caption}
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=220 if fmt == "png" else None, bbox_inches="tight")
        paths[fmt] = path.name
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def _normalize_formats(formats: Iterable[str]) -> tuple[str, ...]:
    out = []
    for fmt in formats:
        val = str(fmt).lower().lstrip(".")
        if val not in {"png", "pdf", "svg"}:
            raise ValueError(f"unsupported figure format: {fmt}")
        if val not in out:
            out.append(val)
    return tuple(out or ["png"])


def _clean_axes(ax: Any) -> None:
    ax.grid(True, color=TOKENS["grid"], linewidth=0.8)
    ax.tick_params(colors=TOKENS["muted"], labelsize=8.5)
    ax.xaxis.label.set_color(TOKENS["ink"])
    ax.yaxis.label.set_color(TOKENS["ink"])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TOKENS["axis"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _textbox() -> dict[str, Any]:
    return {"facecolor": TOKENS["panel"], "edgecolor": TOKENS["axis"], "boxstyle": "round,pad=0.26", "linewidth": 0.6, "alpha": 0.94}


def _advantage_cmap() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("qrc_advantage", [COLORS["orange"], TOKENS["panel"], COLORS["blue"]])


def _feature_cmap() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("qrc_feature", [COLORS["orange_light"], TOKENS["panel"], COLORS["blue_light"]])


def _feature_label(feature: str) -> str:
    return str(feature).replace("ext_", "").replace("_", " ").title()


def _feature_labels(features: Iterable[str]) -> list[str]:
    return [_feature_label(str(f)) for f in features]


def _linear_fit_with_ci(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.unique(x).size < 2:
        return None
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_fit = X @ beta
    residual = y - y_fit
    dof = max(int(x.size) - 2, 1)
    sigma = float(np.sqrt(np.sum(residual**2) / dof))
    x_line = np.linspace(0.0, 1.0, 160)
    X_line = np.column_stack([np.ones_like(x_line), x_line])
    y_hat = X_line @ beta
    x_bar = float(np.mean(x))
    sxx = float(np.sum((x - x_bar) ** 2))
    if sxx <= 1e-12:
        return None
    se = sigma * np.sqrt((1.0 / x.size) + ((x_line - x_bar) ** 2 / sxx))
    ci = 1.96 * se
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 1e-12 else math.nan
    return x_line, y_hat, y_hat - ci, y_hat + ci, r2


def _xerr(mean: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.vstack([np.maximum(mean - low, 0.0), np.maximum(high - mean, 0.0)])


def _robust_scale(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    return iqr if iqr > 1e-12 else float(np.nanstd(arr) or 1.0)


def _bootstrap_stat(values: np.ndarray, *, n_bootstraps: int, seed: int, stat: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or n_bootstraps <= 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(int(n_bootstraps), values.size))
    return np.asarray([stat(values[i]) for i in idx], dtype=float)


def _finite(values: Any) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def _safe_median(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else math.nan


def _percentile(values: Any, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else math.nan


def _safe_corr(x: Any, y: Any) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3 or np.nanstd(x_arr) <= 1e-12 or np.nanstd(y_arr) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _finite_or_nan(value: Any) -> float:
    out = _float_or_nan(value)
    return out if math.isfinite(out) else math.nan


def _row_by_name(df: pd.DataFrame, col: str, value: str) -> dict[str, Any]:
    rows = df[df[col] == value]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if not isinstance(obj, (list, tuple, dict, str)) else False:
        return None
    return obj
