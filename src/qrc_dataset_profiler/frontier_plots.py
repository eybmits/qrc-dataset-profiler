"""Publication figures for the frontier conditional-QRC regime map."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from qrc_dataset_profiler.analysis import load_catalog
from qrc_dataset_profiler.frontier import FRONTIER_ATLAS_VERSION, materialize_frontier_features
from qrc_dataset_profiler.meta_model import fit_meta_model
from qrc_dataset_profiler.spec import FRONTIER_TIER_A_FIELDS
from qrc_dataset_profiler.visual_suite import (
    AXIS,
    BLUE,
    CATEGORY_COLORS,
    FAMILY_COLORS,
    GRID,
    INK,
    MUTED,
    ORANGE,
    PANEL,
    SURFACE,
    _advantage_cmap,
    _clean_axes,
    _feature_labels,
    _figure_header,
    _linear_fit_with_ci,
    _plot_context,
    _save_figure,
)


PLOT_VERSION = "frontier-publication-plots-v1"


def run_frontier_publication_plots(
    *,
    discovery_table: Path,
    validation_table: Path,
    discovery_analysis_dir: Path,
    validation_analysis_dir: Path,
    out_dir: Path,
    seed: int = 0,
    win_threshold: float = 0.05,
    formats: Iterable[str] = ("png", "pdf"),
) -> dict[str, Any]:
    """Write deterministic frontier figures and prospective meta-model outputs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = _normalize_formats(formats)

    discovery = _load_split(discovery_table, "discovery")
    validation = _load_split(validation_table, "validation")
    result = fit_meta_model(discovery, seed=seed, win_threshold=win_threshold, feature_fields=FRONTIER_TIER_A_FIELDS)
    validation_pred, prospective = _predict_validation(discovery, validation, result, seed=seed, win_threshold=win_threshold)
    combined = pd.concat([discovery, validation_pred], ignore_index=True, sort=False)

    importances = result.ranked_importances.copy()
    validation_rules = _load_optional(validation_analysis_dir / "frontier_rule_table.csv")
    grouped_validation = _load_optional(validation_analysis_dir / "frontier_grouped_validation.csv")
    split_metrics = _split_metric_table(
        discovery_analysis_dir=discovery_analysis_dir,
        validation_analysis_dir=validation_analysis_dir,
        discovery=discovery,
        validation=validation_pred,
        prospective=prospective,
        win_threshold=win_threshold,
    )
    family = _family_summary(validation_pred, win_threshold=win_threshold)

    prediction_path = out_dir / "frontier_prospective_predictions.csv"
    validation_pred.to_csv(prediction_path, index=False)
    split_metrics.to_csv(out_dir / "frontier_publication_metrics.csv", index=False)
    family.to_csv(out_dir / "frontier_validation_family_summary.csv", index=False)
    importances.to_csv(out_dir / "frontier_discovery_trained_importances.csv", index=False)

    figures: list[dict[str, str]] = []
    with _plot_context():
        figures.append(_write_regime_map(validation_pred, family, out_dir, fmt, win_threshold=win_threshold))
        figures.append(_write_prospective_meta(validation_pred, importances, split_metrics, grouped_validation, out_dir, fmt, win_threshold=win_threshold))
        figures.append(_write_feature_regressions(validation_pred, importances, out_dir, fmt, win_threshold=win_threshold))
        figures.append(
            _write_rules_and_validation(
                discovery=discovery,
                validation=validation_pred,
                family=family,
                rules=validation_rules,
                grouped=grouped_validation,
                out_dir=out_dir,
                formats=fmt,
                win_threshold=win_threshold,
            )
        )

    report = _write_report(
        out_dir=out_dir,
        discovery=discovery,
        validation=validation_pred,
        split_metrics=split_metrics,
        family=family,
        prospective=prospective,
        win_threshold=win_threshold,
    )
    _write_html_index(out_dir, figures, split_metrics, family)

    manifest = {
        "analysis_version": PLOT_VERSION,
        "frontier_atlas_version": FRONTIER_ATLAS_VERSION,
        "inputs": {
            "discovery_table": str(discovery_table),
            "validation_table": str(validation_table),
            "discovery_analysis_dir": str(discovery_analysis_dir),
            "validation_analysis_dir": str(validation_analysis_dir),
        },
        "n_discovery": int(len(discovery)),
        "n_validation": int(len(validation_pred)),
        "n_features_declared": int(len(FRONTIER_TIER_A_FIELDS)),
        "n_features_used": int(len(result.features_used)),
        "prospective_metrics": prospective,
        "top_features": importances["feature"].head(10).astype(str).tolist() if not importances.empty else [],
        "figures": figures,
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()) + ["frontier_publication_plots_manifest.json"],
        "report": report.name,
        "claim_boundary": (
            "Figures support a conditional, protocol-local regime-map claim for frozen standard_v3 QRC vs ESN. "
            "They do not establish broad average QRC superiority or a fundamental quantum advantage."
        ),
    }
    (out_dir / "frontier_publication_plots_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def _load_split(path: Path, split: str) -> pd.DataFrame:
    df = materialize_frontier_features(load_catalog(path))
    df = df.copy()
    df["evaluation_split"] = split
    df["qrc_advantage"] = pd.to_numeric(df["qrc_advantage"], errors="coerce")
    df = df[np.isfinite(df["qrc_advantage"].to_numpy(dtype=float))].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path} has no finite qrc_advantage rows")
    return df


def _predict_validation(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    result,
    *,
    seed: int,
    win_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    X_train = np.asarray(result.X, dtype=float)
    y_train = np.asarray(result.y, dtype=float)
    X_val = _transform_with_result(validation, result)
    y_val = pd.to_numeric(validation["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    useful_val = y_val > float(win_threshold)

    if X_train.size == 0 or X_val.size == 0:
        raise ValueError("no usable features for prospective frontier prediction")

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
    y_train_bin = y_train > float(win_threshold)
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

    out["actual_usefulness_label"] = np.where(out["qrc_advantage"] >= float(win_threshold), "qrc_useful", np.where(out["qrc_advantage"] >= -float(win_threshold), "near_tie", "baseline_preferred"))
    out["abs_prediction_error"] = (out["qrc_advantage"] - out["predicted_qrc_advantage"]).abs()
    return out, metrics


def _transform_with_result(df: pd.DataFrame, result) -> np.ndarray:
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


def _split_metric_table(
    *,
    discovery_analysis_dir: Path,
    validation_analysis_dir: Path,
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    prospective: dict[str, float],
    win_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, analysis_dir, data in (
        ("discovery_cv", discovery_analysis_dir, discovery),
        ("validation_cv", validation_analysis_dir, validation),
    ):
        row = _load_meta_summary(analysis_dir / "frontier_meta_summary.csv")
        row["metric_scope"] = split
        row["n_rows"] = int(len(data))
        row["qrc_win_rate"] = float((pd.to_numeric(data["qrc_advantage"], errors="coerce") > 0.0).mean())
        row["qrc_useful_rate"] = float((pd.to_numeric(data["qrc_advantage"], errors="coerce") >= float(win_threshold)).mean())
        rows.append(row)
    rows.append(
        {
            "metric_scope": "discovery_train_to_validation",
            "n_rows": int(len(validation)),
            "qrc_win_rate": float((pd.to_numeric(validation["qrc_advantage"], errors="coerce") > 0.0).mean()),
            "qrc_useful_rate": float((pd.to_numeric(validation["qrc_advantage"], errors="coerce") >= float(win_threshold)).mean()),
            "mean_advantage": float(pd.to_numeric(validation["qrc_advantage"], errors="coerce").mean()),
            "median_advantage": float(pd.to_numeric(validation["qrc_advantage"], errors="coerce").median()),
            "regression_r2_mean": prospective["regression_r2"],
            "regression_mae_mean": prospective["regression_mae"],
            "classification_roc_auc_mean": prospective["classification_roc_auc"],
            "classification_pr_auc_mean": prospective["classification_pr_auc"],
            "classification_brier_mean": prospective["classification_brier"],
            "top_features": "",
            "notes": "trained on discovery, scored once on validation",
        }
    )
    return pd.DataFrame(rows)


def _load_meta_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = load_catalog(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _family_summary(df: pd.DataFrame, *, win_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, sub in df.groupby("family", sort=True):
        adv = pd.to_numeric(sub["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
        adv = adv[np.isfinite(adv)]
        if adv.size == 0:
            continue
        mean, low, high = _bootstrap_mean_ci(adv, seed=17 + len(rows))
        rows.append(
            {
                "family": str(family),
                "n": int(adv.size),
                "qrc_win_rate": float((adv > 0.0).mean()),
                "qrc_useful_rate": float((adv >= float(win_threshold)).mean()),
                "mean_qrc_advantage": float(np.mean(adv)),
                "median_qrc_advantage": float(np.median(adv)),
                "mean_ci_low": low,
                "mean_ci_high": high,
            }
        )
    return pd.DataFrame(rows).sort_values(["qrc_useful_rate", "mean_qrc_advantage"], ascending=[False, False]).reset_index(drop=True)


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int = 800) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _write_regime_map(validation: pd.DataFrame, family: pd.DataFrame, out_dir: Path, formats: tuple[str, ...], *, win_threshold: float) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.gridspec import GridSpec

    stem = "frontier_01_validation_regime_map"
    coords = _feature_coordinates(validation)
    y = pd.to_numeric(validation["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y)
    vlim = float(np.nanpercentile(np.abs(y[finite]), 98)) if finite.any() else 1.0
    vlim = max(vlim, 0.10)

    fig = plt.figure(figsize=(14.6, 8.2), facecolor=SURFACE)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.5, 1.5, 1.0], height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.36)
    ax_map = fig.add_subplot(gs[:, :2])
    ax_family = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, 2])

    sc = ax_map.scatter(
        coords[:, 0],
        coords[:, 1],
        c=y,
        cmap=_advantage_cmap(),
        norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim),
        s=13,
        alpha=0.80,
        linewidths=0,
        rasterized=True,
    )
    useful = y >= float(win_threshold)
    if np.any(useful):
        ax_map.scatter(coords[useful, 0], coords[useful, 1], facecolors="none", edgecolors=INK, s=34, linewidths=0.65, label="QRC useful")
    _annotate_centroids(ax_map, validation, coords)
    ax_map.axhline(0.0, color=GRID, linewidth=0.7, zorder=0)
    ax_map.axvline(0.0, color=GRID, linewidth=0.7, zorder=0)
    ax_map.set_xlabel("30-feature map PC1")
    ax_map.set_ylabel("30-feature map PC2")
    ax_map.set_title("All validation datasets in measured-property space", loc="left", pad=10)
    ax_map.legend(frameon=False, loc="lower right")
    cbar = fig.colorbar(sc, ax=ax_map, orientation="horizontal", fraction=0.044, pad=0.075, aspect=36)
    cbar.set_label("QRC advantage vs matched ESN")

    fam = family.sort_values("qrc_useful_rate", ascending=True)
    y_pos = np.arange(len(fam))
    ax_family.barh(y_pos, fam["qrc_useful_rate"], color=[FAMILY_COLORS.get(str(f), BLUE) for f in fam["family"]], height=0.68)
    ax_family.set_yticks(y_pos)
    ax_family.set_yticklabels(fam["family"].str.replace("_", " "))
    ax_family.set_xlabel("Useful rate")
    ax_family.set_xlim(0.0, max(0.35, float(fam["qrc_useful_rate"].max()) * 1.25))
    ax_family.set_title("Validation useful pockets by family", loc="left", pad=10)
    _clean_axes(ax_family)

    bins = np.linspace(float(np.nanpercentile(y, 1)), float(np.nanpercentile(y, 99)), 38)
    ax_hist.hist(y, bins=bins, color="#BFC6D8", edgecolor="white", linewidth=0.45)
    ax_hist.axvline(0.0, color=INK, linewidth=0.9, label="tie")
    ax_hist.axvline(win_threshold, color=BLUE, linewidth=0.9, linestyle="--", label="useful")
    ax_hist.set_xlabel("QRC advantage")
    ax_hist.set_ylabel("Datasets")
    ax_hist.set_title("Validation outcome distribution", loc="left", pad=10)
    ax_hist.legend(frameon=False)
    _clean_axes(ax_hist)

    useful_n = int(useful.sum())
    win_n = int((y > 0.0).sum())
    _figure_header(
        fig,
        "Prospective Validation Regime Map",
        f"All {len(validation):,} validation datasets are plotted; {win_n:,} are QRC wins and {useful_n:,} pass the +{win_threshold:.2f} useful threshold.",
    )
    return _save_figure(fig, out_dir, stem, formats, "All validation rows in 30-feature space with family useful rates and advantage distribution.")


def _write_prospective_meta(
    validation: pd.DataFrame,
    importances: pd.DataFrame,
    split_metrics: pd.DataFrame,
    grouped: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
    *,
    win_threshold: float,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    stem = "frontier_02_prospective_meta_model"
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), facecolor=SURFACE)
    ax_pred, ax_cal, ax_imp, ax_group = axes.ravel()

    _plot_prediction_panel(ax_pred, validation, split_metrics, win_threshold=win_threshold)
    _plot_calibration_panel(ax_cal, validation, win_threshold=win_threshold)
    _plot_importances_panel(ax_imp, importances)
    _plot_grouped_metrics_panel(ax_group, grouped, split_metrics)
    _figure_header(
        fig,
        "Discovery-Trained Meta-Model Evidence",
        "The model is trained on discovery rows and scored on held-out validation rows; grouped-CV diagnostics show how far the claim can generalize.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Prospective prediction, calibration, feature importance, and grouped validation diagnostics.")


def _plot_prediction_panel(ax, validation: pd.DataFrame, split_metrics: pd.DataFrame, *, win_threshold: float) -> None:
    actual = pd.to_numeric(validation["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(validation["predicted_qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    label = validation["actual_usefulness_label"].astype(str)
    colors = [CATEGORY_COLORS.get(v, MUTED) for v in label]
    ax.scatter(actual, pred, c=colors, s=16, alpha=0.70, linewidths=0, rasterized=True)
    lo = float(np.nanpercentile(np.r_[actual, pred], 1))
    hi = float(np.nanpercentile(np.r_[actual, pred], 99))
    pad = 0.05 * max(hi - lo, 1e-6)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=0.9)
    fit = _fit_line(actual, pred)
    if fit is not None:
        x_line, y_line = fit
        ax.plot(x_line, y_line, color=ORANGE, linewidth=1.15)
    ax.axhline(win_threshold, color=BLUE, linestyle="--", linewidth=0.75)
    ax.axvline(win_threshold, color=BLUE, linestyle="--", linewidth=0.75)
    row = split_metrics[split_metrics["metric_scope"] == "discovery_train_to_validation"].iloc[0]
    ax.text(
        0.03,
        0.97,
        f"R2={float(row['regression_r2_mean']):.3f}\nMAE={float(row['regression_mae_mean']):.3f}\nAUC={float(row['classification_roc_auc_mean']):.3f}\nPR={float(row['classification_pr_auc_mean']):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=INK,
        bbox={"facecolor": PANEL, "edgecolor": AXIS, "boxstyle": "round,pad=0.25", "linewidth": 0.65},
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed QRC advantage")
    ax.set_ylabel("Predicted QRC advantage")
    ax.set_title("Prospective regression on validation", loc="left", pad=10)
    _clean_axes(ax)


def _plot_calibration_panel(ax, validation: pd.DataFrame, *, win_threshold: float) -> None:
    prob = pd.to_numeric(validation["predicted_prob_qrc_useful"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(validation["qrc_advantage"], errors="coerce").to_numpy(dtype=float) >= float(win_threshold)
    mask = np.isfinite(prob)
    if mask.sum() < 10:
        ax.text(0.5, 0.5, "Classifier unavailable", ha="center", va="center", transform=ax.transAxes, color=MUTED)
    else:
        qs = np.linspace(0, 1, 11)
        edges = np.unique(np.quantile(prob[mask], qs))
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = mask & (prob >= lo) & (prob <= hi if hi == edges[-1] else prob < hi)
            if idx.sum() == 0:
                continue
            rows.append((float(prob[idx].mean()), float(y[idx].mean()), int(idx.sum())))
        if rows:
            x, rate, n = zip(*rows)
            ax.plot([0, 1], [0, 1], color=INK, linewidth=0.8, linestyle=":")
            ax.scatter(x, rate, s=np.clip(np.asarray(n, dtype=float) * 0.8, 18, 120), color=BLUE, alpha=0.82, edgecolors="white", linewidths=0.4)
            ax.plot(x, rate, color=BLUE, linewidth=1.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted useful probability")
    ax.set_ylabel("Observed useful rate")
    ax.set_title("Validation calibration by probability decile", loc="left", pad=10)
    _clean_axes(ax)


def _plot_importances_panel(ax, importances: pd.DataFrame) -> None:
    if importances.empty:
        ax.text(0.5, 0.5, "No importances", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        return
    top = importances.head(12).iloc[::-1]
    y = np.arange(len(top))
    colors = [BLUE if str(v).lower() == "positive" else ORANGE if str(v).lower() == "negative" else MUTED for v in top.get("direction", [""] * len(top))]
    ax.barh(y, pd.to_numeric(top["importance_mean"], errors="coerce"), color=colors, height=0.72)
    if "importance_std" in top.columns:
        mean = pd.to_numeric(top["importance_mean"], errors="coerce").to_numpy(dtype=float)
        std = pd.to_numeric(top["importance_std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.errorbar(mean, y, xerr=std, fmt="none", ecolor=INK, linewidth=0.7, capsize=2.0)
    ax.set_yticks(y)
    ax.set_yticklabels(_feature_labels(top["feature"]))
    ax.set_xlabel("Permutation importance")
    ax.set_title("Discovery-trained feature importance", loc="left", pad=10)
    _clean_axes(ax)


def _plot_grouped_metrics_panel(ax, grouped: pd.DataFrame, split_metrics: pd.DataFrame) -> None:
    rows = []
    if not grouped.empty:
        for row in grouped.itertuples(index=False):
            data = row._asdict()
            rows.append((str(data.get("group_col")), data.get("regression_r2_mean", math.nan), data.get("classification_roc_auc_mean", math.nan), data.get("classification_pr_auc_mean", math.nan)))
    prospective = split_metrics[split_metrics["metric_scope"] == "discovery_train_to_validation"].iloc[0]
    rows.insert(0, ("prospective", prospective["regression_r2_mean"], prospective["classification_roc_auc_mean"], prospective["classification_pr_auc_mean"]))
    x = np.arange(len(rows))
    width = 0.24
    labels = [r[0].replace("_", " ") for r in rows]
    r2 = [float(r[1]) for r in rows]
    auc = [float(r[2]) for r in rows]
    pr = [float(r[3]) for r in rows]
    ax.axhline(0.0, color=INK, linewidth=0.7)
    ax.bar(x - width, r2, width=width, color=BLUE, label="R2")
    ax.bar(x, auc, width=width, color="#B8A037", label="ROC-AUC")
    ax.bar(x + width, pr, width=width, color=ORANGE, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    finite_vals = [v for arr in (r2, auc, pr) for v in arr if math.isfinite(v)]
    ax.set_ylim(min(-0.2, min(finite_vals) - 0.08), max(0.9, max(finite_vals) + 0.08))
    ax.set_ylabel("Score")
    ax.set_title("Generalization stress tests", loc="left", pad=10)
    ax.legend(frameon=False, loc="lower right")
    _clean_axes(ax)


def _write_feature_regressions(validation: pd.DataFrame, importances: pd.DataFrame, out_dir: Path, formats: tuple[str, ...], *, win_threshold: float) -> dict[str, str]:
    import matplotlib.pyplot as plt

    stem = "frontier_03_all_points_feature_regressions"
    features = [f for f in importances["feature"].astype(str).head(8).tolist() if f in validation.columns] if not importances.empty else []
    if not features:
        features = [f for f in FRONTIER_TIER_A_FIELDS if f in validation.columns][:8]
    n_cols = 4
    n_rows = int(math.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.2, 4.2 * n_rows + 0.9), facecolor=SURFACE, squeeze=False)
    rng = np.random.default_rng(31)
    for ax, feature in zip(axes.ravel(), features):
        _plot_feature_regression(ax, validation, feature, rng, win_threshold=win_threshold)
    for ax in axes.ravel()[len(features) :]:
        ax.set_visible(False)
    handles = [_family_handle(f) for f in sorted(validation["family"].dropna().astype(str).unique())]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.992, 0.49), frameon=False, title="Family", fontsize=7.3)
    fig._qrc_tight_rect = (0.02, 0.02, 0.86, 0.90)
    _figure_header(
        fig,
        "All-Points Feature Regressions",
        "Each panel uses every validation row; x-axes are feature percentile ranks, with OLS fit and 95% confidence band.",
    )
    return _save_figure(fig, out_dir, stem, formats, "All validation points plotted against the top meta-model features with regression bands.")


def _plot_feature_regression(ax, data: pd.DataFrame, feature: str, rng: np.random.Generator, *, win_threshold: float) -> None:
    raw = pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float)
    adv = pd.to_numeric(data["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(raw) & np.isfinite(adv)
    plot = data.loc[mask, ["family"]].copy()
    ranks = pd.Series(raw[mask]).rank(method="average", pct=True).to_numpy(dtype=float)
    if np.unique(ranks).size < 50:
        ranks = np.clip(ranks + rng.normal(0.0, 0.0035, size=ranks.size), 0.0, 1.0)
    plot["feature_rank"] = ranks
    plot["qrc_advantage"] = adv[mask]
    for family, sub in plot.groupby("family", sort=True):
        ax.scatter(
            sub["feature_rank"],
            sub["qrc_advantage"],
            s=9,
            color=FAMILY_COLORS.get(str(family), MUTED),
            alpha=0.34,
            linewidths=0,
            rasterized=True,
        )
    fit = _linear_fit_with_ci(plot["feature_rank"].to_numpy(dtype=float), plot["qrc_advantage"].to_numpy(dtype=float))
    r2 = math.nan
    if fit is not None:
        x_line, y_hat, low, high, r2 = fit
        ax.fill_between(x_line, low, high, color="#E8EEF9", alpha=0.82, linewidth=0)
        ax.plot(x_line, y_hat, color=INK, linewidth=1.0)
    rho = pd.Series(raw[mask]).corr(pd.Series(adv[mask]), method="spearman") if int(mask.sum()) > 2 else math.nan
    ax.axhline(0.0, color=INK, linewidth=0.75)
    ax.axhline(win_threshold, color=BLUE, linewidth=0.75, linestyle="--")
    ax.text(
        0.03,
        0.96,
        f"n={int(mask.sum())}\nrho={rho:.2f}\nR2={r2:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=INK,
        bbox={"facecolor": PANEL, "edgecolor": AXIS, "boxstyle": "round,pad=0.20", "linewidth": 0.55, "alpha": 0.92},
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Feature percentile rank")
    ax.set_ylabel("QRC advantage")
    ax.set_title(_feature_labels([feature])[0], loc="left", pad=8)
    _clean_axes(ax)


def _write_rules_and_validation(
    *,
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    family: pd.DataFrame,
    rules: pd.DataFrame,
    grouped: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
    win_threshold: float,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    stem = "frontier_04_rules_and_claim_boundary"
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 8.8), facecolor=SURFACE)
    ax_rules, ax_dist, ax_split, ax_text = axes.ravel()
    _plot_rule_panel(ax_rules, rules)
    _plot_family_distribution(ax_dist, validation, family, win_threshold=win_threshold)
    _plot_split_composition(ax_split, discovery, validation, win_threshold=win_threshold)
    _plot_claim_boundary(ax_text, validation, grouped)
    _figure_header(
        fig,
        "Rule Pockets and Claim Boundary",
        "High-usefulness pockets are visible, but family-held-out generalization remains the limiting evidence for broad claims.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Rule pockets, family distributions, split agreement, and claim-boundary diagnostics.")


def _plot_rule_panel(ax, rules: pd.DataFrame) -> None:
    if rules.empty:
        ax.text(0.5, 0.5, "No rule table", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return
    data = rules.head(6).iloc[::-1]
    y = np.arange(len(data))
    colors = [BLUE if float(v) >= 0.15 else "#B8A037" if float(v) >= 0.07 else MUTED for v in data["qrc_useful_rate"]]
    ax.barh(y, data["qrc_useful_rate"], color=colors, height=0.68)
    labels = [str(v).replace(" and ", "\n") for v in data["rule"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    for yi, row in zip(y, data.itertuples(index=False)):
        ax.text(float(row.qrc_useful_rate) + 0.008, yi, f"n={int(row.n)}, win={float(row.qrc_win_rate):.0%}", va="center", fontsize=7.0, color=INK)
    ax.set_xlim(0, max(0.35, float(data["qrc_useful_rate"].max()) * 1.25))
    ax.set_xlabel("QRC-useful rate")
    ax.set_title("Validation rule pockets", loc="left", pad=10)
    _clean_axes(ax)


def _plot_family_distribution(ax, validation: pd.DataFrame, family: pd.DataFrame, *, win_threshold: float) -> None:
    order = family.sort_values("qrc_useful_rate", ascending=True)["family"].astype(str).tolist()
    data = [pd.to_numeric(validation.loc[validation["family"].astype(str) == fam, "qrc_advantage"], errors="coerce").dropna().to_numpy(dtype=float) for fam in order]
    y = np.arange(len(order))
    parts = ax.violinplot(data, positions=y, vert=False, widths=0.78, showextrema=False, showmeans=False, showmedians=False)
    for body, fam in zip(parts["bodies"], order):
        body.set_facecolor(FAMILY_COLORS.get(fam, MUTED))
        body.set_alpha(0.46)
        body.set_edgecolor("none")
    for yi, values in zip(y, data):
        if len(values) == 0:
            continue
        ax.scatter(np.median(values), yi, s=24, color=INK, zorder=4)
    ax.axvline(0.0, color=INK, linewidth=0.85)
    ax.axvline(win_threshold, color=BLUE, linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([fam.replace("_", " ") for fam in order])
    ax.set_xlabel("QRC advantage")
    ax.set_title("Family distributions on validation", loc="left", pad=10)
    _clean_axes(ax)


def _plot_split_composition(ax, discovery: pd.DataFrame, validation: pd.DataFrame, *, win_threshold: float) -> None:
    labels = ["baseline_preferred", "near_tie", "qrc_useful"]
    rows = []
    for name, data in (("discovery", discovery), ("validation", validation)):
        adv = pd.to_numeric(data["qrc_advantage"], errors="coerce")
        rows.append(
            [
                name,
                float((adv < -float(win_threshold)).mean()),
                float(((adv >= -float(win_threshold)) & (adv < float(win_threshold))).mean()),
                float((adv >= float(win_threshold)).mean()),
            ]
        )
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for i, label in enumerate(labels, start=1):
        vals = np.asarray([r[i] for r in rows], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=CATEGORY_COLORS[label], label=label.replace("_", " "), width=0.58)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of split")
    ax.set_title("Discovery/validation outcome agreement", loc="left", pad=10)
    ax.legend(frameon=False, loc="upper right")
    _clean_axes(ax)


def _plot_claim_boundary(ax, validation: pd.DataFrame, grouped: pd.DataFrame) -> None:
    ax.set_axis_off()
    adv = pd.to_numeric(validation["qrc_advantage"], errors="coerce")
    base_row = grouped[grouped["group_col"] == "base_generator"].iloc[0].to_dict() if not grouped.empty and (grouped["group_col"] == "base_generator").any() else {}
    fam_row = grouped[grouped["group_col"] == "family"].iloc[0].to_dict() if not grouped.empty and (grouped["group_col"] == "family").any() else {}
    lines = [
        ("Validation size", f"{len(validation):,} labeled datasets from the frozen target-free selection."),
        ("Average result", f"mean advantage {float(adv.mean()):+.3f}; median {float(adv.median()):+.3f}."),
        ("Selective result", f"{int((adv > 0).sum()):,} any wins; {int((adv >= 0.05).sum()):,} useful wins at +0.05."),
        ("Base-generator holdout", f"AUC {float(base_row.get('classification_roc_auc_mean', math.nan)):.3f}, R2 {float(base_row.get('regression_r2_mean', math.nan)):.3f}."),
        ("Family holdout", f"AUC {float(fam_row.get('classification_roc_auc_mean', math.nan)):.3f}, R2 {float(fam_row.get('regression_r2_mean', math.nan)):.3f}."),
        ("Claim boundary", "Dataset-property regime map, not broad average QRC superiority or fundamental quantum advantage."),
    ]
    y = 0.92
    for label, text in lines:
        ax.text(0.02, y, label.upper(), transform=ax.transAxes, fontsize=8.0, color=MUTED, weight="bold", va="top")
        ax.text(0.02, y - 0.055, text, transform=ax.transAxes, fontsize=10.0, color=INK, va="top", wrap=True)
        y -= 0.155


def _feature_coordinates(df: pd.DataFrame) -> np.ndarray:
    features = [c for c in FRONTIER_TIER_A_FIELDS if c in df.columns]
    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    if X.shape[1] == 0:
        return np.zeros((len(df), 2), dtype=float)
    Xs = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    if Xs.shape[1] == 1:
        return np.column_stack([Xs[:, 0], np.zeros(len(Xs))])
    return PCA(n_components=2, random_state=0).fit_transform(Xs)


def _annotate_centroids(ax, data: pd.DataFrame, coords: np.ndarray) -> None:
    tmp = pd.DataFrame({"family": data["family"].astype(str), "x": coords[:, 0], "y": coords[:, 1]})
    for family, row in tmp.groupby("family", sort=True)[["x", "y"]].median().iterrows():
        ax.text(
            float(row["x"]),
            float(row["y"]),
            family.replace("_", " "),
            fontsize=7.0,
            ha="center",
            va="center",
            color=INK,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": PANEL, "edgecolor": AXIS, "alpha": 0.82, "linewidth": 0.5},
        )


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.unique(x).size < 2:
        return None
    beta = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(np.nanpercentile(x, 1)), float(np.nanpercentile(x, 99)), 100)
    return xs, beta[0] * xs + beta[1]


def _family_handle(family: str):
    from matplotlib.patches import Patch

    return Patch(facecolor=FAMILY_COLORS.get(family, MUTED), edgecolor="none", label=family.replace("_", " "))


def _load_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_catalog(path)


def _normalize_formats(formats: Iterable[str]) -> tuple[str, ...]:
    out = tuple(dict.fromkeys(str(fmt).strip().lower() for fmt in formats if str(fmt).strip()))
    valid = {"png", "pdf", "svg"}
    bad = [fmt for fmt in out if fmt not in valid]
    if bad:
        raise ValueError(f"unsupported output formats: {bad}")
    return out or ("png",)


def _write_report(
    *,
    out_dir: Path,
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    split_metrics: pd.DataFrame,
    family: pd.DataFrame,
    prospective: dict[str, float],
    win_threshold: float,
) -> Path:
    adv = pd.to_numeric(validation["qrc_advantage"], errors="coerce")
    top_families = family.head(3)
    body = [
        "# Frontier Publication Plot Report",
        "",
        f"- Discovery rows: {len(discovery):,}",
        f"- Validation rows: {len(validation):,}",
        f"- Validation QRC wins: {int((adv > 0).sum()):,}",
        f"- Validation QRC-useful rows at +{win_threshold:.2f}: {int((adv >= win_threshold).sum()):,}",
        f"- Prospective regression R2: {prospective['regression_r2']:.4f}",
        f"- Prospective classification ROC-AUC: {prospective['classification_roc_auc']:.4f}",
        f"- Prospective classification PR-AUC: {prospective['classification_pr_auc']:.4f}",
        "",
        "Top validation families by useful rate:",
    ]
    for row in top_families.itertuples(index=False):
        body.append(f"- {row.family}: useful {float(row.qrc_useful_rate):.1%}, mean advantage {float(row.mean_qrc_advantage):+.3f}, n={int(row.n)}")
    body.extend(
        [
            "",
            "Claim boundary: these figures support a conditional, protocol-local regime map for frozen standard_v3 QRC vs matched ESN. They do not support broad average QRC superiority or a fundamental quantum-advantage claim.",
            "",
        ]
    )
    path = out_dir / "FRONTIER_PUBLICATION_PLOTS_REPORT.md"
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def _write_html_index(out_dir: Path, figures: list[dict[str, str]], split_metrics: pd.DataFrame, family: pd.DataFrame) -> Path:
    row = split_metrics[split_metrics["metric_scope"] == "discovery_train_to_validation"].iloc[0]
    top_family = family.iloc[0]
    fig_html = "\n".join(
        f"<section><h2>{item['title']}</h2><p>{item['caption']}</p><img src='{item.get('png', '')}' alt='{item['title']}'></section>"
        for item in figures
        if item.get("png")
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Frontier QRC Regime Map Figures</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #FCFCFD; color: #1F2430; }}
    header {{ padding: 32px 6vw 20px; background: white; border-bottom: 1px solid #D7DBE7; }}
    main {{ padding: 24px 6vw 48px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 18px; }}
    .card {{ background: white; border: 1px solid #D7DBE7; border-radius: 8px; padding: 14px; }}
    .card span {{ display: block; color: #6F768A; font-size: 12px; text-transform: uppercase; }}
    .card strong {{ display: block; margin-top: 6px; font-size: 24px; }}
    section {{ margin: 28px 0 40px; }}
    img {{ width: 100%; max-width: 1300px; display: block; border: 1px solid #D7DBE7; background: white; }}
  </style>
</head>
<body>
  <header>
    <h1>Frontier QRC Regime Map Figures</h1>
    <p>Prospective validation figures for the frozen standard_v3 QRC-vs-ESN protocol.</p>
    <div class="cards">
      <div class="card"><span>Prospective R2</span><strong>{float(row['regression_r2_mean']):.3f}</strong></div>
      <div class="card"><span>Prospective AUC</span><strong>{float(row['classification_roc_auc_mean']):.3f}</strong></div>
      <div class="card"><span>Prospective PR-AUC</span><strong>{float(row['classification_pr_auc_mean']):.3f}</strong></div>
      <div class="card"><span>Top Family</span><strong>{str(top_family['family']).replace('_', ' ')}</strong></div>
    </div>
  </header>
  <main>{fig_html}</main>
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else math.nan


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj
