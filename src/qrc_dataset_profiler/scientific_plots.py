"""Dense publication-style figures for the QRC usefulness atlas."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from qrc_dataset_profiler.visual_suite import (
    AXIS,
    BLUE,
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    FAMILY_COLORS,
    GRID,
    INK,
    MUTED,
    ORANGE,
    PANEL,
    SURFACE,
    _advantage_cmap,
    _clean_axes,
    _feature_cmap,
    _feature_labels,
    _figure_header,
    _json_safe,
    _legend_patch,
    _linear_fit_with_ci,
    _load_required,
    _normalize_formats,
    _plot_context,
    _save_figure,
    _select_regression_features,
    _short_label,
    _xerr,
)


def run_scientific_plots(
    *,
    atlas_dir: Path,
    analysis_dir: Path,
    attribution_dir: Path,
    features_dir: Path,
    sweep_catalog: Path,
    out_dir: Path,
    formats: Iterable[str] = ("png", "pdf"),
) -> dict[str, Any]:
    """Write information-dense, publication-oriented multi-panel figures."""

    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = _normalize_formats(formats)
    atlas = _load_required(atlas_dir / "qrc_usefulness_map.csv")
    family = _load_required(atlas_dir / "family_usefulness_summary.csv")
    meta = _load_required(atlas_dir / "meta_model_summary.csv")
    sweep = _load_required(sweep_catalog)
    family_ci = _load_required(analysis_dir / "family_advantage_bootstrap.csv")
    robustness = _load_required(analysis_dir / "robustness_summary.csv")
    importance = _load_required(analysis_dir / "importance_bootstrap.csv")
    attribution = _load_required(attribution_dir / "family_attribution_bootstrap.csv")
    paired_attribution = _load_required(attribution_dir / "paired_attribution.csv")
    extended = _load_required(features_dir / "extended_features_sweep.csv")

    merged = atlas.merge(sweep, on=["dataset_id", "family"], how="inner", suffixes=("", "_sweep"))
    figures: list[dict[str, str]] = []
    with _plot_context():
        figures.append(_write_fig1_atlas_dense(atlas, family, meta, family_ci, out_dir, fmt))
        figures.append(_write_fig2_feature_regressions(merged, importance, out_dir, fmt))
        figures.append(_write_fig3_family_feature_matrix(atlas, family, family_ci, merged, extended, importance, out_dir, fmt))
        figures.append(_write_fig4_model_validation(atlas, meta, robustness, importance, out_dir, fmt))
        figures.append(_write_fig5_attribution_control(attribution, paired_attribution, out_dir, fmt))

    report = _write_report(atlas, family, meta, family_ci, robustness, attribution, out_dir / "SCIENTIFIC_PLOTS_REPORT.md")
    _write_html_index(out_dir, figures, report)
    manifest = {
        "analysis_version": "scientific-plots-v1",
        "n_rows": int(len(atlas)),
        "n_families": int(family["family"].nunique()),
        "figures": figures,
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "claim_boundary": (
            "Dense scientific plots support dataset categorization and meta-model interpretation. "
            "They do not claim broad average QRC superiority or a quantum coupling mechanism."
        ),
    }
    (out_dir / "scientific_plots_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_fig1_atlas_dense(
    atlas: pd.DataFrame,
    family: pd.DataFrame,
    meta: pd.DataFrame,
    family_ci: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13.4, 8.6), facecolor=SURFACE)
    gs = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    ax_map, ax_pred, ax_hist, ax_stack, ax_ci, ax_family_pred = axes

    for label in CATEGORY_ORDER:
        sub = atlas[atlas["actual_usefulness_label"] == label]
        ax_map.scatter(sub["property_pc1"], sub["property_pc2"], s=14, color=CATEGORY_COLORS[label], alpha=0.72, edgecolors="white", linewidths=0.18, rasterized=True, label=label.replace("_", " "))
    ax_map.set_xlabel("Property PC1")
    ax_map.set_ylabel("Property PC2")
    ax_map.set_title("a  Property map, observed usefulness", loc="left")
    ax_map.legend(frameon=False, fontsize=7, loc="best")
    _clean_axes(ax_map)

    colors = [CATEGORY_COLORS.get(v, MUTED) for v in atlas["actual_usefulness_label"]]
    ax_pred.scatter(atlas["qrc_advantage"], atlas["predicted_qrc_advantage"], s=14, c=colors, alpha=0.68, edgecolors="white", linewidths=0.16, rasterized=True)
    lo = float(np.nanmin([atlas["qrc_advantage"].min(), atlas["predicted_qrc_advantage"].min()]))
    hi = float(np.nanmax([atlas["qrc_advantage"].max(), atlas["predicted_qrc_advantage"].max()]))
    ax_pred.plot([lo, hi], [lo, hi], color=INK, linewidth=0.9)
    ax_pred.axhline(0.05, color=BLUE, linewidth=0.7, linestyle="--")
    ax_pred.axvline(0.05, color=BLUE, linewidth=0.7, linestyle="--")
    row = meta.iloc[0]
    ax_pred.text(0.03, 0.96, f"CV R2={float(row['regression_r2_mean']):.2f}\nAUC={float(row['classification_roc_auc_mean']):.2f}", transform=ax_pred.transAxes, ha="left", va="top", fontsize=7.2, bbox=_textbox())
    ax_pred.set_xlabel("Observed QRC advantage")
    ax_pred.set_ylabel("Predicted QRC advantage")
    ax_pred.set_title("b  Predicted vs observed", loc="left")
    _clean_axes(ax_pred)

    bins = np.linspace(float(atlas["qrc_advantage"].min()), float(atlas["qrc_advantage"].max()), 36)
    for label in CATEGORY_ORDER:
        vals = atlas.loc[atlas["actual_usefulness_label"] == label, "qrc_advantage"]
        ax_hist.hist(vals, bins=bins, color=CATEGORY_COLORS[label], alpha=0.72, label=label.replace("_", " "))
    ax_hist.axvline(0.0, color=INK, linewidth=0.9)
    ax_hist.axvline(0.05, color=BLUE, linewidth=0.8, linestyle="--")
    ax_hist.set_xlabel("QRC advantage")
    ax_hist.set_ylabel("Rows")
    ax_hist.set_title("c  Advantage distribution", loc="left")
    _clean_axes(ax_hist)

    fam = family.sort_values("qrc_useful_rate", ascending=True)
    y = np.arange(len(fam))
    left = np.zeros(len(fam))
    for label, col in (
        ("baseline_preferred", "baseline_preferred_rate"),
        ("near_tie", "near_tie_rate"),
        ("qrc_useful", "qrc_useful_rate"),
    ):
        vals = fam[col].to_numpy(dtype=float)
        ax_stack.barh(y, vals, left=left, height=0.72, color=CATEGORY_COLORS[label], label=label.replace("_", " "))
        left += vals
    ax_stack.set_yticks(y)
    ax_stack.set_yticklabels(fam["family"].str.replace("_", " "), fontsize=7)
    ax_stack.set_xlim(0, 1)
    ax_stack.set_xlabel("Share")
    ax_stack.set_title("d  Family label composition", loc="left")
    _clean_axes(ax_stack)

    ci = family_ci[family_ci["family"] != "overall"].sort_values("mean_advantage", ascending=True)
    y = np.arange(len(ci))
    mean = ci["mean_advantage"].to_numpy(dtype=float)
    low = ci["mean_ci_low"].to_numpy(dtype=float)
    high = ci["mean_ci_high"].to_numpy(dtype=float)
    ax_ci.errorbar(mean, y, xerr=_xerr(mean, low, high), fmt="none", ecolor=MUTED, linewidth=0.9, capsize=2)
    ax_ci.scatter(mean, y, color=[BLUE if lo > 0 else ORANGE if hi < 0 else MUTED for lo, hi in zip(low, high)], s=28, edgecolors="white", linewidths=0.4)
    ax_ci.axvline(0.0, color=INK, linewidth=0.9)
    ax_ci.axvline(0.05, color=BLUE, linewidth=0.7, linestyle="--")
    ax_ci.set_yticks(y)
    ax_ci.set_yticklabels(ci["family"].str.replace("_", " "), fontsize=7)
    ax_ci.set_xlabel("Mean advantage, 95% CI")
    ax_ci.set_title("e  Bootstrap family effects", loc="left")
    _clean_axes(ax_ci)

    family_pred = atlas.groupby("family", sort=True).agg(observed=("qrc_advantage", "mean"), predicted=("predicted_qrc_advantage", "mean"), n=("qrc_advantage", "size")).reset_index()
    ax_family_pred.scatter(family_pred["observed"], family_pred["predicted"], s=np.clip(family_pred["n"], 20, 180), color=[FAMILY_COLORS.get(f, MUTED) for f in family_pred["family"]], alpha=0.84, edgecolors="white", linewidths=0.6)
    lo = float(np.nanmin([family_pred["observed"].min(), family_pred["predicted"].min()]))
    hi = float(np.nanmax([family_pred["observed"].max(), family_pred["predicted"].max()]))
    ax_family_pred.plot([lo, hi], [lo, hi], color=INK, linewidth=0.9)
    for _, r in family_pred.iterrows():
        ax_family_pred.text(float(r["observed"]), float(r["predicted"]), _short_label(r["family"], 16), fontsize=6.2, color=INK)
    ax_family_pred.set_xlabel("Family mean observed advantage")
    ax_family_pred.set_ylabel("Family mean predicted advantage")
    ax_family_pred.set_title("f  Family-level meta-model alignment", loc="left")
    _clean_axes(ax_family_pred)

    _figure_header(fig, "Figure 1. Dense atlas summary", "All rows, usefulness labels, family effects, and meta-model alignment for the 1000-row sweep.")
    return _save_figure(fig, out_dir, "fig1_dense_atlas_summary", formats, "Dense atlas summary with row-level map, distributions, family intervals, and family prediction alignment.")


def _write_fig2_feature_regressions(
    merged: pd.DataFrame,
    importance: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    features = _select_regression_features(merged, importance, limit=12)
    fig, axes = plt.subplots(3, 4, figsize=(14.4, 9.1), facecolor=SURFACE, sharey=True)
    rng = np.random.default_rng(23)
    for ax, feature in zip(axes.ravel(), features):
        _regression_panel(ax, merged, feature, rng)
    for ax in axes.ravel()[len(features) :]:
        ax.set_visible(False)
    handles = [_legend_patch(FAMILY_COLORS.get(f, MUTED), str(f).replace("_", " ")) for f in sorted(merged["family"].dropna().unique())]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.01), fontsize=7)
    fig._qrc_tight_rect = (0.02, 0.06, 0.98, 0.90)
    _figure_header(fig, "Figure 2. All-point feature regressions", "Each point is one dataset row; x-axes are percentile ranks of top meta-model features; black lines are OLS fits with 95% confidence bands.")
    return _save_figure(fig, out_dir, "fig2_dense_feature_regressions", formats, "All-point feature regressions with fitted trends and confidence bands.")


def _write_fig3_family_feature_matrix(
    atlas: pd.DataFrame,
    family: pd.DataFrame,
    family_ci: pd.DataFrame,
    merged: pd.DataFrame,
    extended: pd.DataFrame,
    importance: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    ext_join = atlas[["dataset_id", "family", "qrc_advantage"]].merge(extended, on="dataset_id", how="inner", suffixes=("", "_ext"))
    core_features = [f for f in importance.sort_values("importance_mean", ascending=False)["feature"].tolist() if f in merged.columns][:9]
    ext_features = _top_extended_features(ext_join, limit=7)
    matrix_parts: list[pd.DataFrame] = []
    labels: list[str] = []
    if core_features:
        core = merged[["family", *core_features]].copy()
        matrix_parts.append(core.groupby("family").mean(numeric_only=True))
        labels.extend(core_features)
    if ext_features:
        ext = ext_join[["family", *ext_features]].copy()
        matrix_parts.append(ext.groupby("family").mean(numeric_only=True))
        labels.extend(ext_features)
    mat = pd.concat(matrix_parts, axis=1) if matrix_parts else pd.DataFrame()
    order = family.sort_values("qrc_useful_rate", ascending=False)["family"].tolist()
    mat = mat.reindex(order)
    mat_z = mat.apply(lambda s: (s - s.median()) / _robust_scale(s), axis=0).clip(-2.5, 2.5)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 7.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [2.3, 0.65, 0.9]})
    ax_heat, ax_rate, ax_ci = axes
    image = ax_heat.imshow(mat_z.to_numpy(dtype=float), aspect="auto", cmap=_feature_cmap(), norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5), interpolation="nearest")
    ax_heat.set_yticks(np.arange(len(mat_z.index)))
    ax_heat.set_yticklabels([str(v).replace("_", " ") for v in mat_z.index], fontsize=7)
    ax_heat.set_xticks(np.arange(len(mat_z.columns)))
    ax_heat.set_xticklabels(_feature_labels(mat_z.columns), rotation=45, ha="right", fontsize=6.7)
    ax_heat.set_title("a  Family-average standardized descriptors", loc="left")
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_color(AXIS)
    cb = fig.colorbar(image, ax=ax_heat, fraction=0.032, pad=0.012)
    cb.set_label("Robust z-score")
    cb.outline.set_edgecolor(AXIS)

    fam = family.set_index("family").reindex(order)
    y = np.arange(len(fam))
    ax_rate.barh(y, fam["qrc_useful_rate"], color=BLUE, height=0.72)
    ax_rate.set_yticks(y)
    ax_rate.set_yticklabels([])
    ax_rate.invert_yaxis()
    ax_rate.set_xlabel("Useful rate")
    ax_rate.set_title("b  Useful", loc="left")
    _clean_axes(ax_rate)

    ci = family_ci[family_ci["family"].isin(order)].set_index("family").reindex(order)
    mean = ci["mean_advantage"].to_numpy(dtype=float)
    low = ci["mean_ci_low"].to_numpy(dtype=float)
    high = ci["mean_ci_high"].to_numpy(dtype=float)
    ax_ci.errorbar(mean, y, xerr=_xerr(mean, low, high), fmt="none", ecolor=MUTED, linewidth=0.9, capsize=2)
    ax_ci.scatter(mean, y, color=[BLUE if lo > 0 else ORANGE if hi < 0 else MUTED for lo, hi in zip(low, high)], s=25, edgecolors="white", linewidths=0.4)
    ax_ci.axvline(0, color=INK, linewidth=0.8)
    ax_ci.set_yticks(y)
    ax_ci.set_yticklabels([])
    ax_ci.invert_yaxis()
    ax_ci.set_xlabel("Mean advantage")
    ax_ci.set_title("c  Effect", loc="left")
    _clean_axes(ax_ci)

    _figure_header(fig, "Figure 3. Family-property structure", "Core and Tier-B descriptors are shown as family-average robust z-scores alongside useful rates and bootstrap advantage intervals.")
    return _save_figure(fig, out_dir, "fig3_dense_family_feature_matrix", formats, "Family-level core and extended feature matrix with useful rates and effect intervals.")


def _write_fig4_model_validation(
    atlas: pd.DataFrame,
    meta: pd.DataFrame,
    robustness: pd.DataFrame,
    importance: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), facecolor=SURFACE)
    ax_pred, ax_resid, ax_cal, ax_conf, ax_robust, ax_imp = axes.ravel()

    colors = [CATEGORY_COLORS.get(v, MUTED) for v in atlas["actual_usefulness_label"]]
    ax_pred.scatter(atlas["qrc_advantage"], atlas["predicted_qrc_advantage"], s=13, c=colors, alpha=0.68, edgecolors="white", linewidths=0.16, rasterized=True)
    lo = float(np.nanmin([atlas["qrc_advantage"].min(), atlas["predicted_qrc_advantage"].min()]))
    hi = float(np.nanmax([atlas["qrc_advantage"].max(), atlas["predicted_qrc_advantage"].max()]))
    ax_pred.plot([lo, hi], [lo, hi], color=INK, linewidth=0.9)
    ax_pred.set_xlabel("Observed advantage")
    ax_pred.set_ylabel("Predicted advantage")
    ax_pred.set_title("a  Prediction scatter", loc="left")
    _clean_axes(ax_pred)

    resid = atlas["qrc_advantage"] - atlas["predicted_qrc_advantage"]
    ax_resid.scatter(atlas["predicted_qrc_advantage"], resid, s=13, c=colors, alpha=0.65, edgecolors="white", linewidths=0.16, rasterized=True)
    ax_resid.axhline(0, color=INK, linewidth=0.8)
    ax_resid.set_xlabel("Predicted advantage")
    ax_resid.set_ylabel("Residual")
    ax_resid.set_title("b  Residuals", loc="left")
    _clean_axes(ax_resid)

    cal = _calibration_table(atlas, n_bins=10)
    ax_cal.errorbar(cal["mean_prob"], cal["observed_rate"], yerr=cal["se"], fmt="o-", color=BLUE, ecolor=MUTED, capsize=2, linewidth=1.0)
    ax_cal.plot([0, 1], [0, 1], color=INK, linewidth=0.8, linestyle=":")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel("Mean predicted P(qrc_useful)")
    ax_cal.set_ylabel("Observed useful rate")
    ax_cal.set_title("c  Calibration by probability bin", loc="left")
    _clean_axes(ax_cal)

    matrix, actual_labels, pred_labels = _confusion_matrix(atlas)
    cmap = LinearSegmentedColormap.from_list("conf", [PANEL, "#CEDFFE", BLUE], N=256)
    ax_conf.imshow(matrix, cmap=cmap, aspect="auto")
    ax_conf.set_xticks(np.arange(len(pred_labels)))
    ax_conf.set_xticklabels([v.replace("_", "\n") for v in pred_labels], fontsize=6.5)
    ax_conf.set_yticks(np.arange(len(actual_labels)))
    ax_conf.set_yticklabels([v.replace("_", "\n") for v in actual_labels], fontsize=6.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_conf.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7, color=INK)
    ax_conf.set_xlabel("Predicted label")
    ax_conf.set_ylabel("Actual label")
    ax_conf.set_title("d  Label confusion", loc="left")

    labels = [_feature_set_label(v) for v in robustness["feature_set"]]
    x = np.arange(len(labels))
    ax_robust.plot(x, robustness["regression_r2_mean"], marker="o", color=BLUE, linewidth=1.2, label="CV R2")
    ax_robust.plot(x, robustness["classification_roc_auc_mean"], marker="s", color="#B8A037", linewidth=1.2, label="AUC")
    ax_robust.set_xticks(x)
    ax_robust.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax_robust.set_ylim(0.40, 0.92)
    ax_robust.set_ylabel("Score")
    ax_robust.set_title("e  Anti-circularity", loc="left")
    ax_robust.legend(frameon=False, fontsize=7, loc="lower right")
    _clean_axes(ax_robust)

    imp = importance.sort_values("importance_mean", ascending=False).head(12).iloc[::-1]
    y = np.arange(len(imp))
    means = imp["importance_mean"].to_numpy(dtype=float)
    low = imp["ci_low"].to_numpy(dtype=float) if "ci_low" in imp else means
    high = imp["ci_high"].to_numpy(dtype=float) if "ci_high" in imp else means
    ax_imp.barh(y, means, color=[ORANGE if d == "negative" else BLUE for d in imp.get("direction", ["positive"] * len(imp))], height=0.68)
    ax_imp.errorbar(means, y, xerr=_xerr(means, low, high), fmt="none", ecolor=INK, linewidth=0.7, capsize=1.8)
    ax_imp.set_yticks(y)
    ax_imp.set_yticklabels(_feature_labels(imp["feature"]), fontsize=7)
    ax_imp.set_xlabel("Permutation importance")
    ax_imp.set_title("f  Importance intervals", loc="left")
    _clean_axes(ax_imp)

    _figure_header(fig, "Figure 4. Meta-model validation", "Point-level predictions, calibration, label confusion, anti-circularity feature sets, and bootstrap importance intervals.")
    return _save_figure(fig, out_dir, "fig4_dense_model_validation", formats, "Meta-model prediction, calibration, confusion, robustness, and importance diagnostics.")


def _write_fig5_attribution_control(
    attribution: pd.DataFrame,
    paired: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), facecolor=SURFACE)
    ax_int, ax_pair, ax_delta, ax_adv = axes.ravel()

    data = attribution.sort_values("mean_delta_J0_minus_J1", ascending=True)
    y = np.arange(len(data))
    mean = data["mean_delta_J0_minus_J1"].to_numpy(dtype=float)
    low = data["delta_ci_low"].to_numpy(dtype=float)
    high = data["delta_ci_high"].to_numpy(dtype=float)
    ax_int.errorbar(mean, y, xerr=_xerr(mean, low, high), fmt="none", ecolor=MUTED, linewidth=0.9, capsize=2)
    ax_int.scatter(mean, y, color=[BLUE if lo > 0 else ORANGE if hi < 0 else MUTED for lo, hi in zip(low, high)], s=30, edgecolors="white", linewidths=0.4)
    ax_int.axvline(0, color=INK, linewidth=0.9)
    ax_int.set_yticks(y)
    ax_int.set_yticklabels(data["family"].str.replace("_", " "), fontsize=7)
    ax_int.set_xlabel("NRMSE(J=0) - NRMSE(coupled)")
    ax_int.set_title("a  Paired effect intervals", loc="left")
    _clean_axes(ax_int)

    for fam, sub in paired.groupby("family", sort=True):
        ax_pair.scatter(sub["nrmse_qrc_J0"], sub["nrmse_qrc_J1"], s=18, color=FAMILY_COLORS.get(str(fam), MUTED), alpha=0.70, edgecolors="white", linewidths=0.2, rasterized=True, label=str(fam).replace("_", " "))
    lo = float(np.nanmin([paired["nrmse_qrc_J0"].min(), paired["nrmse_qrc_J1"].min()]))
    hi = float(np.nanmax([paired["nrmse_qrc_J0"].max(), paired["nrmse_qrc_J1"].max()]))
    ax_pair.plot([lo, hi], [lo, hi], color=INK, linewidth=0.9)
    ax_pair.set_xlabel("J=0 QRC NRMSE")
    ax_pair.set_ylabel("Coupled QRC NRMSE")
    ax_pair.set_title("b  Matched-dimensional paired scatter", loc="left")
    ax_pair.legend(frameon=False, fontsize=7, loc="best")
    _clean_axes(ax_pair)

    if "paired_delta_J0_minus_J1" in paired.columns:
        delta = paired["paired_delta_J0_minus_J1"]
    else:
        delta = paired["nrmse_qrc_J0"] - paired["nrmse_qrc_J1"]
    bins = np.linspace(float(delta.min()), float(delta.max()), 34)
    for fam, sub in paired.assign(delta=delta).groupby("family", sort=True):
        ax_delta.hist(sub["delta"], bins=bins, alpha=0.55, color=FAMILY_COLORS.get(str(fam), MUTED), label=str(fam).replace("_", " "))
    ax_delta.axvline(0, color=INK, linewidth=0.9)
    ax_delta.set_xlabel("Paired J0-J1 delta")
    ax_delta.set_ylabel("Rows")
    ax_delta.set_title("c  Delta distribution", loc="left")
    _clean_axes(ax_delta)

    if {"advantage_J1_vs_esn", "advantage_J0_vs_esn"}.issubset(paired.columns):
        ax_adv.scatter(paired["advantage_J1_vs_esn"], paired["advantage_J0_vs_esn"], c=[FAMILY_COLORS.get(str(f), MUTED) for f in paired["family"]], s=18, alpha=0.70, edgecolors="white", linewidths=0.2, rasterized=True)
        lo = float(np.nanmin([paired["advantage_J1_vs_esn"].min(), paired["advantage_J0_vs_esn"].min()]))
        hi = float(np.nanmax([paired["advantage_J1_vs_esn"].max(), paired["advantage_J0_vs_esn"].max()]))
        ax_adv.plot([lo, hi], [lo, hi], color=INK, linewidth=0.9)
        ax_adv.axhline(0, color=MUTED, linewidth=0.7, linestyle=":")
        ax_adv.axvline(0, color=MUTED, linewidth=0.7, linestyle=":")
        ax_adv.set_xlabel("Coupled-QRC advantage vs ESN")
        ax_adv.set_ylabel("J=0 advantage vs ESN")
    else:
        ax_adv.text(0.02, 0.92, "Advantage columns unavailable", transform=ax_adv.transAxes, fontsize=9, color=MUTED)
    ax_adv.set_title("d  Attribution does not establish mechanism", loc="left")
    _clean_axes(ax_adv)

    _figure_header(fig, "Figure 5. Quantum-attribution guardrail", "Matched coupled-vs-J=0 controls are shown explicitly; mechanism claims remain bounded.")
    return _save_figure(fig, out_dir, "fig5_dense_attribution_guardrail", formats, "Dense paired coupled-vs-J=0 attribution-control diagnostics.")


def _regression_panel(ax, data: pd.DataFrame, feature: str, rng: np.random.Generator) -> None:
    raw_x = pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(data["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(raw_x) & np.isfinite(y)
    plot = data.loc[mask, ["family"]].copy()
    x_rank = pd.Series(raw_x[mask]).rank(method="average", pct=True).to_numpy(dtype=float)
    if np.unique(x_rank).size < 50:
        x_rank = np.clip(x_rank + rng.normal(0.0, 0.003, size=x_rank.size), 0.0, 1.0)
    plot["rank"] = x_rank
    plot["qrc_advantage"] = y[mask]
    for fam, sub in plot.groupby("family", sort=True):
        ax.scatter(sub["rank"], sub["qrc_advantage"], s=8, color=FAMILY_COLORS.get(str(fam), MUTED), alpha=0.34, edgecolors="none", rasterized=True)
    fit = _linear_fit_with_ci(plot["rank"].to_numpy(dtype=float), plot["qrc_advantage"].to_numpy(dtype=float))
    if fit is not None:
        x_line, y_hat, low, high, r2 = fit
        ax.fill_between(x_line, low, high, color="#EAF1FE", linewidth=0, alpha=0.75)
        ax.plot(x_line, y_hat, color=INK, linewidth=1.0)
    else:
        r2 = float("nan")
    rho = float(pd.Series(raw_x[mask]).corr(pd.Series(y[mask]), method="spearman")) if int(mask.sum()) > 2 else float("nan")
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.axhline(0.05, color=BLUE, linewidth=0.7, linestyle="--")
    ax.text(0.03, 0.95, f"n={int(mask.sum())}\nrho={rho:.2f}\nR2={r2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, bbox=_textbox())
    ax.set_title(_feature_labels([feature])[0], loc="left", fontsize=9)
    ax.set_xlabel("Percentile rank")
    ax.set_ylabel("QRC advantage")
    _clean_axes(ax)


def _top_extended_features(ext_join: pd.DataFrame, *, limit: int) -> list[str]:
    cols = [c for c in ext_join.columns if c.startswith("ext_")]
    rows = []
    y = pd.to_numeric(ext_join["qrc_advantage"], errors="coerce")
    for col in cols:
        x = pd.to_numeric(ext_join[col], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) > 5:
            rows.append((col, abs(float(x[mask].corr(y[mask], method="spearman")))))
    return [name for name, _ in sorted(rows, key=lambda item: item[1], reverse=True)[:limit]]


def _calibration_table(atlas: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    prob = pd.to_numeric(atlas["predicted_prob_qrc_useful"], errors="coerce").clip(0, 1)
    useful = (atlas["actual_usefulness_label"] == "qrc_useful").astype(float)
    bins = pd.qcut(prob.rank(method="first"), q=min(n_bins, len(atlas)), labels=False, duplicates="drop")
    out = pd.DataFrame({"prob": prob, "useful": useful, "bin": bins}).groupby("bin").agg(mean_prob=("prob", "mean"), observed_rate=("useful", "mean"), n=("useful", "size")).reset_index(drop=True)
    out["se"] = np.sqrt(np.maximum(out["observed_rate"] * (1.0 - out["observed_rate"]) / out["n"], 0.0))
    return out


def _confusion_matrix(atlas: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    actual_labels = [label for label in CATEGORY_ORDER if label in set(atlas["actual_usefulness_label"])]
    pred_labels = [label for label in CATEGORY_ORDER if label in set(atlas["predicted_usefulness_label"])]
    matrix = np.zeros((len(actual_labels), len(pred_labels)), dtype=int)
    for i, actual in enumerate(actual_labels):
        for j, pred in enumerate(pred_labels):
            matrix[i, j] = int(((atlas["actual_usefulness_label"] == actual) & (atlas["predicted_usefulness_label"] == pred)).sum())
    return matrix, actual_labels, pred_labels


def _feature_set_label(value: object) -> str:
    mapping = {
        "all": "all",
        "without_r2_linear": "no r2",
        "without_predictability_proxies": "no proxies",
        "chaos_nonlinearity_complexity_only": "complexity",
    }
    return mapping.get(str(value), str(value).replace("_", " "))


def _robust_scale(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return 1.0
    q75, q25 = np.nanpercentile(vals, [75, 25])
    iqr = float(q75 - q25)
    if math.isfinite(iqr) and iqr > 1e-12:
        return iqr
    std = float(np.nanstd(vals))
    return std if math.isfinite(std) and std > 1e-12 else 1.0


def _textbox() -> dict[str, Any]:
    return {"facecolor": PANEL, "edgecolor": AXIS, "boxstyle": "round,pad=0.22", "linewidth": 0.55, "alpha": 0.92}


def _write_report(
    atlas: pd.DataFrame,
    family: pd.DataFrame,
    meta: pd.DataFrame,
    family_ci: pd.DataFrame,
    robustness: pd.DataFrame,
    attribution: pd.DataFrame,
    path: Path,
) -> dict[str, str]:
    counts = atlas["actual_usefulness_label"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    top_family = family.sort_values(["qrc_useful_rate", "mean_qrc_advantage"], ascending=[False, False]).iloc[0]
    meta_row = meta.iloc[0]
    no_proxy = robustness[robustness["feature_set"] == "without_predictability_proxies"].iloc[0]
    overall = family_ci[family_ci["family"] == "overall"].iloc[0]
    attr = attribution[attribution["family"] == "overall"].iloc[0]
    body = f"""# Scientific Plot Package

- Rows: `{len(atlas)}`.
- QRC-useful / near-tie / baseline-preferred: `{int(counts['qrc_useful'])}` / `{int(counts['near_tie'])}` / `{int(counts['baseline_preferred'])}`.
- Strongest useful family: `{top_family['family']}` with useful rate `{float(top_family['qrc_useful_rate']):.3f}`.
- Overall mean advantage CI: `[{float(overall['mean_ci_low']):.4f}, {float(overall['mean_ci_high']):.4f}]`.
- Meta-model CV R2 / AUC: `{float(meta_row['regression_r2_mean']):.4f}` / `{float(meta_row['classification_roc_auc_mean']):.4f}`.
- No-proxy CV R2 / AUC: `{float(no_proxy['regression_r2_mean']):.4f}` / `{float(no_proxy['classification_roc_auc_mean']):.4f}`.
- Attribution control overall J0-J1 CI: `[{float(attr['delta_ci_low']):.4f}, {float(attr['delta_ci_high']):.4f}]`, `{attr['mechanism_signal']}`.

Claim boundary: these figures support dataset categorization and meta-model interpretation. They do not establish broad average QRC superiority or a quantum coupling mechanism.
"""
    path.write_text(body, encoding="utf-8")
    return {"headline": f"{len(atlas)} rows; strongest useful family {top_family['family']}; attribution {attr['mechanism_signal']}"}


def _write_html_index(out_dir: Path, figures: list[dict[str, str]], report: dict[str, str]) -> None:
    sections = "\n".join(
        f"<section><h2>{item['title']}</h2><p>{item['caption']}</p><img src='{item['png']}' alt='{item['title']}'></section>"
        for item in figures
        if item.get("png")
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scientific QRC Atlas Figures</title>
<style>
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#FCFCFD; color:#1F2430; }}
header {{ padding:32px 56px 20px; background:#FFFFFF; border-bottom:1px solid #D7DBE7; }}
h1 {{ margin:0 0 8px; font-size:34px; letter-spacing:0; }}
.sub {{ color:#6F768A; max-width:960px; }}
main {{ padding:28px 56px 56px; }}
section {{ margin:0 0 28px; padding:18px; background:white; border:1px solid #D7DBE7; border-radius:8px; }}
h2 {{ margin:0 0 4px; font-size:19px; }}
p {{ margin:0 0 14px; color:#6F768A; }}
img {{ width:100%; height:auto; display:block; border:1px solid #E6E8F0; border-radius:4px; }}
</style>
</head>
<body>
<header><h1>Scientific QRC Atlas Figures</h1><p class="sub">{report['headline']}. Claim boundary: dataset categorization only; no broad quantum-advantage or mechanism claim.</p></header>
<main>{sections}</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
