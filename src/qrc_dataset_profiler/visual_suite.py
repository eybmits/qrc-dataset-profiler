"""State-of-the-art static visual suite for the QRC usefulness atlas."""

from __future__ import annotations

import ast
import html
import json
import math
import textwrap
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CATEGORY_ORDER = ("baseline_preferred", "near_tie", "qrc_useful")
CATEGORY_COLORS = {
    "baseline_preferred": "#CC6F47",
    "near_tie": "#A7ADBD",
    "qrc_useful": "#5477C4",
}
CATEGORY_DARK = {
    "baseline_preferred": "#804126",
    "near_tie": "#6F768A",
    "qrc_useful": "#2E4780",
}
FAMILY_COLORS = {
    "chaotic_flow": "#5477C4",
    "chaotic_map": "#2E4780",
    "colored_noise": "#B8A037",
    "input_driven": "#BD569B",
    "linear_stochastic": "#71B436",
    "long_range": "#736422",
    "nonlinear_stochastic": "#CC6F47",
    "nonstationary": "#8A3A6F",
    "oscillatory": "#386411",
    "real_bridge": "#6F768A",
}
INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E6E8F0"
AXIS = "#D7DBE7"
PANEL = "#FFFFFF"
SURFACE = "#FCFCFD"
ORANGE = "#CC6F47"
BLUE = "#5477C4"
GOLD = "#B8A037"
OLIVE = "#71B436"
PINK = "#BD569B"


def run_visual_suite(
    *,
    atlas_dir: Path,
    analysis_dir: Path,
    attribution_dir: Path,
    features_dir: Path,
    out_dir: Path,
    sweep_catalog: Path | None = None,
    full_catalog: Path | None = None,
    formats: Iterable[str] = ("png", "pdf"),
) -> dict[str, Any]:
    """Create a deterministic visual package from atlas and analysis artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = _normalize_formats(formats)

    atlas = _load_required(atlas_dir / "qrc_usefulness_map.csv")
    family = _load_required(atlas_dir / "family_usefulness_summary.csv")
    meta = _load_required(atlas_dir / "meta_model_summary.csv")
    atlas_importances = _load_required(atlas_dir / "atlas_importances.csv")
    family_ci = _load_required(analysis_dir / "family_advantage_bootstrap.csv")
    robustness = _load_required(analysis_dir / "robustness_summary.csv")
    importance_ci = _load_required(analysis_dir / "importance_bootstrap.csv")
    attribution = _load_required(attribution_dir / "family_attribution_bootstrap.csv")
    paired_attribution = _load_required(attribution_dir / "paired_attribution.csv")
    extended_features = _load_required(features_dir / "extended_features_sweep.csv")
    sweep = _load_optional(sweep_catalog)
    full = _load_optional(full_catalog)

    _require_columns(
        atlas,
        {
            "dataset_id",
            "name",
            "family",
            "qrc_advantage",
            "actual_usefulness_label",
            "predicted_qrc_advantage",
            "predicted_usefulness_label",
            "predicted_prob_qrc_useful",
            "property_pc1",
            "property_pc2",
        },
        "qrc_usefulness_map.csv",
    )

    figures: list[dict[str, str]] = []
    with _plot_context():
        figures.append(_write_visual_abstract(atlas, family, meta, robustness, attribution, out_dir, fmt))
        figures.append(_write_property_landscape(atlas, family, out_dir, fmt))
        if sweep is not None:
            figures.append(_write_sweep_barcode(atlas, sweep, out_dir, fmt))
        figures.append(_write_family_outcomes(family, family_ci, out_dir, fmt))
        figures.append(_write_advantage_distributions(atlas, family, family_ci, out_dir, fmt))
        figures.append(_write_meta_model_evidence(atlas, meta, atlas_importances, importance_ci, robustness, out_dir, fmt))
        figures.append(_write_extended_feature_map(atlas, extended_features, family, out_dir, fmt))
        figures.append(_write_attribution_guardrail(attribution, paired_attribution, out_dir, fmt))
        if full is not None:
            figures.append(_write_full_catalog_inventory(full, out_dir, fmt))
        if sweep is not None:
            figures.append(_write_all_points_feature_regressions(atlas, sweep, importance_ci, out_dir, fmt))

    report = _visual_report(atlas, family, meta, family_ci, robustness, attribution)
    (out_dir / "VISUAL_SUITE_REPORT.md").write_text(report, encoding="utf-8")
    index_path = _write_html_index(out_dir, figures, atlas, family, meta, attribution)

    outputs = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    manifest = {
        "analysis_version": "visual-suite-v1",
        "inputs": {
            "atlas_dir": str(atlas_dir),
            "analysis_dir": str(analysis_dir),
            "attribution_dir": str(attribution_dir),
            "features_dir": str(features_dir),
            "sweep_catalog": str(sweep_catalog) if sweep_catalog else None,
            "full_catalog": str(full_catalog) if full_catalog else None,
        },
        "n_rows": int(len(atlas)),
        "n_families": int(family["family"].nunique()),
        "n_extended_features": int(len([c for c in extended_features.columns if c.startswith("ext_")])),
        "figures": figures,
        "outputs": outputs,
        "html_index": index_path.name,
        "claim_boundary": (
            "The visual suite supports a dataset-categorization and meta-model claim. "
            "It does not claim broad average QRC superiority or a quantum coupling mechanism."
        ),
    }
    (out_dir / "visual_suite_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_visual_abstract(
    atlas: pd.DataFrame,
    family: pd.DataFrame,
    meta: pd.DataFrame,
    robustness: pd.DataFrame,
    attribution: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyBboxPatch

    stem = "00_visual_abstract"
    counts = atlas["actual_usefulness_label"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    top_family = family.sort_values(["qrc_useful_rate", "mean_qrc_advantage"], ascending=[False, False]).iloc[0]
    meta_row = meta.iloc[0]
    no_proxy = _row_or_first(robustness, "feature_set", "without_predictability_proxies")
    attr_overall = _row_or_first(attribution, "family", "overall")
    overall_rate = float(counts.get("qrc_useful", 0)) / max(len(atlas), 1)

    fig = plt.figure(figsize=(13.5, 7.8), facecolor=SURFACE)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.05], width_ratios=[1.25, 1.05, 1.15], hspace=0.58, wspace=0.36)
    ax_cards = fig.add_subplot(gs[0, 0])
    ax_comp = fig.add_subplot(gs[0, 1])
    ax_family = fig.add_subplot(gs[0, 2])
    ax_robust = fig.add_subplot(gs[1, 0])
    ax_attr = fig.add_subplot(gs[1, 1])
    ax_boundary = fig.add_subplot(gs[1, 2])

    ax_cards.set_axis_off()
    card_data = [
        ("Atlas rows", f"{len(atlas):,}", "50 generators"),
        ("QRC-useful", f"{int(counts.get('qrc_useful', 0)):,}", f"{overall_rate:.1%} of sweep"),
        ("Meta-model", f"R2 {float(meta_row['regression_r2_mean']):.2f}", f"AUC {float(meta_row['classification_roc_auc_mean']):.2f}"),
        ("No-proxy check", f"R2 {float(no_proxy['regression_r2_mean']):.2f}", f"AUC {float(no_proxy['classification_roc_auc_mean']):.2f}"),
    ]
    for i, (label, value, subvalue) in enumerate(card_data):
        y = 0.77 - i * 0.245
        patch = FancyBboxPatch(
            (0.02, y),
            0.95,
            0.19,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax_cards.transAxes,
            linewidth=0.8,
            edgecolor=AXIS,
            facecolor=PANEL,
        )
        ax_cards.add_patch(patch)
        ax_cards.text(0.07, y + 0.145, label.upper(), transform=ax_cards.transAxes, fontsize=8, color=MUTED, weight="bold")
        ax_cards.text(0.07, y + 0.066, value, transform=ax_cards.transAxes, fontsize=17, color=INK, weight="bold")
        ax_cards.text(0.07, y + 0.020, subvalue, transform=ax_cards.transAxes, fontsize=8.2, color=MUTED, va="bottom")

    _plot_outcome_composition(ax_comp, counts)
    _plot_family_useful_rates(ax_family, family)
    _plot_robustness_lines(ax_robust, robustness)
    _plot_attribution_intervals(ax_attr, attribution)

    ax_boundary.set_axis_off()
    boundary_lines = [
        ("Main claim", "QRC usefulness is dataset-conditional and learnable from measured properties."),
        ("Strongest family", f"{top_family['family'].replace('_', ' ')}: useful rate {float(top_family['qrc_useful_rate']):.1%}."),
        (
            "Guardrail",
            f"Coupled vs J=0 CI [{float(attr_overall['delta_ci_low']):.3f}, {float(attr_overall['delta_ci_high']):.3f}], signal={attr_overall['mechanism_signal']}.",
        ),
        ("Not claimed", "No broad average QRC superiority and no coupling/entanglement mechanism claim."),
    ]
    y = 0.86
    for label, text in boundary_lines:
        wrapped = textwrap.fill(f"{label}: {text}", width=50)
        ax_boundary.text(0.02, y, wrapped, transform=ax_boundary.transAxes, fontsize=9.8, color=INK, va="top")
        y -= 0.20 + 0.045 * wrapped.count("\n")

    _figure_header(
        fig,
        "QRC Usefulness Atlas: Visual Abstract",
        "Outcome composition, family structure, anti-circularity robustness, and quantum-attribution guardrails in one reproducible view.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Executive visual summary of the full QRC usefulness atlas.")


def _write_property_landscape(atlas: pd.DataFrame, family: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    stem = "01_property_landscape"
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.4), facecolor=SURFACE)
    ax_actual, ax_prob = axes
    for label in CATEGORY_ORDER:
        sub = atlas[atlas["actual_usefulness_label"] == label]
        if sub.empty:
            continue
        ax_actual.scatter(
            sub["property_pc1"],
            sub["property_pc2"],
            s=26,
            color=CATEGORY_COLORS[label],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
            label=_clean_label(label),
        )
    _annotate_centroids(ax_actual, atlas)
    ax_actual.set_title("Observed usefulness regions", loc="left", pad=10)
    ax_actual.set_xlabel("Property PC1")
    ax_actual.set_ylabel("Property PC2")
    ax_actual.legend(frameon=False, loc="best")
    _clean_axes(ax_actual)

    cmap = _blue_gold_cmap()
    sc = ax_prob.scatter(
        atlas["property_pc1"],
        atlas["property_pc2"],
        c=atlas["predicted_prob_qrc_useful"],
        cmap=cmap,
        vmin=0,
        vmax=1,
        s=28,
        alpha=0.86,
        edgecolors="white",
        linewidths=0.25,
    )
    _annotate_centroids(ax_prob, atlas)
    cb = fig.colorbar(sc, ax=ax_prob, fraction=0.046, pad=0.035)
    cb.set_label("Predicted probability of qrc_useful")
    cb.outline.set_edgecolor(AXIS)
    ax_prob.set_title("Meta-model useful-probability surface", loc="left", pad=10)
    ax_prob.set_xlabel("Property PC1")
    ax_prob.set_ylabel("Property PC2")
    _clean_axes(ax_prob)

    _figure_header(
        fig,
        "Dataset-Property Landscape",
        "Each point is one sweep dataset; colors show observed labels on the left and learned qrc_useful probability on the right.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Two-panel property map of observed and predicted QRC usefulness.")


def _write_sweep_barcode(atlas: pd.DataFrame, sweep: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    stem = "02_sweep_barcode"
    merged = atlas.merge(sweep[["dataset_id", "params"]], on="dataset_id", how="left") if "params" in sweep.columns else atlas.copy()
    merged["generator_label"] = [_generator_label(row.get("params"), row.get("name", "")) for _, row in merged.iterrows()]
    grouped = (
        merged.groupby(["family", "generator_label"], sort=True)
        .agg(n=("qrc_advantage", "size"), mean_advantage=("qrc_advantage", "mean"), useful_rate=("actual_usefulness_label", lambda x: float((x == "qrc_useful").mean())))
        .reset_index()
    )
    family_rank = family_order_from_rows(merged)
    grouped["family_rank"] = grouped["family"].map(family_rank).fillna(999)
    grouped = grouped.sort_values(["family_rank", "useful_rate", "mean_advantage", "generator_label"], ascending=[True, False, False, True])
    row_keys = list(grouped[["family", "generator_label"]].itertuples(index=False, name=None))
    max_n = int(grouped["n"].max())
    matrix = np.full((len(row_keys), max_n), np.nan, dtype=float)
    for r, (family_name, generator) in enumerate(row_keys):
        vals = (
            merged[(merged["family"] == family_name) & (merged["generator_label"] == generator)]
            .sort_values(["seed", "name"], kind="mergesort")["qrc_advantage"]
            .to_numpy(dtype=float)
        )
        matrix[r, : len(vals)] = vals

    fig_height = max(8.5, 0.21 * len(row_keys) + 2.2)
    fig, ax = plt.subplots(figsize=(12.8, fig_height), facecolor=SURFACE)
    finite = np.abs(matrix[np.isfinite(matrix)])
    lim = float(np.nanpercentile(finite, 97.5)) if finite.size else 1.0
    lim = max(lim, 0.15)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    image = ax.imshow(matrix, aspect="auto", cmap=_advantage_cmap(), norm=norm, interpolation="nearest")
    cb = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.012)
    cb.set_label("QRC advantage: ESN NRMSE - QRC NRMSE")
    cb.outline.set_edgecolor(AXIS)
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_yticklabels([_short_label(gen) for _, gen in row_keys], fontsize=6.5)
    ax.set_xlabel("Variant index within generator/control group")
    ax.set_title("All sweep rows as a generator-by-variant barcode", loc="left", pad=10)
    ax.set_facecolor("#F4F5F9")
    ax.tick_params(axis="both", length=0)

    y0 = 0
    for family_name, sub in grouped.groupby("family", sort=False):
        y1 = y0 + len(sub)
        ax.axhline(y0 - 0.5, color=PANEL, linewidth=1.5)
        ax.text(
            max_n - 0.75,
            (y0 + y1 - 1) / 2,
            family_name.replace("_", " "),
            ha="right",
            va="center",
            fontsize=7.5,
            color=MUTED,
            bbox={"facecolor": "#F4F5F9", "edgecolor": "none", "alpha": 0.82, "pad": 1.4},
        )
        y0 = y1
    ax.axhline(len(row_keys) - 0.5, color=PANEL, linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)

    _figure_header(
        fig,
        "1000-Row Sweep Barcode",
        "Every colored cell is one profiled dataset; blue favors QRC over the matched ESN baseline, orange favors the baseline.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Full sweep barcode by generator/control group and parameter variant.")


def _write_family_outcomes(family: pd.DataFrame, family_ci: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    stem = "03_family_outcomes"
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax_stack, ax_ci = axes
    _plot_family_category_stack(ax_stack, family)
    _plot_family_advantage_ci(ax_ci, family_ci)
    _figure_header(
        fig,
        "Family-Level QRC Usefulness",
        "Stacked label shares and bootstrap confidence intervals show where QRC helps selectively rather than on average.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Family outcome shares and advantage confidence intervals.")


def _write_advantage_distributions(
    atlas: pd.DataFrame,
    family: pd.DataFrame,
    family_ci: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    stem = "04_advantage_distributions"
    order = list(family.sort_values("mean_qrc_advantage", ascending=True)["family"])
    data = [atlas.loc[atlas["family"] == fam, "qrc_advantage"].dropna().to_numpy(dtype=float) for fam in order]
    positions = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(12.6, 7.2), facecolor=SURFACE)
    parts = ax.violinplot(data, positions=positions, vert=False, widths=0.82, showextrema=False, showmeans=False, showmedians=False)
    means = family.set_index("family")["mean_qrc_advantage"].to_dict()
    for body, fam in zip(parts["bodies"], order):
        color = BLUE if means.get(fam, 0.0) > 0 else ORANGE
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.30)

    rng = np.random.default_rng(9)
    for y, fam in zip(positions, order):
        vals = atlas.loc[atlas["family"] == fam, "qrc_advantage"].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        sample = vals if vals.size <= 120 else rng.choice(vals, size=120, replace=False)
        jitter = rng.uniform(-0.19, 0.19, size=sample.size)
        colors = [CATEGORY_COLORS[_usefulness_label(v)] for v in sample]
        ax.scatter(sample, y + jitter, s=14, c=colors, alpha=0.68, edgecolors="white", linewidths=0.2, zorder=3)

    ci = family_ci[family_ci["family"].isin(order)].set_index("family")
    for y, fam in zip(positions, order):
        if fam not in ci.index:
            continue
        row = ci.loc[fam]
        mean = float(row["mean_advantage"])
        low = float(row["mean_ci_low"])
        high = float(row["mean_ci_high"])
        lo, hi = sorted((low, high))
        ax.plot([lo, hi], [y, y], color=INK, linewidth=1.2, zorder=4)
        ax.scatter([mean], [y], marker="D", s=28, color=INK, zorder=5)

    ax.axvline(0.0, color=INK, linewidth=0.9)
    ax.axvline(0.05, color=BLUE, linewidth=0.9, linestyle="--")
    ax.axvline(-0.05, color=ORANGE, linewidth=0.9, linestyle=":")
    ax.set_yticks(positions)
    ax.set_yticklabels([fam.replace("_", " ") for fam in order])
    ax.set_xlabel("QRC advantage: ESN NRMSE - QRC NRMSE")
    ax.set_title("Within-family advantage distributions", loc="left", pad=10)
    _clean_axes(ax)
    _figure_header(
        fig,
        "Advantage Distributions",
        "Violin density, sampled rows, and bootstrap mean intervals reveal heterogeneity hidden by family averages.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Within-family QRC advantage distributions with bootstrap mean intervals.")


def _write_meta_model_evidence(
    atlas: pd.DataFrame,
    meta: pd.DataFrame,
    atlas_importances: pd.DataFrame,
    importance_ci: pd.DataFrame,
    robustness: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    stem = "05_meta_model_evidence"
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0), facecolor=SURFACE)
    ax_pred, ax_imp, ax_robust, ax_error = axes.ravel()
    _plot_prediction_scatter(ax_pred, atlas, meta)
    _plot_importance_ci(ax_imp, importance_ci if not importance_ci.empty else atlas_importances)
    _plot_robustness_lines(ax_robust, robustness)
    _plot_prediction_error_by_family(ax_error, atlas)
    _figure_header(
        fig,
        "Meta-Model Evidence And Anti-Circularity",
        "Predictive accuracy, feature importance intervals, robustness ablations, and residual checks for the usefulness classifier/regressor.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Meta-model prediction, importance, robustness, and residual diagnostics.")


def _write_extended_feature_map(
    atlas: pd.DataFrame,
    extended: pd.DataFrame,
    family: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    stem = "06_extended_feature_map"
    merged = atlas[["dataset_id", "family", "qrc_advantage", "actual_usefulness_label"]].merge(extended, on="dataset_id", how="inner", suffixes=("", "_ext"))
    ext_cols = [c for c in merged.columns if c.startswith("ext_")]
    if not ext_cols:
        raise ValueError("extended feature table has no ext_* columns")
    corr_rows = []
    for col in ext_cols:
        x = pd.to_numeric(merged[col], errors="coerce")
        y = pd.to_numeric(merged["qrc_advantage"], errors="coerce")
        corr = float(x.corr(y, method="spearman")) if x.notna().sum() > 2 and y.notna().sum() > 2 else float("nan")
        if math.isfinite(corr):
            corr_rows.append((col, corr))
    corr_df = pd.DataFrame(corr_rows, columns=["feature", "spearman"]).sort_values("spearman", key=lambda s: s.abs(), ascending=False).head(14)
    selected = list(corr_df["feature"])
    family_order = list(family.sort_values("qrc_useful_rate", ascending=False)["family"])
    z = merged[selected].apply(pd.to_numeric, errors="coerce")
    z = (z - z.median()) / z.apply(lambda c: _robust_scale(c), axis=0)
    z = z.clip(-2.5, 2.5)
    heat = pd.concat([merged[["family"]], z], axis=1).groupby("family").mean(numeric_only=True).reindex(family_order)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.0), facecolor=SURFACE, gridspec_kw={"width_ratios": [0.95, 1.35]})
    ax_corr, ax_heat = axes
    corr_plot = corr_df.iloc[::-1]
    colors = [BLUE if v >= 0 else ORANGE for v in corr_plot["spearman"]]
    ax_corr.barh(_feature_labels(corr_plot["feature"]), corr_plot["spearman"], color=colors, height=0.72)
    ax_corr.axvline(0, color=INK, linewidth=0.8)
    ax_corr.set_xlabel("Spearman correlation with QRC advantage")
    ax_corr.set_title("Extended descriptors most aligned with usefulness", loc="left", pad=10)
    _clean_axes(ax_corr)

    image = ax_heat.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap=_feature_cmap(), norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5), interpolation="nearest")
    ax_heat.set_yticks(np.arange(len(heat.index)))
    ax_heat.set_yticklabels([str(v).replace("_", " ") for v in heat.index])
    ax_heat.set_xticks(np.arange(len(heat.columns)))
    ax_heat.set_xticklabels(_feature_labels(heat.columns), rotation=42, ha="right", fontsize=7)
    ax_heat.set_title("Family-average standardized feature structure", loc="left", pad=10)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_color(AXIS)
    cb = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.018)
    cb.set_label("Robust z-score")
    cb.outline.set_edgecolor(AXIS)

    _figure_header(
        fig,
        "Extended Feature Map",
        "Tier-B descriptors add complexity, recurrence, volatility, trend, spectral, and changepoint views on the same usefulness atlas.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Extended feature correlations and family-level feature heatmap.")


def _write_attribution_guardrail(
    attribution: pd.DataFrame,
    paired: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    stem = "07_quantum_attribution_guardrail"
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax_interval, ax_pair = axes
    _plot_attribution_intervals(ax_interval, attribution)
    if {"nrmse_qrc_J0", "nrmse_qrc_J1", "family"}.issubset(paired.columns):
        for fam, sub in paired.groupby("family", sort=True):
            ax_pair.scatter(
                sub["nrmse_qrc_J0"],
                sub["nrmse_qrc_J1"],
                s=22,
                alpha=0.72,
                color=FAMILY_COLORS.get(str(fam), MUTED),
                edgecolors="white",
                linewidths=0.25,
                label=str(fam).replace("_", " "),
            )
        lo = float(np.nanmin([paired["nrmse_qrc_J0"].min(), paired["nrmse_qrc_J1"].min()]))
        hi = float(np.nanmax([paired["nrmse_qrc_J0"].max(), paired["nrmse_qrc_J1"].max()]))
        ax_pair.plot([lo, hi], [lo, hi], color=INK, linewidth=0.9)
        ax_pair.set_xlim(lo, hi)
        ax_pair.set_ylim(lo, hi)
        ax_pair.set_xlabel("J=0 QRC NRMSE")
        ax_pair.set_ylabel("Coupled QRC NRMSE")
        ax_pair.set_title("Paired matched-dimension control", loc="left", pad=10)
        ax_pair.legend(frameon=False, loc="best")
        _clean_axes(ax_pair)
    _figure_header(
        fig,
        "Quantum Attribution Guardrail",
        "The corrected paired coupled-vs-J=0 control is shown explicitly; mechanism claims remain bounded.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Paired coupled-vs-J=0 control intervals and scatter.")


def _write_full_catalog_inventory(full: pd.DataFrame, out_dir: Path, formats: tuple[str, ...]) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import TwoSlopeNorm

    stem = "08_full_catalog_inventory"
    core_cols = [
        "ac_timescale",
        "ami_first_min",
        "r2_linear",
        "nl_gain",
        "snr_db",
        "lyapunov",
        "zero_one_K",
        "spectral_entropy",
        "dom_freq",
        "spectral_flatness",
        "adf_p",
        "kpss_p",
        "n_diffs",
        "dfa_alpha",
        "perm_entropy",
        "forecastability",
        "pred_nrmse_gbm",
    ]
    cols = [c for c in core_cols if c in full.columns]
    if not {"name", "family", "qrc_advantage"}.issubset(full.columns) or not cols:
        raise ValueError("full catalog lacks required inventory columns")
    data = full.sort_values(["family", "name"], kind="mergesort").reset_index(drop=True)
    z = data[cols].apply(pd.to_numeric, errors="coerce")
    z = (z - z.median()) / z.apply(lambda c: _robust_scale(c), axis=0)
    z = z.clip(-2.5, 2.5)

    fig = plt.figure(figsize=(13.2, max(9.0, 0.18 * len(data) + 2.5)), facecolor=SURFACE)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.55, 0.75], wspace=0.16)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    image = ax_heat.imshow(z.to_numpy(dtype=float), aspect="auto", cmap=_feature_cmap(), norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5), interpolation="nearest")
    ax_heat.set_yticks(np.arange(len(data)))
    ax_heat.set_yticklabels([_short_label(v) for v in data["name"]], fontsize=6.6)
    ax_heat.set_xticks(np.arange(len(cols)))
    ax_heat.set_xticklabels(_feature_labels(cols), rotation=45, ha="right", fontsize=7)
    ax_heat.set_title("Core schema-v1 property inventory", loc="left", pad=10)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_color(AXIS)
    cb = fig.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.01)
    cb.set_label("Robust z-score")
    cb.outline.set_edgecolor(AXIS)

    y = np.arange(len(data))
    adv = pd.to_numeric(data["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    colors = [CATEGORY_COLORS[_usefulness_label(v)] if math.isfinite(float(v)) else MUTED for v in adv]
    ax_bar.barh(y, adv, color=colors, height=0.72)
    ax_bar.axvline(0, color=INK, linewidth=0.8)
    ax_bar.axvline(0.05, color=BLUE, linewidth=0.8, linestyle="--")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([])
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("QRC advantage")
    ax_bar.set_title("Base-catalog outcome", loc="left", pad=10)
    _clean_axes(ax_bar)

    _figure_header(
        fig,
        "50-Generator Benchmark Inventory",
        "The first full catalog now includes Santa Fe laser; rows show benchmark-level properties and matched ESN-vs-QRC outcomes.",
    )
    return _save_figure(fig, out_dir, stem, formats, "Full 50-row benchmark inventory with core properties and QRC advantage.")


def _write_all_points_feature_regressions(
    atlas: pd.DataFrame,
    sweep: pd.DataFrame,
    importance: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    stem = "09_all_points_feature_regressions"
    selected = _select_regression_features(sweep, importance, limit=8)
    if not selected:
        raise ValueError("no finite numeric sweep features available for all-points regression plot")
    merged = atlas[["dataset_id", "family", "qrc_advantage"]].merge(sweep[["dataset_id", *selected]], on="dataset_id", how="inner")

    n_cols = 4
    n_rows = int(math.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.6, 4.25 * n_rows + 1.0), facecolor=SURFACE, squeeze=False)
    rng = np.random.default_rng(17)
    for ax, feature in zip(axes.ravel(), selected):
        _plot_feature_regression_panel(ax, merged, feature, rng)
    for ax in axes.ravel()[len(selected) :]:
        ax.set_visible(False)

    handles = [
        _legend_patch(FAMILY_COLORS.get(family, MUTED), str(family).replace("_", " "))
        for family in sorted(merged["family"].dropna().unique())
    ]
    fig.legend(handles=handles, loc="center right", ncol=1, frameon=False, bbox_to_anchor=(0.985, 0.48), fontsize=8, title="Family")
    fig._qrc_tight_rect = (0.02, 0.02, 0.86, 0.90)
    _figure_header(
        fig,
        "All-Points Feature Regressions",
        "Each panel shows all 1000 sweep rows against one top meta-model feature; x-axes are feature percentile ranks for comparable scales, with OLS fit and 95% confidence band.",
    )
    return _save_figure(fig, out_dir, stem, formats, "All finite sweep rows plotted against top features with regression lines and confidence bands.")


def _plot_feature_regression_panel(ax, data: pd.DataFrame, feature: str, rng: np.random.Generator) -> None:
    raw_x = pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(data["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(raw_x) & np.isfinite(y)
    plot = data.loc[mask, ["family"]].copy()
    x_rank = pd.Series(raw_x[mask]).rank(method="average", pct=True).to_numpy(dtype=float)
    if np.unique(x_rank).size < 50:
        x_rank = np.clip(x_rank + rng.normal(0.0, 0.0035, size=x_rank.size), 0.0, 1.0)
    plot["feature_rank"] = x_rank
    plot["qrc_advantage"] = y[mask]

    for family, sub in plot.groupby("family", sort=True):
        ax.scatter(
            sub["feature_rank"],
            sub["qrc_advantage"],
            s=13,
            color=FAMILY_COLORS.get(str(family), MUTED),
            alpha=0.42,
            edgecolors="none",
            rasterized=True,
        )

    fit = _linear_fit_with_ci(plot["feature_rank"].to_numpy(dtype=float), plot["qrc_advantage"].to_numpy(dtype=float))
    if fit is not None:
        x_line, y_hat, low, high, r2 = fit
        ax.fill_between(x_line, low, high, color="#EAF1FE", alpha=0.82, linewidth=0)
        ax.plot(x_line, y_hat, color=INK, linewidth=1.25)
    else:
        r2 = float("nan")

    spearman = float(pd.Series(raw_x[mask]).corr(pd.Series(y[mask]), method="spearman")) if int(mask.sum()) > 2 else float("nan")
    ax.axhline(0.0, color=INK, linewidth=0.8)
    ax.axhline(0.05, color=BLUE, linewidth=0.8, linestyle="--")
    ax.text(
        0.03,
        0.96,
        f"n={int(mask.sum())}\nrho={spearman:.2f}\nR2={r2:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=INK,
        bbox={"facecolor": PANEL, "edgecolor": AXIS, "boxstyle": "round,pad=0.22", "linewidth": 0.6, "alpha": 0.90},
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Feature percentile rank")
    ax.set_ylabel("QRC advantage")
    ax.set_title(_feature_labels([feature])[0], loc="left", pad=8)
    _clean_axes(ax)


def _select_regression_features(sweep: pd.DataFrame, importance: pd.DataFrame, *, limit: int) -> list[str]:
    candidates = list(importance.sort_values("importance_mean", ascending=False)["feature"]) if "importance_mean" in importance.columns else []
    candidates.extend([c for c in sweep.columns if c not in candidates])
    selected: list[str] = []
    for require_all_finite in (True, False):
        for feature in candidates:
            if feature in selected or feature not in sweep.columns:
                continue
            values = pd.to_numeric(sweep[feature], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            if finite.sum() < max(8, int(0.95 * len(values))):
                continue
            if require_all_finite and finite.sum() != len(values):
                continue
            if np.unique(values[finite]).size < 6:
                continue
            selected.append(str(feature))
            if len(selected) >= limit:
                return selected
    return selected[:limit]


def _linear_fit_with_ci(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.unique(x).size < 2:
        return None
    x_design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    y_fit = x_design @ beta
    residual = y - y_fit
    dof = max(int(x.size) - 2, 1)
    sigma = float(np.sqrt(np.sum(residual**2) / dof))
    x_line = np.linspace(0.0, 1.0, 160)
    line_design = np.column_stack([np.ones_like(x_line), x_line])
    y_hat = line_design @ beta
    x_bar = float(np.mean(x))
    sxx = float(np.sum((x - x_bar) ** 2))
    if sxx <= 1e-12:
        return None
    se = sigma * np.sqrt((1.0 / x.size) + ((x_line - x_bar) ** 2 / sxx))
    ci = 1.96 * se
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 1e-12 else float("nan")
    return x_line, y_hat, y_hat - ci, y_hat + ci, r2


def _plot_outcome_composition(ax, counts: pd.Series) -> None:
    total = float(max(counts.sum(), 1))
    left = 0.0
    for label in CATEGORY_ORDER:
        value = float(counts.get(label, 0))
        width = value / total
        ax.barh([0], [width], left=left, color=CATEGORY_COLORS[label], height=0.45)
        if width > 0.10:
            ax.text(left + width / 2, 0, f"{int(value)}\n{width:.0%}", ha="center", va="center", fontsize=9, color="white", weight="bold")
        left += width
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("Share of 1000-row sweep")
    ax.set_title("Outcome composition", loc="left", pad=10)
    ax.legend(
        handles=[_legend_patch(CATEGORY_COLORS[label], _clean_label(label)) for label in CATEGORY_ORDER],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.54),
        ncols=1,
    )
    _clean_axes(ax)


def _plot_family_useful_rates(ax, family: pd.DataFrame) -> None:
    data = family.sort_values("qrc_useful_rate", ascending=True)
    y = np.arange(len(data))
    colors = [FAMILY_COLORS.get(str(v), BLUE) for v in data["family"]]
    ax.barh(y, data["qrc_useful_rate"], color=colors, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(data["family"].str.replace("_", " "))
    ax.set_xlim(0, max(0.5, float(data["qrc_useful_rate"].max()) * 1.15))
    ax.set_xlabel("qrc_useful rate")
    ax.set_title("Family useful-rate ranking", loc="left", pad=10)
    _clean_axes(ax)


def _plot_family_category_stack(ax, family: pd.DataFrame) -> None:
    data = family.sort_values("qrc_useful_rate", ascending=True)
    y = np.arange(len(data))
    left = np.zeros(len(data), dtype=float)
    for label, col in (
        ("baseline_preferred", "baseline_preferred_rate"),
        ("near_tie", "near_tie_rate"),
        ("qrc_useful", "qrc_useful_rate"),
    ):
        vals = pd.to_numeric(data[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.72, color=CATEGORY_COLORS[label], label=_clean_label(label))
        left += vals
    for i, n in enumerate(data["n"] if "n" in data.columns else [np.nan] * len(data)):
        if math.isfinite(float(n)):
            ax.text(1.012, i, f"n={int(n)}", va="center", ha="left", fontsize=7, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(data["family"].str.replace("_", " "))
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Share of family")
    ax.set_title("Usefulness labels by family", loc="left", pad=10)
    ax.legend(frameon=False, loc="lower right")
    _clean_axes(ax)


def _plot_family_advantage_ci(ax, family_ci: pd.DataFrame) -> None:
    data = family_ci[family_ci["family"] != "overall"].sort_values("mean_advantage", ascending=True)
    y = np.arange(len(data))
    mean = pd.to_numeric(data["mean_advantage"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(data["mean_ci_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(data["mean_ci_high"], errors="coerce").to_numpy(dtype=float)
    colors = [_interval_color(lo, hi) for lo, hi in zip(low, high)]
    ax.errorbar(mean, y, xerr=_xerr(mean, low, high), fmt="none", ecolor=MUTED, capsize=2.5, linewidth=1.0)
    ax.scatter(mean, y, color=colors, s=42, zorder=3, edgecolors="white", linewidths=0.5)
    ax.axvline(0.0, color=INK, linewidth=0.9)
    ax.axvline(0.05, color=BLUE, linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(data["family"].str.replace("_", " "))
    ax.set_xlabel("Mean QRC advantage, bootstrap 95% CI")
    ax.set_title("Family mean advantage intervals", loc="left", pad=10)
    _clean_axes(ax)


def _plot_prediction_scatter(ax, atlas: pd.DataFrame, meta: pd.DataFrame) -> None:
    colors = [CATEGORY_COLORS.get(str(v), MUTED) for v in atlas["actual_usefulness_label"]]
    ax.scatter(
        atlas["qrc_advantage"],
        atlas["predicted_qrc_advantage"],
        s=24,
        c=colors,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.25,
    )
    lo = float(np.nanmin([atlas["qrc_advantage"].min(), atlas["predicted_qrc_advantage"].min()]))
    hi = float(np.nanmax([atlas["qrc_advantage"].max(), atlas["predicted_qrc_advantage"].max()]))
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=0.9)
    ax.axhline(0.05, color=BLUE, linewidth=0.8, linestyle="--")
    ax.axvline(0.05, color=BLUE, linewidth=0.8, linestyle="--")
    row = meta.iloc[0]
    ax.text(
        0.03,
        0.97,
        f"CV R2={float(row['regression_r2_mean']):.2f}\nCV AUC={float(row['classification_roc_auc_mean']):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=INK,
        bbox={"facecolor": PANEL, "edgecolor": AXIS, "boxstyle": "round,pad=0.28", "linewidth": 0.7},
    )
    ax.set_xlabel("Observed QRC advantage")
    ax.set_ylabel("Predicted QRC advantage")
    ax.set_title("Predicted vs observed usefulness", loc="left", pad=10)
    _clean_axes(ax)


def _plot_importance_ci(ax, importance: pd.DataFrame) -> None:
    data = importance.sort_values("importance_mean", ascending=False).head(12).iloc[::-1]
    y = np.arange(len(data))
    means = pd.to_numeric(data["importance_mean"], errors="coerce").to_numpy(dtype=float)
    colors = [BLUE if str(v).lower() == "positive" else ORANGE if str(v).lower() == "negative" else GOLD for v in data.get("direction", [""] * len(data))]
    ax.barh(y, means, color=colors, height=0.72, alpha=0.92)
    if {"ci_low", "ci_high"}.issubset(data.columns):
        low = pd.to_numeric(data["ci_low"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(data["ci_high"], errors="coerce").to_numpy(dtype=float)
        ax.errorbar(means, y, xerr=_xerr(means, low, high), fmt="none", ecolor=INK, linewidth=0.8, capsize=2.0)
    ax.set_yticks(y)
    ax.set_yticklabels(_feature_labels(data["feature"]))
    ax.set_xlabel("Permutation importance")
    ax.set_title("Feature importance with bootstrap intervals", loc="left", pad=10)
    _clean_axes(ax)


def _plot_robustness_lines(ax, robustness: pd.DataFrame) -> None:
    data = robustness.copy()
    labels = [_feature_set_label(v) for v in data["feature_set"]]
    x = np.arange(len(data))
    ax.plot(x, data["regression_r2_mean"], marker="o", color=BLUE, label="CV R2", linewidth=1.8)
    ax.plot(x, data["classification_roc_auc_mean"], marker="s", color=GOLD, label="CV AUC", linewidth=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    lo = min(0.40, float(pd.to_numeric(data[["regression_r2_mean", "classification_roc_auc_mean"]].stack(), errors="coerce").min()) - 0.05)
    ax.set_ylim(lo, 0.95)
    ax.set_ylabel("Cross-validated score")
    ax.set_title("Anti-circularity feature-set checks", loc="left", pad=10)
    ax.legend(frameon=False, loc="lower right")
    _clean_axes(ax)


def _plot_prediction_error_by_family(ax, atlas: pd.DataFrame) -> None:
    data = atlas.copy()
    if "abs_prediction_error" not in data.columns:
        data["abs_prediction_error"] = (data["qrc_advantage"] - data["predicted_qrc_advantage"]).abs()
    summary = data.groupby("family", sort=True)["abs_prediction_error"].mean().sort_values(ascending=True)
    y = np.arange(len(summary))
    colors = [FAMILY_COLORS.get(str(f), MUTED) for f in summary.index]
    ax.barh(y, summary.to_numpy(dtype=float), color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([str(v).replace("_", " ") for v in summary.index])
    ax.set_xlabel("Mean absolute prediction error")
    ax.set_title("Residual size by family", loc="left", pad=10)
    _clean_axes(ax)


def _plot_attribution_intervals(ax, attribution: pd.DataFrame) -> None:
    data = attribution.sort_values("mean_delta_J0_minus_J1", ascending=True)
    y = np.arange(len(data))
    mean = pd.to_numeric(data["mean_delta_J0_minus_J1"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(data["delta_ci_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(data["delta_ci_high"], errors="coerce").to_numpy(dtype=float)
    colors = [_interval_color(lo, hi) for lo, hi in zip(low, high)]
    ax.errorbar(mean, y, xerr=_xerr(mean, low, high), fmt="none", ecolor=MUTED, capsize=2.5, linewidth=1.0)
    ax.scatter(mean, y, color=colors, s=46, zorder=3, edgecolors="white", linewidths=0.5)
    ax.axvline(0.0, color=INK, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(data["family"].str.replace("_", " "))
    ax.set_xlabel("NRMSE(J=0) - NRMSE(coupled)")
    ax.set_title("Paired coupling-control intervals", loc="left", pad=10)
    _clean_axes(ax)


def _annotate_centroids(ax, atlas: pd.DataFrame) -> None:
    centroids = atlas.groupby("family", sort=True)[["property_pc1", "property_pc2"]].mean()
    offsets = {
        "input_driven": (0.35, 0.26),
        "nonstationary": (-0.28, 0.28),
        "linear_stochastic": (-0.18, -0.18),
        "nonlinear_stochastic": (0.25, -0.20),
        "long_range": (0.0, -0.25),
        "real_bridge": (-0.25, 0.18),
    }
    for fam, row in centroids.iterrows():
        dx, dy = offsets.get(str(fam), (0.0, 0.0))
        ax.text(
            float(row["property_pc1"]) + dx,
            float(row["property_pc2"]) + dy,
            str(fam).replace("_", " "),
            fontsize=7,
            ha="center",
            va="center",
            color=INK,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": PANEL, "edgecolor": AXIS, "alpha": 0.86, "linewidth": 0.55},
        )


def _write_html_index(
    out_dir: Path,
    figures: list[dict[str, str]],
    atlas: pd.DataFrame,
    family: pd.DataFrame,
    meta: pd.DataFrame,
    attribution: pd.DataFrame,
) -> Path:
    counts = atlas["actual_usefulness_label"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    meta_row = meta.iloc[0]
    attr_overall = _row_or_first(attribution, "family", "overall")
    top_family = family.sort_values(["qrc_useful_rate", "mean_qrc_advantage"], ascending=[False, False]).iloc[0]
    cards = [
        ("Rows", f"{len(atlas):,}"),
        ("QRC-useful", f"{int(counts.get('qrc_useful', 0)):,}"),
        ("Near ties", f"{int(counts.get('near_tie', 0)):,}"),
        ("Baseline preferred", f"{int(counts.get('baseline_preferred', 0)):,}"),
        ("Meta R2", f"{float(meta_row['regression_r2_mean']):.3f}"),
        ("Meta AUC", f"{float(meta_row['classification_roc_auc_mean']):.3f}"),
    ]
    card_html = "\n".join(f"<div class='card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>" for label, value in cards)
    fig_html = "\n".join(
        f"""
        <section class="figure">
          <h2>{html.escape(item['title'])}</h2>
          <p>{html.escape(item['caption'])}</p>
          <img src="{html.escape(item['png'])}" alt="{html.escape(item['title'])}">
          <p class="links"><a href="{html.escape(item['png'])}">PNG</a>{' | <a href="' + html.escape(item.get('pdf', '')) + '">PDF</a>' if item.get('pdf') else ''}</p>
        </section>
        """
        for item in figures
        if item.get("png")
    )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QRC Usefulness Atlas Visual Suite</title>
  <style>
    :root {{
      color-scheme: light;
      --surface: #FCFCFD;
      --panel: #FFFFFF;
      --ink: #1F2430;
      --muted: #6F768A;
      --axis: #D7DBE7;
      --blue: #5477C4;
      --orange: #CC6F47;
    }}
    body {{
      margin: 0;
      background: var(--surface);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 36px min(6vw, 72px) 22px;
      border-bottom: 1px solid var(--axis);
      background: var(--panel);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(30px, 5vw, 52px);
      letter-spacing: 0;
    }}
    .sub {{
      max-width: 1040px;
      color: var(--muted);
      font-size: 16px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-top: 24px;
      max-width: 1120px;
    }}
    .card {{
      border: 1px solid var(--axis);
      background: var(--surface);
      padding: 12px 14px;
      border-radius: 8px;
    }}
    .card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .card strong {{
      display: block;
      margin-top: 4px;
      font-size: 24px;
    }}
    main {{
      padding: 28px min(6vw, 72px) 56px;
    }}
    .claim {{
      max-width: 1120px;
      margin: 0 0 28px;
      padding-left: 14px;
      border-left: 4px solid var(--orange);
      color: var(--muted);
    }}
    .figure {{
      margin: 0 0 34px;
      padding: 20px;
      background: var(--panel);
      border: 1px solid var(--axis);
      border-radius: 8px;
      max-width: 1320px;
    }}
    .figure h2 {{
      margin: 0 0 4px;
      font-size: 20px;
      letter-spacing: 0;
    }}
    .figure p {{
      margin: 0 0 14px;
      color: var(--muted);
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid #E6E8F0;
      border-radius: 4px;
      background: white;
    }}
    .links {{
      margin-top: 10px !important;
      font-size: 13px;
    }}
    a {{ color: var(--blue); text-decoration: none; }}
  </style>
</head>
<body>
  <header>
    <h1>QRC Usefulness Atlas Visual Suite</h1>
    <p class="sub">A reproducible figure package for the 1000-row sweep over 50 benchmark datasets. Strongest useful family: <strong>{html.escape(str(top_family['family']).replace('_', ' '))}</strong>; corrected coupling attribution remains <strong>{html.escape(str(attr_overall['mechanism_signal']))}</strong>.</p>
    <div class="cards">{card_html}</div>
  </header>
  <main>
    <p class="claim">Claim boundary: this suite supports dataset categorization and meta-model analysis. It does not claim broad average QRC superiority or a quantum coupling/entanglement mechanism.</p>
    {fig_html}
  </main>
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(doc, encoding="utf-8")
    return path


def _visual_report(
    atlas: pd.DataFrame,
    family: pd.DataFrame,
    meta: pd.DataFrame,
    family_ci: pd.DataFrame,
    robustness: pd.DataFrame,
    attribution: pd.DataFrame,
) -> str:
    counts = atlas["actual_usefulness_label"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    top_family = family.sort_values(["qrc_useful_rate", "mean_qrc_advantage"], ascending=[False, False]).iloc[0]
    meta_row = meta.iloc[0]
    overall_ci = _row_or_first(family_ci, "family", "overall")
    no_proxy = _row_or_first(robustness, "feature_set", "without_predictability_proxies")
    attr_overall = _row_or_first(attribution, "family", "overall")
    return f"""# QRC Usefulness Atlas Visual Suite

## Headline

- Rows: `{len(atlas)}`.
- QRC-useful / near-tie / baseline-preferred: `{int(counts['qrc_useful'])}` / `{int(counts['near_tie'])}` / `{int(counts['baseline_preferred'])}`.
- Strongest useful family: `{top_family['family']}` with qrc-useful rate `{float(top_family['qrc_useful_rate']):.3f}`.
- Overall mean QRC advantage CI: `[{float(overall_ci['mean_ci_low']):.4f}, {float(overall_ci['mean_ci_high']):.4f}]`.
- Meta-model CV R2 / AUC: `{float(meta_row['regression_r2_mean']):.4f}` / `{float(meta_row['classification_roc_auc_mean']):.4f}`.
- No-direct-proxy CV R2 / AUC: `{float(no_proxy['regression_r2_mean']):.4f}` / `{float(no_proxy['classification_roc_auc_mean']):.4f}`.
- Paired attribution overall J0-J1 delta CI: `[{float(attr_overall['delta_ci_low']):.4f}, {float(attr_overall['delta_ci_high']):.4f}]`, interpretation `{attr_overall['mechanism_signal']}`.

## Claim Boundary

The figures support a dataset-categorization and meta-model claim. They do not establish broad average QRC superiority, fundamental quantum advantage, or a coupling/entanglement mechanism.
"""


def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_optional(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _normalize_formats(formats: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in formats:
        for item in str(value).split(","):
            fmt = item.strip().lower().lstrip(".")
            if not fmt:
                continue
            if fmt not in {"png", "pdf", "svg"}:
                raise ValueError(f"unsupported figure format: {fmt}")
            if fmt not in out:
                out.append(fmt)
    return tuple(out or ["png"])


def _save_figure(fig, out_dir: Path, stem: str, formats: tuple[str, ...], caption: str) -> dict[str, str]:
    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout(rect=getattr(fig, "_qrc_tight_rect", (0.02, 0.02, 0.98, 0.90)))
        except Exception:
            pass
    item = {"stem": stem, "title": stem.replace("_", " ").title(), "caption": caption}
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight", "facecolor": fig.get_facecolor()}
        if fmt == "png":
            kwargs["dpi"] = 260
        fig.savefig(path, **kwargs)
        item[fmt] = path.name
    plt.close(fig)
    return item


def _figure_header(fig, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.02, y=0.985, ha="left", va="top", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.02, 0.946, subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)


def _plot_context():
    import matplotlib.pyplot as plt

    rc = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": PANEL,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    return plt.rc_context(rc)


def _clean_axes(ax) -> None:
    ax.grid(True, axis="x", alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def _row_or_first(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    if column in df.columns:
        sub = df[df[column] == value]
        if not sub.empty:
            return sub.iloc[0]
    return df.iloc[0]


def _legend_patch(color: str, label: str):
    from matplotlib.patches import Patch

    return Patch(facecolor=color, edgecolor="none", label=label)


def _xerr(mean: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    left = np.maximum(0.0, np.asarray(mean, dtype=float) - np.asarray(low, dtype=float))
    right = np.maximum(0.0, np.asarray(high, dtype=float) - np.asarray(mean, dtype=float))
    return np.vstack([left, right])


def _interval_color(low: float, high: float) -> str:
    if math.isfinite(float(low)) and low > 0:
        return BLUE
    if math.isfinite(float(high)) and high < 0:
        return ORANGE
    return MUTED


def _feature_set_label(value: object) -> str:
    mapping = {
        "all": "all",
        "without_r2_linear": "no r2_linear",
        "without_predictability_proxies": "no proxies",
        "chaos_nonlinearity_complexity_only": "complexity only",
    }
    return mapping.get(str(value), str(value).replace("_", " "))


def _clean_label(value: object) -> str:
    return str(value).replace("_", " ")


def _short_label(value: object, limit: int = 28) -> str:
    text = str(value).replace("_", " ")
    return text if len(text) <= limit else text[: limit - 1] + "."


def _feature_labels(values: Iterable[object]) -> list[str]:
    out = []
    replacements = {
        "ext_": "",
        "pred_nrmse": "pred nrmse",
        "r2": "r2",
        "nl": "nl",
        "ac": "ac",
        "psd": "psd",
        "fnn": "fnn",
        "bds": "bds",
        "cv2": "cv2",
    }
    for value in values:
        text = str(value)
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = text.replace("_", " ")
        out.append(_short_label(text, limit=30))
    return out


def _usefulness_label(value: float) -> str:
    if value >= 0.05:
        return "qrc_useful"
    if value >= -0.05:
        return "near_tie"
    return "baseline_preferred"


def _generator_label(params: object, name: object) -> str:
    parsed: dict[str, Any] = {}
    if isinstance(params, str) and params.strip():
        try:
            value = ast.literal_eval(params)
            if isinstance(value, dict):
                parsed = value
        except (SyntaxError, ValueError):
            parsed = {}
    elif isinstance(params, dict):
        parsed = params
    generator = str(parsed.get("generator") or _fallback_generator_name(name))
    if parsed.get("noise_overlay"):
        snr = parsed.get("snr_db")
        return f"{generator} + noise {snr:g}dB" if isinstance(snr, (int, float)) else f"{generator} + noise"
    return generator


def _fallback_generator_name(name: object) -> str:
    text = str(name)
    for marker in ("_s", "_w"):
        idx = text.rfind(marker)
        if idx > 0 and text[idx + 2 : idx + 5].isdigit():
            text = text[:idx]
            break
    return text


def _robust_scale(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return 1.0
    q75, q25 = np.nanpercentile(vals.to_numpy(dtype=float), [75, 25])
    iqr = float(q75 - q25)
    if not math.isfinite(iqr) or iqr <= 1e-12:
        std = float(np.nanstd(vals.to_numpy(dtype=float)))
        return std if math.isfinite(std) and std > 1e-12 else 1.0
    return iqr


def family_order_from_rows(rows: pd.DataFrame) -> dict[str, int]:
    order = (
        rows.groupby("family", sort=True)["qrc_advantage"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return {str(family): i for i, family in enumerate(order)}


def _advantage_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("qrc_advantage", ["#804126", "#F4F5F9", "#2E4780"], N=256)


def _feature_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("qrc_feature", ["#CC6F47", "#FFFFFF", "#5477C4"], N=256)


def _blue_gold_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("qrc_prob", ["#F7F0C6", "#A3BEFA", "#2E4780"], N=256)


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
