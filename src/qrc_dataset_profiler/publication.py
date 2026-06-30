"""Publication-facing figure and report assembly for the QRC atlas."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CATEGORY_COLORS = {
    "baseline_preferred": "#9a4f4f",
    "near_tie": "#6b7280",
    "qrc_useful": "#2f6f6f",
}


def run_publication_package(
    *,
    atlas_dir: Path,
    analysis_dir: Path,
    attribution_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Create multi-panel publication figures and a concise report."""

    out_dir.mkdir(parents=True, exist_ok=True)
    atlas = _load_required(atlas_dir / "qrc_usefulness_map.csv")
    family = _load_required(atlas_dir / "family_usefulness_summary.csv")
    meta = _load_required(atlas_dir / "meta_model_summary.csv")
    importances = _load_required(atlas_dir / "atlas_importances.csv")
    family_ci = _load_required(analysis_dir / "family_advantage_bootstrap.csv")
    robustness = _load_required(analysis_dir / "robustness_summary.csv")
    attribution = _load_required(attribution_dir / "family_attribution_bootstrap.csv")

    _write_fig1_atlas(atlas, family, meta, importances, out_dir / "fig1_qrc_usefulness_atlas")
    _write_fig2_evidence(family_ci, robustness, attribution, out_dir / "fig2_evidence_controls")
    report = _write_report(atlas, family, meta, family_ci, robustness, attribution, out_dir / "ATLAS_REPORT.md")

    manifest = {
        "analysis_version": "publication-package-v1",
        "inputs": {
            "atlas_dir": str(atlas_dir),
            "analysis_dir": str(analysis_dir),
            "attribution_dir": str(attribution_dir),
        },
        "n_rows": int(len(atlas)),
        "n_families": int(family["family"].nunique()),
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "claim_boundary": (
            "Publication figures support a dataset-categorization and meta-model claim. "
            "They do not claim broad average QRC superiority or fundamental quantum advantage."
        ),
        "report_headline": report["headline"],
    }
    (out_dir / "publication_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_fig1_atlas(atlas: pd.DataFrame, family: pd.DataFrame, meta: pd.DataFrame, importances: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
        ax_map, ax_family, ax_pred, ax_imp = axes.ravel()
        _panel_label(ax_map, "a")
        _plot_property_map(ax_map, atlas)
        _panel_label(ax_family, "b")
        _plot_family_categories(ax_family, family)
        _panel_label(ax_pred, "c")
        _plot_prediction(ax_pred, atlas, meta)
        _panel_label(ax_imp, "d")
        _plot_importances(ax_imp, importances)
        fig.savefig(stem.with_suffix(".png"), dpi=240)
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)


def _write_fig2_evidence(family_ci: pd.DataFrame, robustness: pd.DataFrame, attribution: pd.DataFrame, stem: Path) -> None:
    import matplotlib.pyplot as plt

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)
        _panel_label(axes[0], "a")
        _plot_family_ci(axes[0], family_ci)
        _panel_label(axes[1], "b")
        _plot_robustness(axes[1], robustness)
        _panel_label(axes[2], "c")
        _plot_attribution_control(axes[2], attribution)
        fig.savefig(stem.with_suffix(".png"), dpi=240)
        fig.savefig(stem.with_suffix(".pdf"))
        plt.close(fig)


def _plot_property_map(ax, atlas: pd.DataFrame) -> None:
    for label in ("baseline_preferred", "near_tie", "qrc_useful"):
        sub = atlas[atlas["actual_usefulness_label"] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["property_pc1"],
            sub["property_pc2"],
            s=18,
            color=CATEGORY_COLORS[label],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.25,
            label=label.replace("_", " "),
        )
    centroids = atlas.groupby("family", sort=True)[["property_pc1", "property_pc2"]].mean()
    offsets = {
        "input_driven": (0.62, 0.5),
        "nonstationary": (-0.52, 0.55),
        "linear_stochastic": (-0.2, -0.28),
        "nonlinear_stochastic": (0.18, -0.34),
        "long_range": (0.0, -0.3),
    }
    for family, coords in centroids.iterrows():
        dx, dy = offsets.get(str(family), (0.0, 0.0))
        ax.text(
            float(coords["property_pc1"]) + dx,
            float(coords["property_pc2"]) + dy,
            str(family).replace("_", " "),
            fontsize=6.5,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.78, "linewidth": 0.35},
        )
    ax.set_title("Dataset-property map")
    ax.set_xlabel("Property PC1")
    ax.set_ylabel("Property PC2")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    _clean(ax)


def _plot_family_categories(ax, family: pd.DataFrame) -> None:
    data = family.sort_values("qrc_useful_rate", ascending=True)
    y = np.arange(len(data))
    left = np.zeros(len(data))
    for label, col in (
        ("baseline_preferred", "baseline_preferred_rate"),
        ("near_tie", "near_tie_rate"),
        ("qrc_useful", "qrc_useful_rate"),
    ):
        vals = data[col].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, height=0.72, color=CATEGORY_COLORS[label], label=label.replace("_", " "))
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(data["family"].str.replace("_", " "))
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of family")
    ax.set_title("Usefulness categories by family")
    _clean(ax)


def _plot_prediction(ax, atlas: pd.DataFrame, meta: pd.DataFrame) -> None:
    colors = [CATEGORY_COLORS.get(v, "#6b7280") for v in atlas["actual_usefulness_label"]]
    ax.scatter(atlas["qrc_advantage"], atlas["predicted_qrc_advantage"], s=18, c=colors, alpha=0.82, edgecolors="white", linewidths=0.25)
    lo = float(np.nanmin([atlas["qrc_advantage"].min(), atlas["predicted_qrc_advantage"].min()]))
    hi = float(np.nanmax([atlas["qrc_advantage"].max(), atlas["predicted_qrc_advantage"].max()]))
    ax.plot([lo, hi], [lo, hi], color="#222222", linewidth=0.8)
    ax.axhline(0.05, color="#6b7280", linewidth=0.7, linestyle=":")
    ax.axvline(0.05, color="#6b7280", linewidth=0.7, linestyle=":")
    r2 = float(meta.loc[0, "regression_r2_mean"])
    auc = float(meta.loc[0, "classification_roc_auc_mean"])
    ax.text(0.03, 0.97, f"CV R2={r2:.2f}\nCV AUC={auc:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.set_title("Predicted vs observed usefulness")
    ax.set_xlabel("Observed QRC advantage")
    ax.set_ylabel("Predicted QRC advantage")
    _clean(ax)


def _plot_importances(ax, importances: pd.DataFrame) -> None:
    data = importances.head(10).iloc[::-1]
    ax.barh(data["feature"], data["importance_mean"], color="#4c6f7f", height=0.72)
    ax.set_title("Drivers of QRC usefulness")
    ax.set_xlabel("Permutation importance")
    _clean(ax)


def _plot_family_ci(ax, family_ci: pd.DataFrame) -> None:
    data = family_ci[family_ci["family"] != "overall"].sort_values("mean_advantage", ascending=True)
    y = np.arange(len(data))
    mean = data["mean_advantage"].to_numpy(dtype=float)
    low = data["mean_ci_low"].to_numpy(dtype=float)
    high = data["mean_ci_high"].to_numpy(dtype=float)
    colors = ["#2f6f6f" if lo > 0 else "#9a4f4f" if hi < 0 else "#6b7280" for lo, hi in zip(low, high)]
    ax.errorbar(mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="none", ecolor="#6b7280", capsize=2, linewidth=0.8)
    ax.scatter(mean, y, color=colors, s=24, zorder=3)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["family"].str.replace("_", " "))
    ax.set_xlabel("Mean QRC advantage")
    ax.set_title("Family-level advantage intervals")
    _clean(ax)


def _plot_robustness(ax, robustness: pd.DataFrame) -> None:
    label_map = {
        "all": "all",
        "without_r2_linear": "no r2",
        "without_predictability_proxies": "no proxies",
        "chaos_nonlinearity_complexity_only": "complexity",
    }
    labels = [label_map.get(str(v), str(v)) for v in robustness["feature_set"]]
    x = np.arange(len(robustness))
    ax.plot(x, robustness["regression_r2_mean"], marker="o", color="#2f6f6f", label="CV R2", linewidth=1.4)
    ax.plot(x, robustness["classification_roc_auc_mean"], marker="s", color="#7b5e2f", label="CV AUC", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.45, 0.92)
    ax.set_ylabel("Score")
    ax.set_title("Anti-circularity checks")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _clean(ax)


def _plot_attribution_control(ax, attribution: pd.DataFrame) -> None:
    data = attribution.sort_values("mean_delta_J0_minus_J1", ascending=True)
    y = np.arange(len(data))
    mean = data["mean_delta_J0_minus_J1"].to_numpy(dtype=float)
    low = data["delta_ci_low"].to_numpy(dtype=float)
    high = data["delta_ci_high"].to_numpy(dtype=float)
    colors = ["#2f6f6f" if lo > 0 else "#9a4f4f" if hi < 0 else "#6b7280" for lo, hi in zip(low, high)]
    ax.errorbar(mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="none", ecolor="#6b7280", capsize=2, linewidth=0.8)
    ax.scatter(mean, y, color=colors, s=24, zorder=3)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["family"].str.replace("_", " "))
    ax.set_xlabel("NRMSE(J=0) - NRMSE(coupled)")
    ax.set_title("Coupling attribution control")
    _clean(ax)


def _write_report(
    atlas: pd.DataFrame,
    family: pd.DataFrame,
    meta: pd.DataFrame,
    family_ci: pd.DataFrame,
    robustness: pd.DataFrame,
    attribution: pd.DataFrame,
    path: Path,
) -> dict[str, str]:
    counts = atlas["actual_usefulness_label"].value_counts().to_dict()
    top_family = family.sort_values(["qrc_useful_rate", "mean_qrc_advantage"], ascending=[False, False]).iloc[0]
    meta_row = meta.iloc[0]
    overall_ci = family_ci[family_ci["family"] == "overall"].iloc[0]
    no_proxy = robustness[robustness["feature_set"] == "without_predictability_proxies"].iloc[0]
    attr_overall = attribution[attribution["family"] == "overall"].iloc[0]
    row_agreement = float(pd.to_numeric(atlas.get("prediction_correct_label", pd.Series(dtype=float)), errors="coerce").mean())
    if "source" in atlas:
        source_counts = {str(k): int(v) for k, v in atlas["source"].value_counts().sort_index().items()}
    else:
        real_bridge_count = int((atlas.get("family", pd.Series(dtype=str)) == "real_bridge").sum())
        source_counts = {"synthetic": int(len(atlas) - real_bridge_count)}
        if real_bridge_count:
            source_counts["real"] = real_bridge_count
    source_text = ", ".join(f"{key}={value}" for key, value in source_counts.items())
    regime_claim = (
        "synthetic and real-bridge time-series regimes"
        if any(key != "synthetic" and value > 0 for key, value in source_counts.items())
        else "synthetic time-series regimes"
    )
    headline = (
        f"The {len(atlas)}-row atlas identifies {int(counts.get('qrc_useful', 0))} QRC-useful datasets, "
        f"with {top_family['family']} as the clearest useful family."
    )
    body = f"""# QRC Usefulness Atlas Report

## Headline

{headline}

## Dataset And Labels

- Atlas size: `{len(atlas)}` datasets from the parameterized sweep ({source_text}).
- `qrc_useful`: `{int(counts.get('qrc_useful', 0))}` rows.
- `near_tie`: `{int(counts.get('near_tie', 0))}` rows.
- `baseline_preferred`: `{int(counts.get('baseline_preferred', 0))}` rows.
- Label target: `qrc_advantage = nrmse_esn_matched - nrmse_qrc_spin`.

## Main Map Result

- Strongest QRC-useful family: `{top_family['family']}` with qrc-useful rate `{float(top_family['qrc_useful_rate']):.3f}`.
- Overall mean QRC advantage CI: `[{float(overall_ci['mean_ci_low']):.4f}, {float(overall_ci['mean_ci_high']):.4f}]`.
- Atlas in-sample row-level category agreement: `{row_agreement:.3f}`.

## Meta-Model

- Cross-validated regression R2: `{float(meta_row['regression_r2_mean']):.4f}`.
- Cross-validated ROC-AUC for qrc-useful classification: `{float(meta_row['classification_roc_auc_mean']):.4f}`.
- Top features: `{meta_row['top_features']}`.
- Anti-circularity without direct predictability proxies: R2 `{float(no_proxy['regression_r2_mean']):.4f}`, ROC-AUC `{float(no_proxy['classification_roc_auc_mean']):.4f}`.

## Attribution Control

- Corrected paired coupled-vs-J=0 overall effect: `{float(attr_overall['mean_delta_J0_minus_J1']):.4f}`.
- 95% CI: `[{float(attr_overall['delta_ci_low']):.4f}, {float(attr_overall['delta_ci_high']):.4f}]`.
- Interpretation: `{attr_overall['mechanism_signal']}`.

## Claim Boundary

This supports a dataset-categorization claim: the fixed Spin-QRC is selectively useful in identifiable {regime_claim}. It does not establish broad average QRC superiority, fundamental quantum advantage, or a coupling/entanglement mechanism.
"""
    path.write_text(body, encoding="utf-8")
    return {"headline": headline}


def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")


def _clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _rc() -> dict[str, Any]:
    return {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


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
