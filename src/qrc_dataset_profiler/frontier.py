"""Frontier atlas utilities for the conditional QRC-advantage regime map.

This layer is deliberately upstream of expensive QRC/ESN labels. It can build a
large property-only synthetic atlas, freeze a target-free discovery/validation
selection, and then analyze evaluated rows with the 30 Tier-A feature set.
"""

from __future__ import annotations

import ast
import json
import math
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree

from qrc_dataset_profiler.analysis import load_catalog
from qrc_dataset_profiler.generators import generate, make_sweep_specs, make_sweep_specs_v4
from qrc_dataset_profiler.meta_model import fit_meta_model
from qrc_dataset_profiler.properties import compute_backstop, profile_dataset
from qrc_dataset_profiler.run_study import _study_spec, build_catalog
from qrc_dataset_profiler.spec import DatasetSpec, FRONTIER_TIER_A_FIELDS, TIER_A_UPGRADE_FIELDS


FRONTIER_ATLAS_VERSION = "frontier-atlas-v1"
FRONTIER_V4_ATLAS_VERSION = "frontier-atlas-v4"
DEFAULT_PROPERTY_N_PER_TEMPLATE = 400  # 50 sweep slots * 400 = 20000 rows.
DEFAULT_DISCOVERY_EVALUATED_ROWS = 5000
DEFAULT_VALIDATION_EVALUATED_ROWS = 5000

IDENTITY_COLUMNS = (
    "dataset_id",
    "name",
    "family",
    "source",
    "task_type",
    "params",
    "seed",
    "length",
    "horizon",
    "base_generator",
)
TARGET_COLUMNS = (
    "nrmse_linear",
    "nrmse_esn_matched",
    "nrmse_qrc_spin",
    "nrmse_gbm",
    "qrc_advantage",
)
DIRECT_PREDICTABILITY_PROXIES_30 = (
    "r2_linear",
    "forecastability",
    "pred_nrmse_gbm",
    "predictability_gap_linear_gbm",
)


def build_property_atlas(
    specs: list[DatasetSpec],
    *,
    fast: bool = True,
    smoke: bool = False,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Generate datasets once and compute the 30-feature property atlas.

    This does not run QRC or ESN targets. It computes the core profile and the
    deterministic extended descriptors in one pass over each generated series.
    """

    rows: list[dict[str, Any]] = []
    selected_specs = specs[: int(max_rows)] if max_rows is not None else specs
    for base_spec in selected_specs:
        row = _property_row_from_spec(base_spec, fast=fast, smoke=smoke)
        if row is None:
            continue
        rows.append(row)
    return materialize_frontier_features(pd.DataFrame(rows))


def _property_row_from_spec(base_spec: DatasetSpec, *, fast: bool, smoke: bool) -> dict[str, Any] | None:
    spec = _study_spec(base_spec, smoke=smoke, fast=fast)
    ds = generate(spec)
    if ds.ground_truth.get("_unavailable"):
        warnings.warn(f"skipping unavailable dataset {spec.name}", RuntimeWarning)
        return None
    rec = profile_dataset(ds)
    source = ds.inputs if spec.task_type == "input_driven" and ds.inputs is not None else ds.series
    extended = compute_backstop(source)
    row = {
        **rec.to_row(),
        **extended,
    }
    row["base_generator"] = infer_base_generator(row)
    return row


def write_property_atlas(
    *,
    out_dir: Path,
    n_per_template: int = DEFAULT_PROPERTY_N_PER_TEMPLATE,
    seed: int = 0,
    fast: bool = True,
    smoke: bool = False,
    max_rows: int | None = None,
    taxonomy: str = "v3",
    checkpoint_every: int = 0,
) -> tuple[pd.DataFrame, Path]:
    """Write a synthetic property-only frontier atlas and manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if taxonomy not in {"v3", "v4"}:
        raise ValueError("taxonomy must be 'v3' or 'v4'")
    specs = make_sweep_specs_v4(n_per_template, seed=seed) if taxonomy == "v4" else make_sweep_specs(n_per_template, seed=seed)
    if int(checkpoint_every) > 0:
        df = _build_property_atlas_checkpointed(
            specs,
            out_dir=out_dir,
            fast=fast,
            smoke=smoke,
            max_rows=max_rows,
            checkpoint_every=int(checkpoint_every),
        )
    else:
        df = build_property_atlas(specs, fast=fast, smoke=smoke, max_rows=max_rows)
    path = out_dir / "frontier_property_atlas.csv"
    df.to_csv(path, index=False)
    manifest = {
        "analysis_version": FRONTIER_V4_ATLAS_VERSION if taxonomy == "v4" else FRONTIER_ATLAS_VERSION,
        "artifact": "property_atlas",
        "taxonomy": taxonomy,
        "n_rows": int(len(df)),
        "requested_n_per_template": int(n_per_template),
        "sweep_seed": int(seed),
        "fast": bool(fast),
        "smoke": bool(smoke),
        "max_rows": None if max_rows is None else int(max_rows),
        "checkpoint_every": int(checkpoint_every),
        "n_tier_a_features": int(len(FRONTIER_TIER_A_FIELDS)),
        "tier_a_features": list(FRONTIER_TIER_A_FIELDS),
        "families": _count_dict(df.get("family")),
        "base_generators": _count_dict(df.get("base_generator")),
        "claim_boundary": "Property-only candidate atlas. No QRC/ESN targets are used for candidate selection.",
        "outputs": ["frontier_property_atlas.csv", "frontier_property_manifest.json"],
    }
    (out_dir / "frontier_property_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return df, path


def _build_property_atlas_checkpointed(
    specs: list[DatasetSpec],
    *,
    out_dir: Path,
    fast: bool,
    smoke: bool,
    max_rows: int | None,
    checkpoint_every: int,
) -> pd.DataFrame:
    selected_specs = specs[: int(max_rows)] if max_rows is not None else specs
    partial_path = out_dir / "frontier_property_atlas_partial.csv"
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if partial_path.exists():
        partial = load_catalog(partial_path)
        rows = partial.to_dict(orient="records")
        seen = set(partial.get("dataset_id", pd.Series(dtype=str)).astype(str))
        print(f"property checkpoint reuse rows={len(rows)} path={partial_path}", flush=True)
    checkpoint_every = max(1, int(checkpoint_every))
    for idx, base_spec in enumerate(selected_specs, start=1):
        study_spec = _study_spec(base_spec, smoke=smoke, fast=fast)
        expected_id = f"{study_spec.name}:{study_spec.seed}:{study_spec.length}"
        if expected_id in seen:
            continue
        row = _property_row_from_spec(base_spec, fast=fast, smoke=smoke)
        if row is not None:
            rows.append(row)
            seen.add(str(row["dataset_id"]))
        if idx % checkpoint_every == 0:
            materialize_frontier_features(pd.DataFrame(rows)).to_csv(partial_path, index=False)
            print(f"property checkpoint rows={len(rows)} processed={idx}/{len(selected_specs)}", flush=True)
    df = materialize_frontier_features(pd.DataFrame(rows))
    df.to_csv(partial_path, index=False)
    return df


def build_frontier_30_table(
    catalog: pd.DataFrame,
    extended_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join existing sweep and extended feature artifacts into one 30-feature table."""

    base = catalog.copy()
    if extended_features is not None:
        ext = extended_features.copy()
        if "dataset_id" not in base.columns or "dataset_id" not in ext.columns:
            raise ValueError("both catalog and extended_features need dataset_id for deterministic joining")
        duplicate_cols = [c for c in ext.columns if c in base.columns and c != "dataset_id"]
        ext = ext.drop(columns=duplicate_cols)
        base = base.merge(ext, on="dataset_id", how="left", validate="one_to_one")
    out = materialize_frontier_features(base)
    keep = [c for c in IDENTITY_COLUMNS if c in out.columns]
    keep += [c for c in FRONTIER_TIER_A_FIELDS if c in out.columns]
    keep += [c for c in TARGET_COLUMNS if c in out.columns]
    keep += [c for c in ("evaluation_split", "selection_role", "frontier_score") if c in out.columns]
    return out.loc[:, list(dict.fromkeys(keep))]


def write_frontier_30_table(
    *,
    catalog_path: Path,
    out_dir: Path,
    extended_features_path: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Write the joined 30-feature table from existing evaluated artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(catalog_path)
    extended = load_catalog(extended_features_path) if extended_features_path is not None else None
    df = build_frontier_30_table(catalog, extended)
    path = out_dir / "frontier_30_features.csv"
    df.to_csv(path, index=False)
    manifest = {
        "analysis_version": FRONTIER_ATLAS_VERSION,
        "artifact": "frontier_30_features",
        "catalog_path": str(catalog_path),
        "extended_features_path": None if extended_features_path is None else str(extended_features_path),
        "n_rows": int(len(df)),
        "n_tier_a_features": int(len(FRONTIER_TIER_A_FIELDS)),
        "tier_a_features": list(FRONTIER_TIER_A_FIELDS),
        "available_tier_a_features": [c for c in FRONTIER_TIER_A_FIELDS if c in df.columns],
        "missing_tier_a_features": [c for c in FRONTIER_TIER_A_FIELDS if c not in df.columns],
        "claim_boundary": "Thirty-feature explanatory table; target columns are carried only for evaluated catalogs.",
        "outputs": ["frontier_30_features.csv", "frontier_30_features_manifest.json"],
    }
    (out_dir / "frontier_30_features_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return df, path


def materialize_frontier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all declared Tier-A fields exist when source columns permit it."""

    out = df.copy()
    if "base_generator" not in out.columns:
        out["base_generator"] = [infer_base_generator(row._asdict() if hasattr(row, "_asdict") else row.to_dict()) for _, row in out.iterrows()]
    if "predictability_gap_linear_gbm" not in out.columns:
        out["predictability_gap_linear_gbm"] = np.nan
    if out["predictability_gap_linear_gbm"].isna().all():
        if {"pred_nrmse_linear", "pred_nrmse_gbm"}.issubset(out.columns):
            out["predictability_gap_linear_gbm"] = pd.to_numeric(out["pred_nrmse_linear"], errors="coerce") - pd.to_numeric(out["pred_nrmse_gbm"], errors="coerce")
        elif {"nrmse_linear", "nrmse_gbm"}.issubset(out.columns):
            out["predictability_gap_linear_gbm"] = pd.to_numeric(out["nrmse_linear"], errors="coerce") - pd.to_numeric(out["nrmse_gbm"], errors="coerce")
        elif "nl_gain" in out.columns:
            out["predictability_gap_linear_gbm"] = pd.to_numeric(out["nl_gain"], errors="coerce")
    for col in FRONTIER_TIER_A_FIELDS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def select_evaluation_atlas(
    property_atlas: pd.DataFrame,
    *,
    n_discovery: int = DEFAULT_DISCOVERY_EVALUATED_ROWS,
    n_validation: int = DEFAULT_VALIDATION_EVALUATED_ROWS,
    seed: int = 0,
    selection_protocol: str = "v3",
) -> pd.DataFrame:
    """Freeze a target-free discovery/validation selection from the property atlas."""

    if selection_protocol not in {"v3", "v4"}:
        raise ValueError("selection_protocol must be 'v3' or 'v4'")
    df = materialize_frontier_features(property_atlas).copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("property_atlas is empty")
    if "source" in df.columns and set(df["source"].dropna().astype(str)) - {"synthetic"}:
        raise ValueError("frontier primary atlas must be synthetic-only")
    df["frontier_score"] = frontier_enrichment_score(df)
    df["candidate_pool_split"] = _split_candidate_pool(df, seed=seed)
    df["evaluation_split"] = "unselected"
    df["selection_role"] = ""
    df["selection_order"] = np.nan

    selected_frames: list[pd.DataFrame] = []
    for split_name, n_rows in (("discovery", n_discovery), ("validation", n_validation)):
        pool = df[df["candidate_pool_split"] == split_name].copy()
        selected = _select_subset(
            pool,
            n_rows=int(n_rows),
            seed=seed + (0 if split_name == "discovery" else 100_000),
            split_name=split_name,
            selection_protocol=selection_protocol,
        )
        selected_frames.append(selected)

    if selected_frames:
        selected_all = pd.concat(selected_frames, ignore_index=True)
        for _, row in selected_all.iterrows():
            idx = int(row["__index__"])
            df.loc[idx, "evaluation_split"] = row["evaluation_split"]
            df.loc[idx, "selection_role"] = row["selection_role"]
            df.loc[idx, "selection_order"] = row["selection_order"]
    return df


def write_evaluation_selection(
    *,
    property_atlas_path: Path,
    out_dir: Path,
    n_discovery: int = DEFAULT_DISCOVERY_EVALUATED_ROWS,
    n_validation: int = DEFAULT_VALIDATION_EVALUATED_ROWS,
    seed: int = 0,
    selection_protocol: str = "v3",
) -> tuple[pd.DataFrame, Path]:
    """Write the frozen 5k+5k target-free evaluation selection."""

    out_dir.mkdir(parents=True, exist_ok=True)
    atlas = load_catalog(property_atlas_path)
    df = select_evaluation_atlas(atlas, n_discovery=n_discovery, n_validation=n_validation, seed=seed, selection_protocol=selection_protocol)
    path = out_dir / "frontier_evaluation_selection.csv"
    df.to_csv(path, index=False)
    selected = df[df["evaluation_split"].isin(["discovery", "validation"])]
    manifest = {
        "analysis_version": FRONTIER_ATLAS_VERSION,
        "artifact": "evaluation_selection",
        "property_atlas_path": str(property_atlas_path),
        "n_property_rows": int(len(df)),
        "n_selected": int(len(selected)),
        "n_discovery": int((selected["evaluation_split"] == "discovery").sum()),
        "n_validation": int((selected["evaluation_split"] == "validation").sum()),
        "selection_protocol": selection_protocol,
        "selection_uses_targets": False,
        "target_columns_ignored": [c for c in TARGET_COLUMNS if c in df.columns],
        "selection_roles": _count_dict(selected.get("selection_role")),
        "families_selected": _count_dict(selected.get("family")),
        "claim_boundary": "Selection is target-free and must be frozen before QRC/ESN labels are computed.",
        "outputs": ["frontier_evaluation_selection.csv", "frontier_selection_manifest.json"],
    }
    (out_dir / "frontier_selection_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return df, path


def run_frontier_regime_analysis(
    evaluated_table: pd.DataFrame,
    *,
    out_dir: Path,
    seed: int = 0,
    win_threshold: float = 0.05,
) -> dict[str, Any]:
    """Analyze evaluated rows with the 30-feature frontier model suite."""

    out_dir.mkdir(parents=True, exist_ok=True)
    df = materialize_frontier_features(evaluated_table)
    if "qrc_advantage" not in df.columns:
        raise ValueError("evaluated_table must include qrc_advantage")
    df["qrc_advantage"] = pd.to_numeric(df["qrc_advantage"], errors="coerce")
    evaluated = df[np.isfinite(df["qrc_advantage"].to_numpy(dtype=float))].reset_index(drop=True)
    if len(evaluated) < 20:
        raise ValueError("need at least 20 evaluated rows with finite qrc_advantage")

    result = fit_meta_model(evaluated, seed=seed, win_threshold=win_threshold, feature_fields=FRONTIER_TIER_A_FIELDS)
    result.ranked_importances.to_csv(out_dir / "frontier_importances.csv", index=False)
    summary = _frontier_meta_summary(evaluated, result, win_threshold=win_threshold)
    summary.to_csv(out_dir / "frontier_meta_summary.csv", index=False)
    grouped = grouped_validation_summary(evaluated, seed=seed, win_threshold=win_threshold)
    grouped.to_csv(out_dir / "frontier_grouped_validation.csv", index=False)
    rules = rule_table_from_result(result, raw_features=evaluated, win_threshold=win_threshold, seed=seed)
    rules.to_csv(out_dir / "frontier_rule_table.csv", index=False)
    _write_frontier_importance_plot(result.ranked_importances, out_dir / "frontier_importances.png")
    _write_regime_projection(evaluated, result, out_dir / "frontier_regime_projection.png", win_threshold=win_threshold)

    manifest = {
        "analysis_version": FRONTIER_ATLAS_VERSION,
        "artifact": "regime_analysis",
        "seed": int(seed),
        "win_threshold": float(win_threshold),
        "n_rows": int(len(evaluated)),
        "n_tier_a_features": int(len(FRONTIER_TIER_A_FIELDS)),
        "features_used_after_preprocessing": list(result.features_used),
        "top_features": result.ranked_importances["feature"].head(10).astype(str).tolist() if not result.ranked_importances.empty else [],
        "summary": summary.to_dict(orient="records"),
        "claim_boundary": "Regime-map analysis supports conditional protocol-local QRC usefulness claims only.",
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()) + ["frontier_regime_manifest.json"],
    }
    (out_dir / "frontier_regime_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def write_frontier_regime_analysis(
    *,
    evaluated_table_path: Path,
    out_dir: Path,
    seed: int = 0,
    win_threshold: float = 0.05,
) -> dict[str, Any]:
    return run_frontier_regime_analysis(load_catalog(evaluated_table_path), out_dir=out_dir, seed=seed, win_threshold=win_threshold)


def write_support_scores(
    *,
    discovery_table_path: Path,
    target_table_path: Path,
    out_dir: Path,
    k_values: tuple[int, ...] = (15, 30, 50),
) -> tuple[pd.DataFrame, Path]:
    """Fit discovery-only atlas support scores and apply them to target rows."""

    out_dir.mkdir(parents=True, exist_ok=True)
    discovery = materialize_frontier_features(load_catalog(discovery_table_path))
    target = materialize_frontier_features(load_catalog(target_table_path))
    support = compute_support_scores(discovery, target, k_values=k_values)
    path = out_dir / "frontier_support_scores.csv"
    support.to_csv(path, index=False)
    manifest = {
        "analysis_version": FRONTIER_ATLAS_VERSION,
        "artifact": "atlas_support_scores",
        "discovery_table_path": str(discovery_table_path),
        "target_table_path": str(target_table_path),
        "n_discovery": int(len(discovery)),
        "n_target": int(len(target)),
        "k_values": [int(k) for k in k_values],
        "n_ood": int(support["ood_flag"].sum()) if "ood_flag" in support else 0,
        "claim_boundary": "Support scores are discovery-fitted validity guardrails, not proof that a prediction is correct.",
        "outputs": ["frontier_support_scores.csv", "frontier_support_manifest.json"],
    }
    (out_dir / "frontier_support_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return support, path


def compute_support_scores(
    discovery: pd.DataFrame,
    target: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = (15, 30, 50),
) -> pd.DataFrame:
    """Discovery-fitted kNN/PCA atlas support score for target rows."""

    features = [c for c in FRONTIER_TIER_A_FIELDS if c in discovery.columns and c in target.columns]
    if not features:
        raise ValueError("no shared Tier-A features for support scoring")
    Xd_df = discovery[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xt_df = target[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = Xd_df.median(numeric_only=True).fillna(0.0)
    Xd_df = Xd_df.fillna(med)
    Xt_df = Xt_df.fillna(med)
    q75 = Xd_df.quantile(0.75)
    q25 = Xd_df.quantile(0.25)
    iqr = (q75 - q25).replace(0.0, np.nan)
    keep = [c for c in features if np.isfinite(float(iqr.get(c, np.nan))) and abs(float(iqr.get(c, np.nan))) > 1e-12]
    if not keep:
        keep = features
        iqr = pd.Series(1.0, index=features)
    Xd_scaled = ((Xd_df[keep] - med[keep]) / iqr[keep].fillna(1.0)).to_numpy(dtype=float)
    Xt_scaled = ((Xt_df[keep] - med[keep]) / iqr[keep].fillna(1.0)).to_numpy(dtype=float)
    keep_idx = _noncollinear_feature_indices(Xd_scaled)
    Xd_scaled = Xd_scaled[:, keep_idx]
    Xt_scaled = Xt_scaled[:, keep_idx]
    used_features = [keep[i] for i in keep_idx]

    n_disc = Xd_scaled.shape[0]
    if n_disc < 3:
        raise ValueError("need at least 3 discovery rows for support scoring")
    max_k = min(max(int(k) for k in k_values), n_disc - 1)
    raw_ref = _self_knn_mean_distances(Xd_scaled, max_k)
    raw_target = _target_knn_mean_distances(Xd_scaled, Xt_scaled, max_k)
    raw_percentile = _distance_percentiles(raw_target, raw_ref)

    n_comp = min(10, Xd_scaled.shape[1], n_disc - 1)
    if n_comp >= 1:
        pca = PCA(n_components=n_comp, random_state=0).fit(Xd_scaled)
        Xd_pca = pca.transform(Xd_scaled)
        Xt_pca = pca.transform(Xt_scaled)
        pca_ref = _self_knn_mean_distances(Xd_pca, max_k)
        pca_target = _target_knn_mean_distances(Xd_pca, Xt_pca, max_k)
        pca_percentile = _distance_percentiles(pca_target, pca_ref)
    else:
        pca_percentile = raw_percentile

    conservative_percentile = np.maximum(raw_percentile, pca_percentile)
    support_score = 1.0 - conservative_percentile
    family_mix, family_entropy = _nearest_family_mix(discovery, Xd_scaled, Xt_scaled, k=min(15, n_disc))

    identity_cols = [c for c in IDENTITY_COLUMNS if c in target.columns]
    out = target[identity_cols].copy()
    out["support_score"] = np.clip(support_score, 0.0, 1.0)
    out["ood_flag"] = out["support_score"] < 0.05
    out["raw_distance_percentile"] = raw_percentile
    out["pca_distance_percentile"] = pca_percentile
    out["family_entropy"] = family_entropy
    out["nearest_family_mixture"] = family_mix
    out["support_features_used"] = ",".join(used_features)
    return out


def specs_from_selection(selection: pd.DataFrame, *, split: str = "discovery") -> list[DatasetSpec]:
    """Reconstruct DatasetSpec rows from a frozen frontier selection table."""

    if split not in {"discovery", "validation", "all"}:
        raise ValueError("split must be 'discovery', 'validation', or 'all'")
    df = selection.copy()
    if "evaluation_split" in df.columns and split != "all":
        df = df[df["evaluation_split"] == split]
    elif "evaluation_split" in df.columns:
        df = df[df["evaluation_split"].isin(["discovery", "validation"])]
    specs: list[DatasetSpec] = []
    for row in df.itertuples(index=False):
        data = row._asdict()
        params = _parse_params(data.get("params", {}))
        specs.append(
            DatasetSpec(
                name=str(data.get("name")),
                family=str(data.get("family")),
                source=str(data.get("source", "synthetic")),
                task_type=str(data.get("task_type", "forecast")),
                params=params,
                seed=int(data.get("seed", 0)),
                length=int(data.get("length", 4000)),
                horizon=int(data.get("horizon", 1)) if np.isfinite(float(data.get("horizon", 1))) else 1,
            )
        )
    return specs


def selected_rows_from_selection(selection: pd.DataFrame, *, split: str = "discovery") -> pd.DataFrame:
    """Return target-free selected rows in the same order as ``specs_from_selection``."""

    if split not in {"discovery", "validation", "all"}:
        raise ValueError("split must be 'discovery', 'validation', or 'all'")
    selected = selection.copy()
    if "evaluation_split" in selected.columns and split != "all":
        selected = selected[selected["evaluation_split"] == split]
    elif "evaluation_split" in selected.columns:
        selected = selected[selected["evaluation_split"].isin(["discovery", "validation"])]
    return selected.drop(columns=[c for c in TARGET_COLUMNS if c in selected.columns], errors="ignore").reset_index(drop=True)


def write_evaluated_selection(
    *,
    selection_path: Path,
    out_dir: Path,
    split: str = "discovery",
    comparison_protocol: str = "standard_v3",
    calibration_config: Path | None = None,
    fast: bool = True,
    smoke: bool = False,
    seeds: int | None = None,
    checkpoint_every: int = 0,
) -> tuple[pd.DataFrame, Path]:
    """Evaluate selected frontier rows with the frozen QRC/ESN study runner.

    The output preserves the precomputed 30-feature selection rows and only merges in
    target columns from the expensive evaluator.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    selection = load_catalog(selection_path)
    if int(checkpoint_every) > 0:
        return _write_evaluated_selection_checkpointed(
            selection,
            selection_path=selection_path,
            out_dir=out_dir,
            split=split,
            comparison_protocol=comparison_protocol,
            calibration_config=calibration_config,
            fast=fast,
            smoke=smoke,
            seeds=seeds,
            checkpoint_every=int(checkpoint_every),
        )
    specs = specs_from_selection(selection, split=split)
    if not specs:
        raise ValueError(f"no selected specs found for split={split}")
    output_stem = f"frontier_{split}_catalog"
    eval_df, catalog_path, qrc_cfg, seed_count = build_catalog(
        specs,
        out_dir=out_dir,
        fast=fast,
        smoke=smoke,
        seeds=seeds,
        output_stem=output_stem,
        comparison_protocol=comparison_protocol,
        calibration_config=calibration_config,
    )
    selected = selected_rows_from_selection(selection, split=split)
    targets = eval_df.loc[:, ["dataset_id", *[c for c in TARGET_COLUMNS if c in eval_df.columns]]]
    joined = selected.merge(targets, on="dataset_id", how="left", validate="one_to_one")
    joined = materialize_frontier_features(joined)
    path = out_dir / f"frontier_{split}_evaluated_30_features.csv"
    joined.to_csv(path, index=False)
    manifest = {
        "analysis_version": FRONTIER_ATLAS_VERSION,
        "artifact": "evaluated_selection",
        "selection_path": str(selection_path),
        "split": split,
        "comparison_protocol": comparison_protocol,
        "calibration_config": None if calibration_config is None else str(calibration_config),
        "fast": bool(fast),
        "smoke": bool(smoke),
        "seed_count": int(seed_count),
        "n_rows": int(len(joined)),
        "n_finite_qrc_advantage": int(pd.to_numeric(joined["qrc_advantage"], errors="coerce").notna().sum()) if "qrc_advantage" in joined.columns else 0,
        "qrc_feature_dim": int(qrc_cfg.feature_dim),
        "claim_boundary": "Evaluated frontier rows support conditional protocol-local QRC usefulness claims only.",
        "outputs": [catalog_path.name, path.name, f"frontier_{split}_evaluation_manifest.json"],
    }
    (out_dir / f"frontier_{split}_evaluation_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return joined, path


def _write_evaluated_selection_checkpointed(
    selection: pd.DataFrame,
    *,
    selection_path: Path,
    out_dir: Path,
    split: str,
    comparison_protocol: str,
    calibration_config: Path | None,
    fast: bool,
    smoke: bool,
    seeds: int | None,
    checkpoint_every: int,
) -> tuple[pd.DataFrame, Path]:
    """Chunked evaluator that can resume from completed chunk catalogs."""

    specs = specs_from_selection(selection, split=split)
    if not specs:
        raise ValueError(f"no selected specs found for split={split}")
    selected = selected_rows_from_selection(selection, split=split)
    if len(selected) != len(specs):
        raise ValueError("selection/spec reconstruction mismatch")

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, int(checkpoint_every))
    chunk_targets: list[pd.DataFrame] = []
    seed_count_observed = int(seeds) if seeds is not None else None
    qrc_feature_dim: int | None = None
    total_chunks = int(math.ceil(len(specs) / chunk_size))
    for chunk_idx, start in enumerate(range(0, len(specs), chunk_size), start=1):
        stop = min(start + chunk_size, len(specs))
        chunk_specs = specs[start:stop]
        expected_ids = {
            f"{study_spec.name}:{study_spec.seed}:{study_spec.length}"
            for study_spec in (_study_spec(s, smoke=smoke, fast=fast) for s in chunk_specs)
        }
        chunk_stem = f"frontier_{split}_chunk_{chunk_idx:04d}"
        chunk_path = _existing_chunk_path(checkpoint_dir, chunk_stem)
        if chunk_path is not None and _chunk_complete(chunk_path, expected_ids):
            eval_df = load_catalog(chunk_path)
            print(f"checkpoint reuse split={split} chunk={chunk_idx}/{total_chunks} rows={len(eval_df)}", flush=True)
        else:
            eval_df, chunk_path, qrc_cfg, seed_count = build_catalog(
                chunk_specs,
                out_dir=checkpoint_dir,
                fast=fast,
                smoke=smoke,
                seeds=seeds,
                output_stem=chunk_stem,
                comparison_protocol=comparison_protocol,
                calibration_config=calibration_config,
            )
            seed_count_observed = int(seed_count)
            qrc_feature_dim = int(qrc_cfg.feature_dim)
            print(f"checkpoint wrote split={split} chunk={chunk_idx}/{total_chunks} rows={len(eval_df)} path={chunk_path}", flush=True)
        chunk_targets.append(eval_df.loc[:, ["dataset_id", *[c for c in TARGET_COLUMNS if c in eval_df.columns]]])
        _write_evaluated_checkpoint_partial(
            selected,
            chunk_targets,
            out_dir=out_dir,
            split=split,
            selection_path=selection_path,
            comparison_protocol=comparison_protocol,
            calibration_config=calibration_config,
            fast=fast,
            smoke=smoke,
            seed_count=seed_count_observed,
            qrc_feature_dim=qrc_feature_dim,
            checkpoint_every=checkpoint_every,
            total_rows=len(specs),
            total_chunks=total_chunks,
        )

    joined, path = _write_evaluated_checkpoint_partial(
        selected,
        chunk_targets,
        out_dir=out_dir,
        split=split,
        selection_path=selection_path,
        comparison_protocol=comparison_protocol,
        calibration_config=calibration_config,
        fast=fast,
        smoke=smoke,
        seed_count=seed_count_observed,
        qrc_feature_dim=qrc_feature_dim,
        checkpoint_every=checkpoint_every,
        total_rows=len(specs),
        total_chunks=total_chunks,
        final=True,
    )
    return joined, path


def _write_evaluated_checkpoint_partial(
    selected: pd.DataFrame,
    chunk_targets: list[pd.DataFrame],
    *,
    out_dir: Path,
    split: str,
    selection_path: Path,
    comparison_protocol: str,
    calibration_config: Path | None,
    fast: bool,
    smoke: bool,
    seed_count: int | None,
    qrc_feature_dim: int | None,
    checkpoint_every: int,
    total_rows: int,
    total_chunks: int,
    final: bool = False,
) -> tuple[pd.DataFrame, Path]:
    targets = pd.concat(chunk_targets, ignore_index=True) if chunk_targets else pd.DataFrame(columns=["dataset_id", *TARGET_COLUMNS])
    targets = targets.drop_duplicates(subset=["dataset_id"], keep="last")
    joined = selected.merge(targets, on="dataset_id", how="left", validate="one_to_one")
    joined = materialize_frontier_features(joined)
    stem = f"frontier_{split}_evaluated_30_features"
    path = out_dir / (f"{stem}.csv" if final else f"{stem}_partial.csv")
    joined.to_csv(path, index=False)
    finite_count = int(pd.to_numeric(joined.get("qrc_advantage", pd.Series(dtype=float)), errors="coerce").notna().sum())
    manifest = {
        "analysis_version": FRONTIER_ATLAS_VERSION,
        "artifact": "evaluated_selection",
        "selection_path": str(selection_path),
        "split": split,
        "comparison_protocol": comparison_protocol,
        "calibration_config": None if calibration_config is None else str(calibration_config),
        "fast": bool(fast),
        "smoke": bool(smoke),
        "seed_count": seed_count,
        "checkpoint_every": int(checkpoint_every),
        "total_chunks": int(total_chunks),
        "n_rows": int(len(joined)),
        "n_finite_qrc_advantage": finite_count,
        "qrc_feature_dim": qrc_feature_dim,
        "complete": bool(finite_count == total_rows),
        "claim_boundary": "Evaluated frontier rows support conditional protocol-local QRC usefulness claims only.",
        "outputs": [path.name, f"frontier_{split}_evaluation_manifest.json"],
    }
    (out_dir / f"frontier_{split}_evaluation_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return joined, path


def _existing_chunk_path(checkpoint_dir: Path, chunk_stem: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        path = checkpoint_dir / f"{chunk_stem}{suffix}"
        if path.exists():
            return path
    return None


def _chunk_complete(path: Path, expected_ids: set[str]) -> bool:
    try:
        df = load_catalog(path)
    except Exception:
        return False
    if set(df.get("dataset_id", pd.Series(dtype=str)).astype(str)) != set(expected_ids):
        return False
    if "qrc_advantage" not in df.columns:
        return False
    return bool(pd.to_numeric(df["qrc_advantage"], errors="coerce").notna().all())


def frontier_enrichment_score(df: pd.DataFrame) -> pd.Series:
    """Target-free score for rows near the currently interesting frontier.

    The score uses only measured properties. It intentionally avoids QRC/ESN
    target columns and is only a sampling heuristic, not a scientific result.
    """

    data = materialize_frontier_features(df)

    def rank(col: str, *, ascending: bool = True) -> pd.Series:
        vals = pd.to_numeric(data[col], errors="coerce") if col in data.columns else pd.Series(np.nan, index=data.index)
        if vals.notna().sum() == 0:
            return pd.Series(0.5, index=data.index)
        med = float(vals.median())
        vals = vals.fillna(med)
        return vals.rank(pct=True, ascending=ascending)

    persistence = rank("dfa_alpha")
    entropy = 0.5 * rank("perm_entropy") + 0.5 * rank("sample_entropy")
    recurrence = 0.5 * rank("ext_recurrence_determinism") + 0.5 * rank("ext_recurrence_rate")
    volatility = 0.5 * rank("ext_volatility_ac1") + 0.5 * rank("ext_arch_lm5")
    noise = rank("snr_db", ascending=False)
    nonlin_gap_low = rank("nl_gain", ascending=False)
    stationarity_boundary = 1.0 - (rank("adf_p") - 0.5).abs() * 2.0
    regime = 0.5 * rank("ext_changepoint_count") + 0.5 * rank("ext_trend_strength")
    score = (
        0.22 * persistence
        + 0.16 * entropy
        + 0.14 * recurrence
        + 0.12 * volatility
        + 0.10 * noise
        + 0.10 * nonlin_gap_low
        + 0.08 * stationarity_boundary
        + 0.08 * regime
    )
    return score.clip(0.0, 1.0)


def grouped_validation_summary(
    evaluated: pd.DataFrame,
    *,
    seed: int,
    win_threshold: float,
) -> pd.DataFrame:
    """Grouped-CV metrics by base generator and family for the 30-feature suite."""

    rows: list[dict[str, Any]] = []
    for group_col in ("base_generator", "family"):
        if group_col not in evaluated.columns or evaluated[group_col].nunique(dropna=True) < 2:
            continue
        rows.append(_grouped_cv(evaluated, group_col=group_col, seed=seed, win_threshold=win_threshold))
    return pd.DataFrame(rows)


def rule_table_from_result(result, *, raw_features: pd.DataFrame | None = None, win_threshold: float, seed: int) -> pd.DataFrame:
    """Fit a shallow tree and return readable high-usefulness leaf rules."""

    if raw_features is not None and result.features_used:
        X_df = raw_features[result.features_used].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)
        X = X_df.to_numpy(dtype=float)
        feature_names = list(X_df.columns)
    else:
        X = np.asarray(result.X, dtype=float)
        feature_names = list(result.features_used)
    y = np.asarray(result.y, dtype=float)
    if X.size == 0 or y.size < 20 or len(np.unique(y >= win_threshold)) < 2:
        return pd.DataFrame(columns=["rule", "n", "qrc_useful_rate", "mean_advantage"])
    labels = y >= float(win_threshold)
    min_leaf = max(10, int(round(0.02 * len(y))))
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=min_leaf, random_state=seed)
    tree.fit(X, labels)
    leaf_id = tree.apply(X)
    rules = _tree_rules(tree, feature_names)
    rows: list[dict[str, Any]] = []
    for leaf, rule in rules.items():
        idx = leaf_id == leaf
        if not np.any(idx):
            continue
        rows.append(
            {
                "rule": rule,
                "n": int(idx.sum()),
                "qrc_useful_rate": float(labels[idx].mean()),
                "qrc_win_rate": float((y[idx] > 0.0).mean()),
                "mean_advantage": float(np.mean(y[idx])),
            }
        )
    return pd.DataFrame(rows).sort_values(["qrc_useful_rate", "mean_advantage", "n"], ascending=[False, False, False]).reset_index(drop=True)


def infer_base_generator(row: dict[str, Any] | pd.Series) -> str:
    params = row.get("params", {}) if isinstance(row, dict) else row.get("params", {})
    parsed = _parse_params(params)
    generator = parsed.get("generator")
    if generator:
        suffix = "_noise" if parsed.get("noise_overlay") else ""
        return f"{generator}{suffix}"
    name = str(row.get("name", "unknown"))
    if "_snr" in name:
        return f"{name.split('_snr', 1)[0]}_noise"
    return name.rsplit("_s", 1)[0]


def _parse_params(params: Any) -> dict[str, Any]:
    if isinstance(params, dict):
        return params
    if isinstance(params, str) and params.strip():
        try:
            value = ast.literal_eval(params)
            if isinstance(value, dict):
                return value
        except Exception:
            return {}
    return {}


def _split_candidate_pool(df: pd.DataFrame, *, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    out = pd.Series(index=df.index, dtype=object)
    group_cols = [c for c in ("family", "base_generator") if c in df.columns]
    if not group_cols:
        perm = rng.permutation(df.index.to_numpy())
        half = len(perm) // 2
        out.loc[perm[:half]] = "discovery"
        out.loc[perm[half:]] = "validation"
        return out
    for _, group in df.groupby(group_cols, dropna=False, sort=True):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        half = len(idx) // 2
        out.loc[idx[:half]] = "discovery"
        out.loc[idx[half:]] = "validation"
    return out.fillna("validation")


def _select_subset(pool: pd.DataFrame, *, n_rows: int, seed: int, split_name: str, selection_protocol: str = "v3") -> pd.DataFrame:
    if n_rows <= 0:
        return pd.DataFrame(columns=list(pool.columns) + ["__index__", "evaluation_split", "selection_role", "selection_order"])
    if len(pool) <= n_rows:
        selected = pool.copy()
        selected["__index__"] = selected.index
        selected["evaluation_split"] = split_name
        selected["selection_role"] = "all_available"
        selected["selection_order"] = np.arange(len(selected))
        return selected

    selected_parts: list[pd.DataFrame] = []
    used: set[int] = set()

    def add(frame: pd.DataFrame, role: str, limit: int) -> None:
        nonlocal used
        if limit <= 0:
            return
        cand = frame[~frame.index.isin(used)].head(limit).copy()
        if cand.empty:
            return
        cand["selection_role"] = role
        selected_parts.append(cand)
        used.update(int(i) for i in cand.index)

    if selection_protocol == "v4":
        broad_n = int(round(n_rows * 0.35))
        coverage_n = int(round(n_rows * 0.25))
        frontier_n = int(round(n_rows * 0.20))
        control_n = int(round(n_rows * 0.10))
        perturb_n = max(0, n_rows - broad_n - coverage_n - frontier_n - control_n)
        add(_stratified_sample(pool, broad_n, seed=seed), "broad_balanced", broad_n)
        add(_coverage_order(pool[~pool.index.isin(used)], seed=seed), "feature_space_coverage", coverage_n)
        add(pool.sort_values("frontier_score", ascending=False), "frontier_enriched", frontier_n)
        add(pool.sort_values("frontier_score", ascending=True), "v3_informed_stress_control", control_n)
        add(_perturbation_axis_rows(pool[~pool.index.isin(used)], seed=seed), "perturbation_axis", perturb_n)
    else:
        broad_n = int(round(n_rows * 0.50))
        frontier_n = int(round(n_rows * 0.25))
        control_n = int(round(n_rows * 0.10))
        coverage_n = max(0, n_rows - broad_n - frontier_n - control_n)
        add(_stratified_sample(pool, broad_n, seed=seed), "broad_balanced", broad_n)
        add(pool.sort_values("frontier_score", ascending=False), "frontier_enriched", frontier_n)
        add(pool.sort_values("frontier_score", ascending=True), "negative_boundary_control", control_n)
        add(_coverage_order(pool[~pool.index.isin(used)], seed=seed), "feature_space_coverage", coverage_n)
    if sum(len(part) for part in selected_parts) < n_rows:
        rest = pool[~pool.index.isin(used)].sample(frac=1.0, random_state=seed)
        add(rest, "fill", n_rows - sum(len(part) for part in selected_parts))

    selected = pd.concat(selected_parts, axis=0)
    selected = selected.head(n_rows).copy()
    selected["__index__"] = selected.index.astype(int)
    selected["evaluation_split"] = split_name
    selected["selection_order"] = np.arange(len(selected))
    return selected


def _perturbation_axis_rows(df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    if df.empty or "params" not in df.columns:
        return df.iloc[0:0]
    mask = df["params"].astype(str).str.contains("perturbation_axes", regex=False)
    perturbed = df[mask]
    if perturbed.empty:
        return perturbed
    return _stratified_sample(perturbed, len(perturbed), seed=seed)


def _stratified_sample(df: pd.DataFrame, n_rows: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if n_rows <= 0 or df.empty:
        return df.iloc[0:0]
    group_cols = [c for c in ("family", "base_generator") if c in df.columns]
    if not group_cols:
        return df.sample(n=min(n_rows, len(df)), random_state=seed)
    groups = list(df.groupby(group_cols, dropna=False, sort=True))
    if not groups:
        return df.sample(n=min(n_rows, len(df)), random_state=seed)
    selected: list[int] = []
    shuffled = []
    for _, group in groups:
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        shuffled.append(list(idx))
    pos = 0
    while len(selected) < min(n_rows, len(df)) and any(shuffled):
        bucket = shuffled[pos % len(shuffled)]
        if bucket:
            selected.append(int(bucket.pop()))
        pos += 1
        shuffled = [b for b in shuffled if b]
    return df.loc[selected]


def _coverage_order(df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    coords = _feature_coordinates(df)
    if coords.shape[0] <= 2:
        return df
    center = np.nanmedian(coords, axis=0)
    distance = np.sqrt(np.sum((coords - center) ** 2, axis=1))
    angle = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    order = np.lexsort((-distance, angle))
    # Deterministically interleave angular coverage and extremes.
    return df.iloc[order[::-1]]


def _feature_coordinates(df: pd.DataFrame) -> np.ndarray:
    features = [c for c in FRONTIER_TIER_A_FIELDS if c in df.columns]
    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    if X.shape[1] == 0:
        return np.zeros((len(df), 2), dtype=float)
    Xs = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    if Xs.shape[1] == 1:
        return np.column_stack([Xs[:, 0], np.zeros(Xs.shape[0])])
    return PCA(n_components=2, random_state=0).fit_transform(Xs)


def _noncollinear_feature_indices(X: np.ndarray, *, threshold: float = 0.985) -> list[int]:
    if X.shape[1] <= 1:
        return list(range(X.shape[1]))
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    keep: list[int] = []
    for j in range(X.shape[1]):
        if not keep:
            keep.append(j)
            continue
        if all(abs(float(corr[j, k])) < threshold for k in keep):
            keep.append(j)
    return keep or [0]


def _self_knn_mean_distances(X: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(max(1, int(k)), max(1, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    return distances[:, 1:].mean(axis=1)


def _target_knn_mean_distances(X_ref: np.ndarray, X_target: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(max(1, int(k)), X_ref.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X_ref)
    distances, _ = nn.kneighbors(X_target)
    return distances.mean(axis=1)


def _distance_percentiles(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    if ref.size == 0:
        return np.ones_like(values, dtype=float)
    vals = np.asarray(values, dtype=float)
    pct = np.searchsorted(ref, vals, side="right") / ref.size
    return np.clip(pct, 0.0, 1.0)


def _nearest_family_mix(discovery: pd.DataFrame, X_ref: np.ndarray, X_target: np.ndarray, *, k: int) -> tuple[list[str], np.ndarray]:
    families = discovery.get("family", pd.Series(["unknown"] * len(discovery))).astype(str).to_numpy()
    k_eff = min(max(1, int(k)), X_ref.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X_ref)
    _, indices = nn.kneighbors(X_target)
    mixtures: list[str] = []
    entropy: list[float] = []
    for row in indices:
        vals, counts = np.unique(families[row], return_counts=True)
        probs = counts.astype(float) / max(1, counts.sum())
        order = np.argsort(-probs)
        mixtures.append(";".join(f"{vals[i]}:{probs[i]:.3f}" for i in order[:5]))
        entropy.append(float(-np.sum(probs * np.log(probs + 1e-12))))
    return mixtures, np.asarray(entropy, dtype=float)


def _frontier_meta_summary(evaluated: pd.DataFrame, result, *, win_threshold: float) -> pd.DataFrame:
    gb_reg = result.regression_cv.get("models", {}).get("gradient_boosting", {})
    gb_clf = result.classification_cv.get("models", {}).get("gradient_boosting", {})
    y = pd.to_numeric(evaluated["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    useful = y >= float(win_threshold)
    return pd.DataFrame(
        [
            {
                "model_suite": "frontier_30_tier_a",
                "n_rows": int(len(evaluated)),
                "n_features_declared": int(len(FRONTIER_TIER_A_FIELDS)),
                "n_features_used": int(len(result.features_used)),
                "qrc_win_rate": float((y > 0.0).mean()),
                "qrc_useful_rate": float(useful.mean()),
                "mean_advantage": float(np.mean(y)),
                "median_advantage": float(np.median(y)),
                "regression_r2_mean": _float_or_nan(gb_reg.get("r2_mean")),
                "regression_mae_mean": _float_or_nan(gb_reg.get("mae_mean")),
                "classification_roc_auc_mean": _float_or_nan(gb_clf.get("roc_auc_mean")),
                "classification_pr_auc_mean": _float_or_nan(gb_clf.get("average_precision_mean")),
                "classification_brier_mean": _float_or_nan(gb_clf.get("brier_mean")),
                "top_features": ",".join(result.ranked_importances["feature"].head(8).astype(str).tolist()) if not result.ranked_importances.empty else "",
                "notes": "; ".join(result.notes),
            }
        ]
    )


def _grouped_cv(evaluated: pd.DataFrame, *, group_col: str, seed: int, win_threshold: float) -> dict[str, Any]:
    features = [c for c in FRONTIER_TIER_A_FIELDS if c in evaluated.columns]
    X_df = evaluated[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)
    X = StandardScaler().fit_transform(X_df.to_numpy(dtype=float))
    y = pd.to_numeric(evaluated["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    y_bin = y >= float(win_threshold)
    groups = evaluated[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2 or len(y) < 20:
        return {"group_col": group_col, "n_groups": int(len(unique_groups)), "note": "insufficient groups"}
    n_splits = min(5, len(unique_groups))
    reg_r2: list[float] = []
    reg_mae: list[float] = []
    auc: list[float] = []
    ap: list[float] = []
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        reg = GradientBoostingRegressor(random_state=seed).fit(X[train_idx], y[train_idx])
        pred = reg.predict(X[test_idx])
        reg_r2.append(float(r2_score(y[test_idx], pred)) if np.var(y[test_idx]) > 1e-12 else np.nan)
        reg_mae.append(float(mean_absolute_error(y[test_idx], pred)))
        if len(np.unique(y_bin[train_idx])) < 2 or len(np.unique(y_bin[test_idx])) < 2:
            auc.append(np.nan)
            ap.append(np.nan)
            continue
        clf = GradientBoostingClassifier(random_state=seed).fit(X[train_idx], y_bin[train_idx])
        prob = clf.predict_proba(X[test_idx])[:, 1]
        auc.append(float(roc_auc_score(y_bin[test_idx], prob)))
        ap.append(float(average_precision_score(y_bin[test_idx], prob)))
    return {
        "group_col": group_col,
        "n_groups": int(len(unique_groups)),
        "n_splits": int(n_splits),
        "regression_r2_mean": _finite_mean(reg_r2),
        "regression_mae_mean": _finite_mean(reg_mae),
        "classification_roc_auc_mean": _finite_mean(auc),
        "classification_pr_auc_mean": _finite_mean(ap),
    }


def _tree_rules(tree: DecisionTreeClassifier, feature_names: list[str]) -> dict[int, str]:
    rules: dict[int, str] = {}

    def walk(node: int, conditions: list[str]) -> None:
        if tree.tree_.feature[node] == _tree.TREE_UNDEFINED:
            rules[node] = " and ".join(conditions) if conditions else "all rows"
            return
        feature = feature_names[tree.tree_.feature[node]]
        threshold = float(tree.tree_.threshold[node])
        walk(tree.tree_.children_left[node], conditions + [f"{feature} <= {threshold:.3f}"])
        walk(tree.tree_.children_right[node], conditions + [f"{feature} > {threshold:.3f}"])

    walk(0, [])
    return rules


def _write_frontier_importance_plot(importances: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    top = importances.head(12).iloc[::-1] if not importances.empty else pd.DataFrame({"feature": [], "importance_mean": []})
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if not top.empty:
        ax.barh(top["feature"], top["importance_mean"], color="#2f6f6f")
    ax.set_xlabel("Permutation importance")
    ax.set_title("Frontier 30-feature meta-model importances")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_regime_projection(evaluated: pd.DataFrame, result, path: Path, *, win_threshold: float) -> None:
    import matplotlib.pyplot as plt

    coords = _feature_coordinates(evaluated)
    y = pd.to_numeric(evaluated["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="coolwarm", s=18, alpha=0.82, linewidths=0.0)
    useful = y >= float(win_threshold)
    if np.any(useful):
        ax.scatter(coords[useful, 0], coords[useful, 1], facecolors="none", edgecolors="#111111", s=40, linewidths=0.7, label="QRC useful")
        ax.legend(loc="best", frameon=True, fontsize=8)
    fig.colorbar(sc, ax=ax, label="QRC advantage")
    ax.set_xlabel("30-feature map PC1")
    ax.set_ylabel("30-feature map PC2")
    ax.set_title("Conditional QRC advantage regime map")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _count_dict(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    counts = series.dropna().astype(str).value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def _finite_mean(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def _float_or_nan(value: Any) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else math.nan
    except Exception:
        return math.nan


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj
