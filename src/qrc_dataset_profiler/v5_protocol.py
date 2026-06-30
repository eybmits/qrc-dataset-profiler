"""v5 multi-QRC protocol for the QRC-vs-ESN regime atlas.

This layer is intentionally separate from the v4 artifacts.  It calibrates three
canonical QRC variants once on held-out synthetic data, freezes a feature-matched
canonical sparse ESN once, and then evaluates all selected atlas rows without
per-dataset reservoir tuning.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from qrc_dataset_profiler.analysis import load_catalog
from qrc_dataset_profiler.baselines import esn_sparse_baseline, gbm_baseline, linear_baseline, qrc_scores_standard
from qrc_dataset_profiler.calibration import (
    _best_index,
    _best_qrc_index,
    _esn_candidates,
    _esn_row,
    _json_safe,
    _key_tuple,
    _load_existing_rows,
    _materialize_calibration_datasets,
    _qrc_key,
    _qrc_key_columns,
    _qrc_row,
    _score_esn_candidate,
    _score_qrc_candidate,
    select_calibration_specs,
)
from qrc_dataset_profiler.frontier import materialize_frontier_features, selected_rows_from_selection, specs_from_selection
from qrc_dataset_profiler.generators import generate, make_sweep_specs_v4
from qrc_dataset_profiler.paper_robustness import nvar_baseline_scores
from qrc_dataset_profiler.properties import profile_dataset
from qrc_dataset_profiler.reservoir import StandardSpinV1
from qrc_dataset_profiler.run_study import _study_horizon, _study_spec
from qrc_dataset_profiler.spec import Dataset, DatasetSpec


V5_PROTOCOL_VERSION = "v5-multi-qrc-atlas-v1"
V5_VARIANTS: tuple[str, ...] = ("qrc_m", "qrc_e", "qrc_d")
V5_VARIANT_DESCRIPTIONS: dict[str, str] = {
    "qrc_m": "mechanistic disordered spin-QRC: input injection/reset only, no RZ reuploading, no dissipation",
    "qrc_e": "encoding-enhanced disordered spin-QRC: input injection plus fixed RZ reuploading",
    "qrc_d": "dissipative disordered spin-QRC: input injection plus fixed mild local dissipation",
}
_CALIBRATION_SCORE_COLUMNS = ("mean_val_nrmse", "median_val_nrmse", "mean_test_nrmse", "median_test_nrmse")
DEFAULT_V5_ESN_GRID = {
    "rho": (0.7, 0.9, 1.1),
    "leak": (0.3, 0.6, 1.0),
    "input_scale": (0.3, 1.0),
}


def run_v5_calibration(
    *,
    out_dir: Path,
    sweep_seed: int = 1501,
    sweep_n_per_template: int = 20,
    calibration_rows_per_family: int = 20,
    fast: bool = True,
    seeds: int = 1,
    small_grid: bool = False,
    selection_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Calibrate QRC-M/QRC-E/QRC-D and the canonical ESN once globally."""

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_values = tuple(range(max(1, int(seeds))))
    sweep_specs = make_sweep_specs_v4(n_per_template=sweep_n_per_template, seed=sweep_seed)
    specs = select_calibration_specs(sweep_specs, rows_per_family=calibration_rows_per_family, seed=sweep_seed)
    datasets, spec_rows = _materialize_calibration_datasets(specs, fast=fast)
    if not datasets:
        raise ValueError("no calibration datasets were generated")
    pd.DataFrame(spec_rows).to_csv(out_dir / "calibration_catalog.csv", index=False)

    variant_manifests: dict[str, Any] = {}
    selected_cfgs: dict[str, StandardSpinV1] = {}
    qrc_score_summary: dict[str, Any] = {}
    for variant in V5_VARIANTS:
        candidates = list(_v5_qrc_candidates(variant, small_grid=small_grid))
        rows: list[dict[str, Any]] = []
        scores: list[dict[str, float]] = []
        score_path = out_dir / f"{variant}_calibration_scores.csv"
        existing = _load_existing_rows(score_path, key_cols=("variant", *_qrc_key_columns()))
        for idx, cfg in enumerate(candidates, start=1):
            key = {"variant": variant, **_qrc_key(cfg)}
            cached = existing.get(_key_tuple(key))
            cached_score = _cached_calibration_score(cached)
            if cached_score is not None:
                score = cached_score
            else:
                score = _score_qrc_candidate(datasets, cfg, seed_values)
                existing[_key_tuple(key)] = {"variant": variant, **_qrc_row(cfg, score)}
            rows.append({"variant": variant, **_qrc_row(cfg, score)})
            scores.append(score)
            pd.DataFrame(rows).to_csv(score_path, index=False)
            print(
                f"v5 qrc calibration {'reuse' if cached_score is not None else 'scored'} "
                f"variant={variant} candidate={idx}/{len(candidates)} "
                f"dim={cfg.feature_dim} val={score['mean_val_nrmse']:.6g}",
                flush=True,
            )
        qrc_table = pd.DataFrame(rows)
        qrc_table.to_csv(score_path, index=False)
        best_idx = _best_qrc_index(candidates, scores, tolerance=float(selection_tolerance))
        selected = candidates[best_idx]
        selected_cfgs[variant] = selected
        variant_manifests[variant] = _v5_qrc_manifest(variant, selected)
        qrc_score_summary[variant] = {
            "best_mean_val_nrmse": float(scores[best_idx]["mean_val_nrmse"]),
            "best_mean_test_nrmse_on_calibration": float(scores[best_idx]["mean_test_nrmse"]),
            "n_candidates": int(len(candidates)),
        }

    feature_dims = sorted({int(cfg.feature_dim) for cfg in selected_cfgs.values()})
    if len(feature_dims) != 1:
        raise ValueError(f"v5 currently expects all QRC variants to share one feature dimension, got {feature_dims}")
    esn_candidates = list(_esn_candidates(DEFAULT_V5_ESN_GRID if not small_grid else {"rho": (0.9,), "leak": (0.6,), "input_scale": (1.0,)}))
    esn_scores: list[dict[str, float]] = []
    esn_rows: list[dict[str, Any]] = []
    reference_qrc = selected_cfgs["qrc_m"]
    esn_score_path = out_dir / "esn_calibration_scores.csv"
    existing_esn = _load_existing_rows(esn_score_path, key_cols=("feature_dim", "rho", "leak", "input_scale"))
    for idx, cand in enumerate(esn_candidates, start=1):
        key = {"feature_dim": int(reference_qrc.feature_dim), **cand}
        cached = existing_esn.get(_key_tuple(key))
        cached_score = _cached_calibration_score(cached)
        if cached_score is not None:
            score = cached_score
        else:
            score = _score_esn_candidate(datasets, reference_qrc, cand, seed_values)
            existing_esn[_key_tuple(key)] = {"feature_dim": int(reference_qrc.feature_dim), **_esn_row(cand, score)}
        esn_scores.append(score)
        esn_rows.append({"feature_dim": int(reference_qrc.feature_dim), **_esn_row(cand, score)})
        pd.DataFrame(esn_rows).to_csv(esn_score_path, index=False)
        print(
            f"v5 esn calibration {'reuse' if cached_score is not None else 'scored'} "
            f"candidate={idx}/{len(esn_candidates)} val={score['mean_val_nrmse']:.6g}",
            flush=True,
        )
    best_esn_idx = _best_index(esn_scores)
    best_esn = esn_candidates[best_esn_idx]
    pd.DataFrame(esn_rows).to_csv(esn_score_path, index=False)

    manifest = {
        "analysis_version": V5_PROTOCOL_VERSION,
        "comparison_protocol": "standard_v5_multi_qrc",
        "protocol_layer": "globally_frozen_multi_qrc_vs_canonical_esn",
        "calibration_data": {
            "source": "synthetic held-out v4 taxonomy sweep variants",
            "sweep_seed": int(sweep_seed),
            "sweep_n_per_template": int(sweep_n_per_template),
            "rows_per_family": int(calibration_rows_per_family),
            "n_rows": int(len(datasets)),
            "families": sorted({ds.spec.family for ds in datasets}),
            "fast": bool(fast),
        },
        "qrc_variants": variant_manifests,
        "esn": {
            "class": "frozen_sparse_random_leaky_esn",
            "reservoir_size": int(reference_qrc.feature_dim),
            "density": 0.1,
            "bias_scale": 0.2,
            "rho": float(best_esn["rho"]),
            "leak": float(best_esn["leak"]),
            "input_scale": float(best_esn["input_scale"]),
            "hyperparameter_selection": "selected_once_on_same_held_out_calibration_set_then_frozen",
        },
        "scores": {
            "qrc": qrc_score_summary,
            "esn": {
                "best_mean_val_nrmse": float(esn_scores[best_esn_idx]["mean_val_nrmse"]),
                "best_mean_test_nrmse_on_calibration": float(esn_scores[best_esn_idx]["mean_test_nrmse"]),
                "n_candidates": int(len(esn_candidates)),
            },
        },
        "selection_metric": "mean validation NRMSE across held-out calibration datasets and reservoir seeds",
        "selection_tolerance": float(selection_tolerance),
        "no_per_dataset_tuning": True,
        "claim_boundary": (
            "v5 supports a regime-atlas comparison of globally frozen canonical QRC mechanisms against a "
            "globally frozen canonical ESN. It does not establish broad quantum advantage."
        ),
        "outputs": [
            "calibration_catalog.csv",
            "qrc_m_calibration_scores.csv",
            "qrc_e_calibration_scores.csv",
            "qrc_d_calibration_scores.csv",
            "esn_calibration_scores.csv",
            "frozen_v5_config.json",
        ],
    }
    (out_dir / "frozen_v5_config.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def write_v5_evaluated_selection(
    *,
    selection_path: Path,
    calibration_config: Path,
    out_dir: Path,
    split: str = "discovery",
    fast: bool = True,
    smoke: bool = False,
    seeds: int = 1,
    include_nvar: bool = True,
    checkpoint_every: int = 0,
) -> tuple[pd.DataFrame, Path]:
    """Evaluate selected synthetic atlas rows with frozen v5 QRC variants and ESN."""

    out_dir.mkdir(parents=True, exist_ok=True)
    selection = load_catalog(selection_path)
    selected = selected_rows_from_selection(selection, split=split)
    specs = specs_from_selection(selection, split=split)
    if not specs:
        raise ValueError(f"no selected specs found for split={split}")
    if len(selected) != len(specs):
        raise ValueError("selection/spec reconstruction mismatch")
    cfg = load_v5_config(calibration_config)
    qrc_cfgs = _qrc_cfgs_from_v5_config(cfg)
    esn_grid = _esn_grid_from_v5_config(cfg)
    seed_values = tuple(range(max(1, int(seeds))))

    rows: list[pd.DataFrame] = []
    checkpoint_dir = out_dir / "checkpoints_v5"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, int(checkpoint_every)) if checkpoint_every else len(specs)
    total_chunks = int(math.ceil(len(specs) / chunk_size))
    for chunk_idx, start in enumerate(range(0, len(specs), chunk_size), start=1):
        stop = min(start + chunk_size, len(specs))
        chunk_stem = f"v5_{split}_chunk_{chunk_idx:04d}"
        chunk_path = checkpoint_dir / f"{chunk_stem}.csv"
        if chunk_path.exists() and _v5_chunk_complete(chunk_path, stop - start):
            chunk = load_catalog(chunk_path)
            print(f"v5 checkpoint reuse split={split} chunk={chunk_idx}/{total_chunks} rows={len(chunk)}", flush=True)
        else:
            chunk = _evaluate_v5_specs(
                specs[start:stop],
                qrc_cfgs=qrc_cfgs,
                esn_grid=esn_grid,
                fast=fast,
                smoke=smoke,
                seed_values=seed_values,
                include_nvar=include_nvar,
            )
            chunk.to_csv(chunk_path, index=False)
            print(f"v5 checkpoint wrote split={split} chunk={chunk_idx}/{total_chunks} rows={len(chunk)} path={chunk_path}", flush=True)
        rows.append(chunk)
        _write_v5_partial(selected, rows, out_dir=out_dir, split=split, calibration_config=calibration_config, cfg=cfg, complete=False)
    return _write_v5_partial(selected, rows, out_dir=out_dir, split=split, calibration_config=calibration_config, cfg=cfg, complete=True)


def load_v5_config(path: Path | str) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg.get("comparison_protocol") != "standard_v5_multi_qrc":
        raise ValueError("v5 calibration config must have comparison_protocol='standard_v5_multi_qrc'")
    return cfg


def _cached_calibration_score(row: dict[str, Any] | None) -> dict[str, float] | None:
    if row is None:
        return None
    try:
        return {key: float(row[key]) for key in _CALIBRATION_SCORE_COLUMNS}
    except (KeyError, TypeError, ValueError):
        return None


def _v5_qrc_candidates(variant: str, *, small_grid: bool) -> Iterable[StandardSpinV1]:
    if variant not in V5_VARIANTS:
        raise ValueError(f"unknown v5 variant: {variant}")
    common = {
        "n_qubits": 6,
        "h": 1.0,
        "depth": 5,
        "topology": "complete",
        "virtual_nodes": 3,
        "coupling_mode": "disordered",
        "coupling_seed": 314159,
        "dissipation_method": "trajectory",
    }
    J_values = (1.0,) if small_grid else (0.5, 1.0, 1.5)
    dt_values = (0.25,) if small_grid else (0.15, 0.25, 0.35)
    if variant in {"qrc_m", "qrc_e"}:
        damping_pairs = ((0.0, 0.0),)
    else:
        damping_pairs = ((0.02, 0.01),) if small_grid else ((0.005, 0.005), (0.02, 0.01), (0.05, 0.02))
    for J in J_values:
        for dt in dt_values:
            for amp, deph in damping_pairs:
                yield StandardSpinV1(
                    **common,
                    J=float(J),
                    dt=float(dt),
                    reupload=bool(variant == "qrc_e"),
                    amplitude_damping=float(amp),
                    dephasing=float(deph),
                )


def _v5_qrc_manifest(variant: str, cfg: StandardSpinV1) -> dict[str, Any]:
    return {
        "variant": variant,
        "description": V5_VARIANT_DESCRIPTIONS[variant],
        "class": "StandardSpinV1",
        "n_qubits": int(cfg.n_qubits),
        "J": float(cfg.J),
        "h": float(cfg.h),
        "dt": float(cfg.dt),
        "depth": int(cfg.depth),
        "topology": cfg.topology,
        "virtual_nodes": int(cfg.virtual_nodes),
        "reupload": bool(cfg.reupload),
        "amplitude_damping": float(cfg.amplitude_damping),
        "dephasing": float(cfg.dephasing),
        "dissipation_method": cfg.dissipation_method,
        "coupling_mode": cfg.coupling_mode,
        "coupling_seed": int(cfg.coupling_seed),
        "feature_dim": int(cfg.feature_dim),
        "hyperparameter_selection": "selected_once_on_same_held_out_calibration_set_then_frozen",
    }


def _qrc_cfgs_from_v5_config(cfg: dict[str, Any]) -> dict[str, StandardSpinV1]:
    out: dict[str, StandardSpinV1] = {}
    allowed = {
        "n_qubits",
        "J",
        "h",
        "dt",
        "depth",
        "topology",
        "virtual_nodes",
        "reupload",
        "amplitude_damping",
        "dephasing",
        "dissipation_method",
        "coupling_mode",
        "coupling_seed",
        "shots",
        "seed",
    }
    for variant, data in cfg.get("qrc_variants", {}).items():
        kwargs = {key: data[key] for key in allowed if key in data}
        out[str(variant)] = StandardSpinV1(**kwargs)
    missing = set(V5_VARIANTS) - set(out)
    if missing:
        raise ValueError(f"v5 config missing variants: {sorted(missing)}")
    return out


def _esn_grid_from_v5_config(cfg: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    esn = dict(cfg.get("esn", {}))
    return {
        "rho": (float(esn["rho"]),),
        "leak": (float(esn["leak"]),),
        "input_scale": (float(esn["input_scale"]),),
    }


def _evaluate_v5_specs(
    specs: list[DatasetSpec],
    *,
    qrc_cfgs: dict[str, StandardSpinV1],
    esn_grid: dict[str, tuple[float, ...]],
    fast: bool,
    smoke: bool,
    seed_values: tuple[int, ...],
    include_nvar: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference_qrc = qrc_cfgs["qrc_m"]
    for base_spec in specs:
        spec = _study_spec(base_spec, smoke=smoke, fast=fast)
        ds = generate(spec)
        if ds.ground_truth.get("_unavailable"):
            continue
        rec = profile_dataset(ds)
        horizon = _study_horizon(spec, rec.ac_timescale, override=None)
        if horizon != spec.horizon:
            spec = replace(spec, horizon=horizon)
            ds = Dataset(spec, ds.series, inputs=ds.inputs, ground_truth=ds.ground_truth)
        row: dict[str, Any] = {
            "dataset_id": f"{spec.name}:{spec.seed}:{spec.length}",
            "nrmse_linear": float(linear_baseline(ds)),
            "nrmse_gbm": float(gbm_baseline(ds, seed=spec.seed)),
        }
        if include_nvar:
            nvar = nvar_baseline_scores(ds, lag=20, degree=2)
            row["nrmse_nvar"] = float(nvar["test_nrmse"])
            row["nmae_nvar"] = float(nvar["test_nmae"])
        esn_scores = [esn_sparse_baseline(ds, qrc_cfg=reference_qrc, seed=s, esn_grid=esn_grid, return_details=True) for s in seed_values]
        row["nrmse_esn_v5"] = _mean_key(esn_scores, "nrmse")
        row["nmae_esn_v5"] = _mean_key(esn_scores, "nmae")
        for variant in V5_VARIANTS:
            qrc_scores = [qrc_scores_standard(ds, qrc_cfgs[variant], seed=s) for s in seed_values]
            nrmse = _mean_key(qrc_scores, "test_nrmse")
            nmae = _mean_key(qrc_scores, "test_nmae")
            row[f"nrmse_{variant}"] = nrmse
            row[f"nmae_{variant}"] = nmae
            row[f"advantage_{variant}_vs_esn"] = float(row["nrmse_esn_v5"] - nrmse)
            row[f"nmae_advantage_{variant}_vs_esn"] = float(row["nmae_esn_v5"] - nmae)
            row[f"{variant}_useful"] = bool(row[f"advantage_{variant}_vs_esn"] >= 0.05)
        adv_cols = [f"advantage_{variant}_vs_esn" for variant in V5_VARIANTS]
        best_col = max(adv_cols, key=lambda col: float(row[col]))
        best_variant = best_col.removeprefix("advantage_").removesuffix("_vs_esn")
        row["best_qrc_variant"] = best_variant
        row["best_qrc_advantage_vs_esn"] = float(row[best_col])
        row["qrc_any_win"] = bool(row["best_qrc_advantage_vs_esn"] > 0.0)
        row["qrc_any_useful"] = bool(row["best_qrc_advantage_vs_esn"] >= 0.05)
        row["label_seed_count"] = int(len(seed_values))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_v5_partial(
    selected: pd.DataFrame,
    chunks: list[pd.DataFrame],
    *,
    out_dir: Path,
    split: str,
    calibration_config: Path,
    cfg: dict[str, Any],
    complete: bool,
) -> tuple[pd.DataFrame, Path]:
    targets = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    targets = targets.drop_duplicates(subset=["dataset_id"], keep="last") if "dataset_id" in targets.columns else targets
    joined = selected.merge(targets, on="dataset_id", how="left", validate="one_to_one")
    joined = materialize_frontier_features(joined)
    stem = f"frontier_{split}_evaluated_v5_multi_qrc"
    path = out_dir / (f"{stem}.csv" if complete else f"{stem}_partial.csv")
    joined.to_csv(path, index=False)
    finite = int(pd.to_numeric(joined.get("best_qrc_advantage_vs_esn", pd.Series(dtype=float)), errors="coerce").notna().sum())
    manifest = {
        "analysis_version": V5_PROTOCOL_VERSION,
        "artifact": "evaluated_v5_multi_qrc_selection",
        "split": split,
        "calibration_config": str(calibration_config),
        "complete": bool(complete and finite == len(joined)),
        "n_rows": int(len(joined)),
        "n_labeled": finite,
        "feature_dim": int(cfg["esn"]["reservoir_size"]),
        "qrc_variants": list(V5_VARIANTS),
        "target_columns": [
            "advantage_qrc_m_vs_esn",
            "advantage_qrc_e_vs_esn",
            "advantage_qrc_d_vs_esn",
            "best_qrc_variant",
            "best_qrc_advantage_vs_esn",
            "qrc_any_useful",
        ],
        "claim_boundary": "v5 evaluated rows support a broad QRC-vs-ESN regime atlas, not a broad quantum-advantage claim.",
        "outputs": [path.name, f"frontier_{split}_v5_manifest.json"],
    }
    (out_dir / f"frontier_{split}_v5_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return joined, path


def _v5_chunk_complete(path: Path, expected_rows: int) -> bool:
    try:
        df = load_catalog(path)
    except Exception:
        return False
    return bool(len(df) == int(expected_rows) and pd.to_numeric(df.get("best_qrc_advantage_vs_esn"), errors="coerce").notna().all())


def summarize_v5_evaluation(evaluated: pd.DataFrame) -> pd.DataFrame:
    """Compact summary for v5 evaluated rows."""

    rows = [_v5_summary_row("overall", evaluated)]
    for family, group in evaluated.groupby("family", sort=True):
        rows.append(_v5_summary_row(f"family:{family}", group))
    return pd.DataFrame(rows)


def _v5_summary_row(name: str, df: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {"group": name, "n": int(len(df))}
    for variant in V5_VARIANTS:
        adv = pd.to_numeric(df.get(f"advantage_{variant}_vs_esn"), errors="coerce")
        row[f"mean_advantage_{variant}"] = _safe_mean(adv)
        row[f"win_rate_{variant}"] = _safe_mean(adv > 0.0)
        row[f"useful_rate_{variant}"] = _safe_mean(adv >= 0.05)
    best = pd.to_numeric(df.get("best_qrc_advantage_vs_esn"), errors="coerce")
    row["mean_best_qrc_advantage"] = _safe_mean(best)
    row["best_qrc_win_rate"] = _safe_mean(best > 0.0)
    row["best_qrc_useful_rate"] = _safe_mean(best >= 0.05)
    if "best_qrc_variant" in df.columns and len(df):
        row["most_common_best_variant"] = str(df["best_qrc_variant"].mode(dropna=True).iloc[0]) if df["best_qrc_variant"].notna().any() else ""
    return row


def _mean_key(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _safe_mean(values: Any) -> float:
    series = pd.Series(values)
    if series.dtype == bool:
        series = series.astype(float)
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan
