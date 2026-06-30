"""Held-out global reservoir calibration for the fair fixed-vs-fixed protocol."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from qrc_dataset_profiler.baselines import esn_sparse_baseline, qrc_scores_standard
from qrc_dataset_profiler.generators import generate, make_sweep_specs, make_sweep_specs_v4
from qrc_dataset_profiler.properties import profile_dataset
from qrc_dataset_profiler.reservoir import StandardSpinV1
from qrc_dataset_profiler.run_study import _study_horizon, _study_spec
from qrc_dataset_profiler.spec import Dataset, DatasetSpec


DEFAULT_QRC_GRID = {
    "J": (0.8, 1.0, 1.2),
    "h": (1.0,),
    "dt": (0.20, 0.25),
    "amplitude_damping": (0.0, 0.02),
    "dephasing": (0.0, 0.01),
}
DEFAULT_ESN_GRID = {
    "rho": (0.7, 0.9, 1.0, 1.1, 1.3),
    "leak": (0.1, 0.3, 0.6, 1.0),
    "input_scale": (0.3, 1.0, 2.0),
}


def run_global_calibration(
    *,
    out_dir: Path,
    sweep_seed: int = 917,
    sweep_n_per_family: int = 2,
    calibration_rows_per_family: int = 3,
    fast: bool = True,
    seeds: int = 1,
    n_qubits: int | None = None,
    n_qubits_options: Iterable[int] | None = None,
    depth: int = 5,
    depth_options: Iterable[int] | None = None,
    virtual_nodes: int = 5,
    virtual_nodes_options: Iterable[int] | None = None,
    taxonomy: str = "v3",
    selection_tolerance: float = 0.0,
    qrc_grid: dict[str, Iterable[float]] | None = None,
    esn_grid: dict[str, Iterable[float]] | None = None,
) -> dict[str, Any]:
    """Calibrate one global QRC config and one global ESN config on held-out specs.

    The selected settings are meant to be frozen and passed to ``run_study`` for the
    primary fixed-vs-fixed atlas.  Reservoir hyperparameters are selected once by
    mean validation NRMSE across calibration datasets, never per evaluation dataset.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    if taxonomy not in {"v3", "v4"}:
        raise ValueError("taxonomy must be 'v3' or 'v4'")
    n_qubits = int(6 if n_qubits is None and fast else 8 if n_qubits is None else n_qubits)
    n_qubits_values = tuple(int(v) for v in (n_qubits_options if n_qubits_options is not None else (n_qubits,)))
    depth_values = tuple(int(v) for v in (depth_options if depth_options is not None else (depth,)))
    virtual_node_values = tuple(int(v) for v in (virtual_nodes_options if virtual_nodes_options is not None else (virtual_nodes,)))
    seed_values = tuple(range(max(1, int(seeds))))
    sweep_specs = make_sweep_specs_v4(n_per_template=sweep_n_per_family, seed=sweep_seed) if taxonomy == "v4" else make_sweep_specs(n_per_family=sweep_n_per_family, seed=sweep_seed)
    specs = select_calibration_specs(
        sweep_specs,
        rows_per_family=calibration_rows_per_family,
        seed=sweep_seed,
    )
    datasets, spec_rows = _materialize_calibration_datasets(specs, fast=fast)
    if not datasets:
        raise ValueError("no calibration datasets were generated")

    qrc_candidates = list(
        _qrc_candidates(
            qrc_grid or DEFAULT_QRC_GRID,
            n_qubits=n_qubits_values,
            depth=depth_values,
            virtual_nodes=virtual_node_values,
        )
    )
    esn_candidates = list(_esn_candidates(esn_grid or DEFAULT_ESN_GRID))
    qrc_scores = _score_qrc_candidates_checkpointed(datasets, qrc_candidates, seed_values, out_dir=out_dir)
    best_qrc_idx = _best_qrc_index(qrc_candidates, qrc_scores, tolerance=float(selection_tolerance))
    best_qrc = qrc_candidates[best_qrc_idx]
    esn_scores = _score_esn_candidates_checkpointed(datasets, best_qrc, esn_candidates, seed_values, out_dir=out_dir)
    best_esn_idx = _best_index(esn_scores)
    best_esn = esn_candidates[best_esn_idx]

    qrc_table = pd.DataFrame([_qrc_row(cfg, score) for cfg, score in zip(qrc_candidates, qrc_scores)])
    esn_table = pd.DataFrame([_esn_row(cand, score) for cand, score in zip(esn_candidates, esn_scores)])
    calibration_catalog = pd.DataFrame(spec_rows)
    qrc_table.to_csv(out_dir / "qrc_calibration_scores.csv", index=False)
    esn_table.to_csv(out_dir / "esn_calibration_scores.csv", index=False)
    calibration_catalog.to_csv(out_dir / "calibration_catalog.csv", index=False)

    manifest = {
        "analysis_version": "global-reservoir-calibration-v1",
        "protocol_layer": "fixed_vs_fixed_primary_fairness",
        "comparison_protocol": "standard_v3",
        "calibration_data": {
            "source": "synthetic held-out sweep variants",
            "taxonomy": taxonomy,
            "sweep_seed": int(sweep_seed),
            "sweep_n_per_family": int(sweep_n_per_family),
            "rows_per_family": int(calibration_rows_per_family),
            "n_rows": int(len(datasets)),
            "families": sorted({ds.spec.family for ds in datasets}),
            "fast": bool(fast),
        },
        "qrc_candidate_grid": {
            "n_qubits": list(n_qubits_values),
            "depth": list(depth_values),
            "virtual_nodes": list(virtual_node_values),
            "n_candidates": int(len(qrc_candidates)),
        },
        "esn_candidate_grid": {
            "n_candidates": int(len(esn_candidates)),
        },
        "selection_metric": "mean validation NRMSE across held-out calibration datasets and seeds",
        "selection_tolerance": float(selection_tolerance),
        "selection_tie_breaker": "smallest feature_dim within tolerance of best validation NRMSE, then lowest validation NRMSE",
        "qrc": _standard_spin_manifest(best_qrc),
        "esn": {
            "class": "frozen_sparse_random_leaky_esn",
            "reservoir_size": int(best_qrc.feature_dim),
            "density": 0.1,
            "bias_scale": 0.2,
            "rho": float(best_esn["rho"]),
            "leak": float(best_esn["leak"]),
            "input_scale": float(best_esn["input_scale"]),
            "hyperparameter_selection": "selected_once_on_held_out_calibration_set_then_frozen",
        },
        "scores": {
            "best_qrc_mean_val_nrmse": float(qrc_scores[best_qrc_idx]["mean_val_nrmse"]),
            "best_qrc_mean_test_nrmse_on_calibration": float(qrc_scores[best_qrc_idx]["mean_test_nrmse"]),
            "best_esn_mean_val_nrmse": float(esn_scores[best_esn_idx]["mean_val_nrmse"]),
            "best_esn_mean_test_nrmse_on_calibration": float(esn_scores[best_esn_idx]["mean_test_nrmse"]),
        },
        "outputs": [
            "calibration_catalog.csv",
            "esn_calibration_scores.csv",
            "frozen_config.json",
            "qrc_calibration_scores.csv",
        ],
        "claim_boundary": (
            "This calibration only selects global reservoir hyperparameters for a fair fixed-vs-fixed atlas. "
            "It does not use evaluation-atlas test labels and does not establish quantum advantage."
        ),
    }
    (out_dir / "frozen_config.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def select_calibration_specs(specs: list[DatasetSpec], *, rows_per_family: int, seed: int) -> list[DatasetSpec]:
    """Return a deterministic balanced held-out calibration subset."""

    rng = np.random.default_rng(seed)
    rows: list[DatasetSpec] = []
    frame = pd.DataFrame([asdict(spec) for spec in specs])
    for family in sorted(frame["family"].unique()):
        idx = frame.index[frame["family"] == family].to_numpy()
        if idx.size == 0:
            continue
        chosen = np.sort(rng.choice(idx, size=min(int(rows_per_family), idx.size), replace=False))
        rows.extend(specs[int(i)] for i in chosen)
    return rows


def _materialize_calibration_datasets(specs: list[DatasetSpec], *, fast: bool) -> tuple[list[Dataset], list[dict[str, Any]]]:
    datasets: list[Dataset] = []
    rows: list[dict[str, Any]] = []
    for base_spec in specs:
        spec = _study_spec(base_spec, smoke=False, fast=fast)
        ds = generate(spec)
        if ds.ground_truth.get("_unavailable"):
            continue
        rec = profile_dataset(ds)
        horizon = _study_horizon(spec, rec.ac_timescale, override=None)
        if horizon != spec.horizon:
            spec = replace(spec, horizon=horizon)
            ds = Dataset(spec, ds.series, inputs=ds.inputs, ground_truth=ds.ground_truth)
        datasets.append(ds)
        rows.append(
            {
                "name": spec.name,
                "family": spec.family,
                "source": spec.source,
                "task_type": spec.task_type,
                "seed": spec.seed,
                "length": spec.length,
                "horizon": spec.horizon,
                "params": repr(spec.params),
            }
        )
    return datasets, rows


def _qrc_candidates(
    grid: dict[str, Iterable[float]],
    *,
    n_qubits: Iterable[int],
    depth: Iterable[int],
    virtual_nodes: Iterable[int],
) -> Iterable[StandardSpinV1]:
    keys = ("J", "h", "dt", "amplitude_damping", "dephasing")
    values = [tuple(float(v) for v in grid[key]) for key in keys]
    for nq, dep, virt, J, h, dt, amp, deph in product(tuple(n_qubits), tuple(depth), tuple(virtual_nodes), *values):
        if int(virt) > int(dep):
            continue
        yield StandardSpinV1(
            n_qubits=int(nq),
            J=J,
            h=h,
            dt=dt,
            depth=int(dep),
            topology="ring",
            virtual_nodes=int(virt),
            reupload=False,
            amplitude_damping=amp,
            dephasing=deph,
            dissipation_method="trajectory",
        )


def _esn_candidates(grid: dict[str, Iterable[float]]) -> Iterable[dict[str, float]]:
    for rho, leak, input_scale in product(grid["rho"], grid["leak"], grid["input_scale"]):
        yield {"rho": float(rho), "leak": float(leak), "input_scale": float(input_scale)}


def _score_qrc_candidates_checkpointed(
    datasets: list[Dataset],
    candidates: list[StandardSpinV1],
    seeds: tuple[int, ...],
    *,
    out_dir: Path,
) -> list[dict[str, float]]:
    path = out_dir / "qrc_calibration_scores.csv"
    existing = _load_existing_rows(path, key_cols=_qrc_key_columns())
    rows: list[dict[str, Any]] = []
    scores: list[dict[str, float]] = []
    for idx, cfg in enumerate(candidates, start=1):
        key = _qrc_key(cfg)
        cached = existing.get(_key_tuple(key))
        if cached is not None:
            score = {k: float(cached[k]) for k in ("mean_val_nrmse", "median_val_nrmse", "mean_test_nrmse", "median_test_nrmse")}
            row = {**key, **score}
        else:
            score = _score_qrc_candidate(datasets, cfg, seeds)
            row = _qrc_row(cfg, score)
            existing[_key_tuple(key)] = row
        rows.append(row)
        scores.append(score)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"qrc calibration candidate {idx}/{len(candidates)} feature_dim={cfg.feature_dim} val={score['mean_val_nrmse']:.6g}", flush=True)
    return scores


def _score_esn_candidates_checkpointed(
    datasets: list[Dataset],
    qrc_cfg: StandardSpinV1,
    candidates: list[dict[str, float]],
    seeds: tuple[int, ...],
    *,
    out_dir: Path,
) -> list[dict[str, float]]:
    path = out_dir / "esn_calibration_scores.csv"
    key_cols = ("qrc_feature_dim", "rho", "leak", "input_scale")
    existing = _load_existing_rows(path, key_cols=key_cols)
    rows: list[dict[str, Any]] = []
    scores: list[dict[str, float]] = []
    for idx, cand in enumerate(candidates, start=1):
        key = {"qrc_feature_dim": int(qrc_cfg.feature_dim), **cand}
        cached = existing.get(_key_tuple(key))
        if cached is not None:
            score = {k: float(cached[k]) for k in ("mean_val_nrmse", "median_val_nrmse", "mean_test_nrmse", "median_test_nrmse")}
            row = {**key, **score}
        else:
            score = _score_esn_candidate(datasets, qrc_cfg, cand, seeds)
            row = {"qrc_feature_dim": int(qrc_cfg.feature_dim), **_esn_row(cand, score)}
            existing[_key_tuple(key)] = row
        rows.append(row)
        scores.append(score)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"esn calibration candidate {idx}/{len(candidates)} val={score['mean_val_nrmse']:.6g}", flush=True)
    return scores


def _qrc_key_columns() -> tuple[str, ...]:
    return (
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
        "coupling_mode",
        "coupling_seed",
    )


def _qrc_key(cfg: StandardSpinV1) -> dict[str, Any]:
    return {
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
        "coupling_mode": cfg.coupling_mode,
        "coupling_seed": int(cfg.coupling_seed),
    }


def _load_existing_rows(path: Path, *, key_cols: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if not set(key_cols).issubset(frame.columns):
        return {}
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        if all(col in row for col in key_cols):
            out[_key_tuple({col: row[col] for col in key_cols})] = row
    return out


def _key_tuple(row: dict[str, Any]) -> tuple[Any, ...]:
    vals: list[Any] = []
    for key in sorted(row):
        val = row[key]
        if isinstance(val, (float, np.floating)):
            vals.append(round(float(val), 12))
        elif isinstance(val, (bool, np.bool_)):
            vals.append(bool(val))
        elif isinstance(val, (int, np.integer)):
            vals.append(int(val))
        else:
            vals.append(val)
    return tuple(vals)


def _score_qrc_candidate(datasets: list[Dataset], cfg: StandardSpinV1, seeds: tuple[int, ...]) -> dict[str, float]:
    val_scores: list[float] = []
    test_scores: list[float] = []
    for ds in datasets:
        for seed in seeds:
            scores = qrc_scores_standard(ds, cfg, seed=seed)
            val_scores.append(scores["val_nrmse"])
            test_scores.append(scores["test_nrmse"])
    return _score_summary(val_scores, test_scores)


def _score_esn_candidate(
    datasets: list[Dataset],
    qrc_cfg: StandardSpinV1,
    cand: dict[str, float],
    seeds: tuple[int, ...],
) -> dict[str, float]:
    val_scores: list[float] = []
    test_scores: list[float] = []
    singleton = {key: (value,) for key, value in cand.items()}
    for ds in datasets:
        for seed in seeds:
            scores = esn_sparse_baseline(ds, qrc_cfg=qrc_cfg, seed=seed, esn_grid=singleton, return_details=True)
            val_scores.append(float(scores["val_nrmse"]))
            test_scores.append(float(scores["nrmse"]))
    return _score_summary(val_scores, test_scores)


def _score_summary(val_scores: list[float], test_scores: list[float]) -> dict[str, float]:
    return {
        "mean_val_nrmse": _safe_mean(val_scores),
        "median_val_nrmse": _safe_median(val_scores),
        "mean_test_nrmse": _safe_mean(test_scores),
        "median_test_nrmse": _safe_median(test_scores),
    }


def _qrc_row(cfg: StandardSpinV1, score: dict[str, float]) -> dict[str, Any]:
    return {
        "n_qubits": int(cfg.n_qubits),
        "J": cfg.J,
        "h": cfg.h,
        "dt": cfg.dt,
        "depth": int(cfg.depth),
        "topology": cfg.topology,
        "virtual_nodes": int(cfg.virtual_nodes),
        "reupload": bool(cfg.reupload),
        "amplitude_damping": cfg.amplitude_damping,
        "dephasing": cfg.dephasing,
        "coupling_mode": cfg.coupling_mode,
        "coupling_seed": int(cfg.coupling_seed),
        "feature_dim": cfg.feature_dim,
        **score,
    }


def _esn_row(cand: dict[str, float], score: dict[str, float]) -> dict[str, Any]:
    return {**cand, **score}


def _best_index(scores: list[dict[str, float]]) -> int:
    vals = np.asarray([s["mean_val_nrmse"] for s in scores], dtype=float)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    if np.all(np.isinf(vals)):
        raise ValueError("all calibration candidates failed")
    return int(np.argmin(vals))


def _best_qrc_index(candidates: list[StandardSpinV1], scores: list[dict[str, float]], *, tolerance: float) -> int:
    vals = np.asarray([s["mean_val_nrmse"] for s in scores], dtype=float)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    if np.all(np.isinf(vals)):
        raise ValueError("all QRC calibration candidates failed")
    best = float(np.min(vals))
    tol = max(0.0, float(tolerance))
    eligible = [i for i, val in enumerate(vals) if float(val) <= best + tol]
    if not eligible:
        return int(np.argmin(vals))
    eligible.sort(key=lambda i: (int(candidates[i].feature_dim), int(candidates[i].n_qubits), float(vals[i])))
    return int(eligible[0])


def _standard_spin_manifest(cfg: StandardSpinV1) -> dict[str, Any]:
    return {
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
        "encoding": "train_split_scaled_input_qubit_injection_no_rz_reupload_with_fixed_local_dissipation",
        "hyperparameter_selection": "selected_once_on_held_out_calibration_set_then_frozen",
    }


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else math.nan


def _safe_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else math.nan


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
