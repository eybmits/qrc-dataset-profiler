"""Corrected paired J=1 vs J=0 quantum-attribution protocol.

This module reruns the QRC readout on the same datasets with matched seeds and
identical reservoir dimensions.  It deliberately reports negative or null
paired effects as first-class results.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from qrc_dataset_profiler.analysis import load_catalog
from qrc_dataset_profiler.baselines import qrc_nrmse_standard as qrc_nrmse
from qrc_dataset_profiler.generators import generate
from qrc_dataset_profiler.reservoir import StandardSpinV1
from qrc_dataset_profiler.run_study import _load_calibration_config, _qrc_from_calibration_config
from qrc_dataset_profiler.spec import Dataset, DatasetSpec


DEFAULT_ATTRIBUTION_FAMILIES = ("chaotic_flow", "chaotic_map")


def run_quantum_attribution(
    catalog: pd.DataFrame,
    *,
    out_dir: Path,
    families: Iterable[str] = DEFAULT_ATTRIBUTION_FAMILIES,
    seeds: int = 1,
    n_qubits: int = 6,
    depth: int = 5,
    virtual_nodes: int = 5,
    dt: float = 0.25,
    h: float = 1.0,
    topology: str = "ring",
    shots: int | None = None,
    amplitude_damping: float = 0.02,
    dephasing: float = 0.01,
    dissipation_method: str = "trajectory",
    calibration_config: Path | str | None = None,
    bootstrap_replicates: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the corrected paired attribution experiment and write artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_catalog_rows(catalog, tuple(families))
    if calibration_config is not None:
        frozen = _load_calibration_config(calibration_config)
        assert frozen is not None
        qrc_j1 = replace(_qrc_from_calibration_config(frozen), shots=shots)
    else:
        qrc_j1 = StandardSpinV1(
            n_qubits=n_qubits,
            J=1.0,
            h=h,
            dt=dt,
            depth=depth,
            topology=topology,
            virtual_nodes=virtual_nodes,
            shots=shots,
            reupload=False,
            amplitude_damping=float(amplitude_damping),
            dephasing=float(dephasing),
            dissipation_method=dissipation_method,
        )
    qrc_j0 = replace(qrc_j1, J=0.0)
    if qrc_j1.feature_dim != qrc_j0.feature_dim:
        raise RuntimeError("paired J=1 and J=0 reservoirs must have identical feature_dim")

    seed_values = tuple(range(max(1, int(seeds))))
    rows: list[dict[str, Any]] = []
    for _row_idx, row in selected.iterrows():
        spec = spec_from_catalog_row(row)
        ds = generate(spec)
        if ds.ground_truth.get("_unavailable"):
            continue
        ds = Dataset(spec, ds.series, inputs=ds.inputs, ground_truth=ds.ground_truth)
        j1_vals = [qrc_nrmse(ds, qrc_j1, seed=s) for s in seed_values]
        j0_vals = [qrc_nrmse(ds, qrc_j0, seed=s) for s in seed_values]
        nrmse_j1 = float(np.mean(j1_vals))
        nrmse_j0 = float(np.mean(j0_vals))
        esn = _float_or_nan(row.get("nrmse_esn_matched"))
        rows.append(
            {
                "source_index": int(row.get("original_index", _row_idx)),
                "name": str(row["name"]),
                "family": str(row["family"]),
                "task_type": str(row["task_type"]),
                "seed": int(spec.seed),
                "length": int(spec.length),
                "horizon": int(spec.horizon),
                "nrmse_esn_matched": esn,
                "nrmse_qrc_J1": nrmse_j1,
                "nrmse_qrc_J0": nrmse_j0,
                "advantage_J1_vs_esn": esn - nrmse_j1 if math.isfinite(esn) else np.nan,
                "advantage_J0_vs_esn": esn - nrmse_j0 if math.isfinite(esn) else np.nan,
                "paired_delta_J0_minus_J1": nrmse_j0 - nrmse_j1,
                "paired_effect_J1_better": nrmse_j0 - nrmse_j1,
                "feature_dim": int(qrc_j1.feature_dim),
                "n_qubits": int(n_qubits),
                "depth": int(depth),
                "virtual_nodes": int(virtual_nodes),
                "matched_qrc_seeds": ",".join(str(s) for s in seed_values),
            }
        )

    paired = pd.DataFrame(rows)
    if paired.empty:
        raise ValueError("no attribution rows were produced")
    summary = summarize_attribution(paired, bootstrap_replicates=bootstrap_replicates, seed=seed)
    paired.to_csv(out_dir / "paired_attribution.csv", index=False)
    summary.to_csv(out_dir / "family_attribution_bootstrap.csv", index=False)
    _write_attribution_figure(summary, out_dir / "family_attribution_bootstrap.png")

    manifest = {
        "analysis_version": "quantum-attribution-v1",
        "catalog_rows_input": int(len(catalog)),
        "catalog_rows_selected": int(len(selected)),
        "rows_written": int(len(paired)),
        "families": sorted(str(f) for f in paired["family"].unique()),
        "matched_qrc_seeds": list(seed_values),
        "bootstrap_replicates": int(bootstrap_replicates),
        "reservoir": {
            "J_values": [float(qrc_j1.J), 0.0],
            "h": float(qrc_j1.h),
            "dt": float(qrc_j1.dt),
            "n_qubits": int(qrc_j1.n_qubits),
            "depth": int(qrc_j1.depth),
            "virtual_nodes": int(qrc_j1.virtual_nodes),
            "topology": qrc_j1.topology,
            "shots": shots,
            "reupload": bool(qrc_j1.reupload),
            "amplitude_damping": float(qrc_j1.amplitude_damping),
            "dephasing": float(qrc_j1.dephasing),
            "dissipation_method": qrc_j1.dissipation_method,
            "feature_dim_J1": int(qrc_j1.feature_dim),
            "feature_dim_J0": int(qrc_j0.feature_dim),
        },
        "calibration_config": str(calibration_config) if calibration_config is not None else None,
        "scoring_protocol": "qrc_nrmse_standard with train-split-only input scaling",
        "paired_effect_definition": "paired_delta_J0_minus_J1 = nrmse_qrc_J0 - nrmse_qrc_J1; positive means J=1 has lower NRMSE.",
        "claim_boundary": (
            "Only a robust positive paired effect would support a coupling-attribution claim. "
            "Null or negative intervals must be reported as evidence against that mechanism claim."
        ),
        "outputs": [
            "attribution_manifest.json",
            "family_attribution_bootstrap.csv",
            "family_attribution_bootstrap.png",
            "paired_attribution.csv",
        ],
    }
    (out_dir / "attribution_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def run_quantum_attribution_from_path(catalog_path: Path, **kwargs: Any) -> dict[str, Any]:
    return run_quantum_attribution(load_catalog(catalog_path), **kwargs)


def spec_from_catalog_row(row: pd.Series) -> DatasetSpec:
    params_raw = row.get("params", "{}")
    params = ast.literal_eval(params_raw) if isinstance(params_raw, str) else dict(params_raw)
    return DatasetSpec(
        name=str(row["name"]),
        family=str(row["family"]),
        source=str(row.get("source", "synthetic")),
        task_type=str(row["task_type"]),
        params=params,
        seed=int(row.get("seed", 0)),
        length=int(row.get("length", 800)),
        n_channels=int(row.get("n_channels", 1)),
        horizon=int(row.get("horizon", 1)),
    )


def summarize_attribution(paired: pd.DataFrame, *, bootstrap_replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for family, group in _iter_groups_with_overall(paired):
        delta = _finite(group["paired_delta_J0_minus_J1"])
        adv_j1 = _finite(group["advantage_J1_vs_esn"])
        adv_j0 = _finite(group["advantage_J0_vs_esn"])
        boot_delta = _bootstrap_mean(delta, n_bootstraps=bootstrap_replicates, rng=rng)
        rows.append(
            {
                "family": family,
                "n": int(delta.size),
                "mean_delta_J0_minus_J1": _safe_mean(delta),
                "delta_ci_low": _percentile(boot_delta, 2.5),
                "delta_ci_high": _percentile(boot_delta, 97.5),
                "median_delta_J0_minus_J1": _safe_median(delta),
                "frac_J1_better": _safe_mean(delta > 0.0),
                "mean_advantage_J1_vs_esn": _safe_mean(adv_j1),
                "mean_advantage_J0_vs_esn": _safe_mean(adv_j0),
                "mechanism_signal": _mechanism_signal(_percentile(boot_delta, 2.5), _percentile(boot_delta, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def _select_catalog_rows(catalog: pd.DataFrame, families: tuple[str, ...]) -> pd.DataFrame:
    required = ("name", "family", "task_type", "params", "seed", "length", "horizon")
    missing = [c for c in required if c not in catalog.columns]
    if missing:
        raise ValueError(f"catalog is missing required columns: {', '.join(missing)}")
    if families:
        selected = catalog[catalog["family"].astype(str).isin(set(families))].copy()
    else:
        selected = catalog.copy()
    if selected.empty:
        raise ValueError("family filter selected no rows")
    return selected.reset_index(drop=False).rename(columns={"index": "original_index"})


def _iter_groups_with_overall(df: pd.DataFrame):
    yield "overall", df
    for family, group in df.groupby("family", sort=True):
        yield str(family), group


def _bootstrap_mean(values: np.ndarray, *, n_bootstraps: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or n_bootstraps <= 0:
        return np.asarray([], dtype=float)
    idx = rng.integers(0, values.size, size=(int(n_bootstraps), values.size))
    return values[idx].mean(axis=1)


def _write_attribution_figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    data = summary[summary["family"] != "overall"].sort_values("mean_delta_J0_minus_J1", ascending=True)
    y = np.arange(len(data))
    mean = data["mean_delta_J0_minus_J1"].to_numpy(dtype=float)
    low = data["delta_ci_low"].to_numpy(dtype=float)
    high = data["delta_ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    ax.errorbar(mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="o", color="#4c6f7f", ecolor="#6b7280", capsize=3)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["family"])
    ax.set_xlabel("Mean paired delta: NRMSE(J=0) - NRMSE(J=1)")
    ax.set_title("Corrected paired coupling attribution")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _mechanism_signal(low: float, high: float) -> str:
    if math.isfinite(low) and low > 0.0:
        return "positive"
    if math.isfinite(high) and high < 0.0:
        return "negative"
    return "null_or_mixed"


def _finite(series: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)) if arr.size else np.nan


def _safe_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)) if arr.size else np.nan


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
