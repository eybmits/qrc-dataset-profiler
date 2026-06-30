"""External real-world validation probes for the synthetic QRC regime atlas.

This module keeps the paper design clean:

* the regime map is trained on the completed synthetic discovery atlas;
* real-world series are scored as external probes with atlas support/OOD scores;
* optional real labels are computed only on a stratified held-out subset.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score

from qrc_dataset_profiler.analysis import load_catalog
from qrc_dataset_profiler.baselines import esn_sparse_baseline, linear_baseline, qrc_scores_standard
from qrc_dataset_profiler.frontier import compute_support_scores, materialize_frontier_features
from qrc_dataset_profiler.paper_robustness import (
    _feature_matrices,
    _load_csv_series,
    _load_noaa_co2,
    _load_silso_sunspots,
    _url_text,
    nvar_baseline_scores,
)
from qrc_dataset_profiler.properties import compute_backstop, profile_dataset
from qrc_dataset_profiler.run_study import _esn_grid_from_calibration_config, _load_calibration_config, _qrc_from_calibration_config
from qrc_dataset_profiler.spec import Dataset, DatasetSpec, FRONTIER_TIER_A_FIELDS


REAL_VALIDATION_VERSION = "real-external-validation-v1"


@dataclass(frozen=True)
class RealSeries:
    source_name: str
    series_id: str
    domain: str
    description: str
    url: str
    values: np.ndarray


@dataclass(frozen=True)
class RealSource:
    name: str
    domain: str
    description: str
    url: str
    loader: Callable[[int], list[RealSeries]]


def build_real_probe_atlas(
    *,
    out_dir: Path,
    window_length: int = 800,
    min_length: int = 120,
    max_windows: int = 120,
    max_windows_per_series: int = 3,
    seed: int = 0,
    include_m4: bool = True,
    include_nab: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """Fetch real benchmark series, window them, and compute the 30 features."""

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    errors: list[dict[str, str]] = []
    sources = real_sources(include_m4=include_m4, include_nab=include_nab)
    for source in sources:
        try:
            series_list = source.loader(seed)
        except Exception as exc:  # pragma: no cover - network sources can fail
            errors.append({"source_name": source.name, "url": source.url, "error": repr(exc)})
            continue
        for series in series_list:
            windows = _windows(series.values, window_length=window_length, min_length=min_length, max_windows=max_windows_per_series)
            for win_idx, start, values in windows:
                window_key = f"w{len(rows):05d}"
                spec = DatasetSpec(
                    name=f"{series.source_name}_{series.series_id}_w{win_idx}",
                    family=f"real_{series.domain}",
                    source="real",
                    task_type="forecast",
                    params={
                        "source_name": series.source_name,
                        "series_id": series.series_id,
                        "source_url": series.url,
                        "window_start": int(start),
                        "window_stop": int(start + values.size),
                    },
                    seed=int(seed),
                    length=int(values.size),
                    horizon=1,
                )
                ds = Dataset(spec, values)
                try:
                    rec = profile_dataset(ds)
                    row = {**rec.to_row(), **compute_backstop(values)}
                except Exception as exc:
                    errors.append({"source_name": series.source_name, "url": series.url, "error": f"profile {series.series_id}: {exc!r}"})
                    continue
                row.update(
                    {
                        "window_key": window_key,
                        "source_name": series.source_name,
                        "series_id": series.series_id,
                        "real_domain": series.domain,
                        "source_url": series.url,
                        "source_description": series.description,
                        "window_start": int(start),
                        "window_stop": int(start + values.size),
                        "base_generator": f"real_{series.source_name}",
                    }
                )
                rows.append(row)
                arrays[window_key] = np.asarray(values, dtype=float)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "real_probe_fetch_errors.csv", index=False)
    if not rows:
        raise ValueError("no real probe rows were produced")
    atlas = materialize_frontier_features(pd.DataFrame(rows))
    # Stable bounded sampling prevents source-order dominance while staying deterministic.
    order = rng.permutation(len(atlas))[: int(max_windows)]
    atlas = atlas.iloc[order].reset_index(drop=True)
    arrays = {str(k): arrays[str(k)] for k in atlas["window_key"].astype(str) if str(k) in arrays}
    path = out_dir / "real_window_property_atlas.csv"
    atlas.to_csv(path, index=False)
    np.savez_compressed(out_dir / "real_windows.npz", **arrays)
    manifest = {
        "analysis_version": REAL_VALIDATION_VERSION,
        "artifact": "real_window_property_atlas",
        "n_rows": int(len(atlas)),
        "window_length": int(window_length),
        "min_length": int(min_length),
        "max_windows_requested": int(max_windows),
        "max_windows_per_series": int(max_windows_per_series),
        "seed": int(seed),
        "include_m4": bool(include_m4),
        "include_nab": bool(include_nab),
        "sources_requested": [source.name for source in sources],
        "sources_written": sorted(atlas["source_name"].dropna().astype(str).unique().tolist()),
        "domains": _count_dict(atlas.get("real_domain")),
        "claim_boundary": "Real property probes are external validation inputs; no real QRC/ESN labels are used to train the synthetic regime map.",
        "outputs": ["real_window_property_atlas.csv", "real_windows.npz", "real_probe_property_manifest.json"],
    }
    (out_dir / "real_probe_property_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return atlas, path


def score_real_probe_atlas(
    *,
    property_atlas: Path,
    discovery_table: Path,
    out_dir: Path,
    seed: int = 0,
) -> tuple[pd.DataFrame, Path]:
    """Score real probes with a synthetic-discovery-trained meta-model."""

    out_dir.mkdir(parents=True, exist_ok=True)
    real = materialize_frontier_features(load_catalog(property_atlas))
    discovery = materialize_frontier_features(load_catalog(discovery_table))
    discovery = discovery[np.isfinite(pd.to_numeric(discovery["qrc_advantage"], errors="coerce").to_numpy(dtype=float))].reset_index(drop=True)
    X_train, X_real, features_used = _feature_matrices(discovery, real, tuple(FRONTIER_TIER_A_FIELDS))
    y_train = pd.to_numeric(discovery["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    reg = GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=seed).fit(X_train, y_train)
    real["predicted_qrc_advantage"] = reg.predict(X_real)
    y_train_bin = y_train >= 0.05
    if len(np.unique(y_train_bin)) == 2:
        clf = GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=seed).fit(X_train, y_train_bin)
        real["predicted_prob_qrc_useful"] = clf.predict_proba(X_real)[:, 1]
    else:
        real["predicted_prob_qrc_useful"] = np.nan
    support = compute_support_scores(discovery, real, k_values=(15, 30, 50))
    for col in ("support_score", "ood_flag", "family_entropy", "nearest_family_mixture"):
        real[col] = support[col].to_numpy()
    real["external_validation_role"] = "prediction_only_unlabeled_real_probe"
    path = out_dir / "real_window_predictions.csv"
    real.to_csv(path, index=False)
    source_summary = summarize_real_predictions(real)
    source_summary.to_csv(out_dir / "real_window_source_summary.csv", index=False)
    _write_real_prediction_figure(real, out_dir / "real_window_prediction_map.png")
    manifest = {
        "analysis_version": REAL_VALIDATION_VERSION,
        "artifact": "real_window_predictions",
        "property_atlas": str(property_atlas),
        "discovery_table": str(discovery_table),
        "n_rows": int(len(real)),
        "n_ood": int(real["ood_flag"].sum()) if "ood_flag" in real else 0,
        "features_used": list(features_used),
        "claim_boundary": "Predictions are synthetic-trained external probe scores. They are not real performance labels.",
        "outputs": ["real_window_predictions.csv", "real_window_source_summary.csv", "real_window_prediction_map.png", "real_prediction_manifest.json"],
    }
    (out_dir / "real_prediction_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return real, path


def select_real_label_subset(
    *,
    predictions_path: Path,
    out_dir: Path,
    n_rows: int = 32,
    seed: int = 0,
) -> tuple[pd.DataFrame, Path]:
    """Select a target-free real subset to label with frozen QRC/ESN."""

    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_catalog(predictions_path).copy()
    if df.empty:
        raise ValueError("prediction table is empty")
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    used: set[int] = set()

    def take(role: str, mask: pd.Series, k: int) -> None:
        sub = df.loc[mask.fillna(False) & ~df.index.isin(used)].copy()
        if sub.empty or k <= 0:
            return
        idx = rng.choice(sub.index.to_numpy(), size=min(k, len(sub)), replace=False)
        piece = df.loc[idx].copy()
        piece["real_selection_role"] = role
        parts.append(piece)
        used.update(int(i) for i in idx)

    n_rows = int(n_rows)
    pred = pd.to_numeric(df["predicted_qrc_advantage"], errors="coerce")
    prob = pd.to_numeric(df.get("predicted_prob_qrc_useful"), errors="coerce")
    support = pd.to_numeric(df.get("support_score"), errors="coerce")
    ood = df.get("ood_flag", pd.Series(False, index=df.index)).astype(bool)
    promising_quota = max(1, n_rows // 4)
    before = sum(len(p) for p in parts)
    take("predicted_promising", (pred >= 0.0) & (prob >= prob.median()), promising_quota)
    remaining_promising = promising_quota - (sum(len(p) for p in parts) - before)
    if remaining_promising > 0:
        take("least_unfavorable", pred >= pred.quantile(0.75), remaining_promising)
    take("predicted_unfavorable", pred <= pred.quantile(0.25), max(1, n_rows // 4))
    take("ood_guardrail", ood | (support <= 0.05), max(1, n_rows // 4))
    per_domain = max(1, math.ceil((n_rows - sum(len(p) for p in parts)) / max(1, df["real_domain"].nunique())))
    for _domain, group in df.groupby("real_domain", sort=True):
        take("domain_balanced", pd.Series(df.index.isin(group.index), index=df.index), per_domain)
        if sum(len(p) for p in parts) >= n_rows:
            break
    if sum(len(p) for p in parts) < n_rows:
        take("fill", pd.Series(True, index=df.index), n_rows - sum(len(p) for p in parts))
    selected = pd.concat(parts, ignore_index=False).head(n_rows).reset_index(drop=True)
    path = out_dir / "real_label_selection.csv"
    selected.to_csv(path, index=False)
    manifest = {
        "analysis_version": REAL_VALIDATION_VERSION,
        "artifact": "real_label_selection",
        "prediction_table": str(predictions_path),
        "n_rows": int(len(selected)),
        "seed": int(seed),
        "selection_uses_real_labels": False,
        "selection_roles": _count_dict(selected.get("real_selection_role")),
        "domains": _count_dict(selected.get("real_domain")),
        "claim_boundary": "Selection uses only predictions/support/source metadata, not real QRC/ESN outcomes.",
        "outputs": ["real_label_selection.csv", "real_label_selection_manifest.json"],
    }
    (out_dir / "real_label_selection_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return selected, path


def evaluate_real_label_subset(
    *,
    selection_path: Path,
    windows_path: Path,
    calibration_config: Path,
    out_dir: Path,
    seeds: int = 1,
) -> tuple[pd.DataFrame, Path]:
    """Evaluate frozen QRC/ESN labels on selected real windows."""

    out_dir.mkdir(parents=True, exist_ok=True)
    selected = load_catalog(selection_path)
    windows = np.load(windows_path)
    frozen = _load_calibration_config(calibration_config)
    if frozen is None:
        raise ValueError("calibration_config is required")
    qrc_cfg = _qrc_from_calibration_config(frozen)
    esn_grid = _esn_grid_from_calibration_config(frozen)
    seed_values = tuple(range(max(1, int(seeds))))
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        key = str(row["window_key"])
        if key not in windows:
            continue
        series = np.asarray(windows[key], dtype=float)
        spec = DatasetSpec(
            name=str(row["name"]),
            family=str(row["family"]),
            source="real",
            task_type="forecast",
            params=_safe_params(row),
            seed=int(row.get("seed", 0)),
            length=int(series.size),
            horizon=int(row.get("horizon", 1)),
        )
        ds = Dataset(spec, series)
        qrc_scores = [qrc_scores_standard(ds, qrc_cfg, seed=s) for s in seed_values]
        esn_scores = [esn_sparse_baseline(ds, qrc_cfg=qrc_cfg, seed=s, esn_grid=esn_grid, return_details=True) for s in seed_values]
        nvar = nvar_baseline_scores(ds, lag=20, degree=2)
        linear = linear_baseline(ds)
        qrc_nrmse = _mean_key(qrc_scores, "test_nrmse")
        esn_nrmse = _mean_key(esn_scores, "nrmse")
        qrc_nmae = _mean_key(qrc_scores, "test_nmae")
        esn_nmae = _mean_key(esn_scores, "nmae")
        rows.append(
            {
                **row.to_dict(),
                "nrmse_linear_real": float(linear),
                "nrmse_nvar_real": float(nvar["test_nrmse"]),
                "nmae_nvar_real": float(nvar["test_nmae"]),
                "nrmse_esn_matched_real": esn_nrmse,
                "nrmse_qrc_spin_real": qrc_nrmse,
                "qrc_advantage_real": esn_nrmse - qrc_nrmse,
                "nmae_esn_matched_real": esn_nmae,
                "nmae_qrc_spin_real": qrc_nmae,
                "qrc_nmae_advantage_real": esn_nmae - qrc_nmae,
                "qrc_vs_nvar_nrmse_advantage_real": float(nvar["test_nrmse"]) - qrc_nrmse,
                "qrc_vs_nvar_nmae_advantage_real": float(nvar["test_nmae"]) - qrc_nmae,
                "label_seeds": ",".join(str(s) for s in seed_values),
            }
        )
    labeled = pd.DataFrame(rows)
    if labeled.empty:
        raise ValueError("no real labels were produced")
    path = out_dir / "real_labeled_external_validation.csv"
    labeled.to_csv(path, index=False)
    summary = summarize_real_labels(labeled)
    summary.to_csv(out_dir / "real_labeled_external_summary.csv", index=False)
    _write_real_label_figure(labeled, out_dir / "real_labeled_prediction_vs_observed.png")
    report_path = _write_real_validation_report(out_dir)
    manifest = {
        "analysis_version": REAL_VALIDATION_VERSION,
        "artifact": "real_labeled_external_validation",
        "selection_path": str(selection_path),
        "windows_path": str(windows_path),
        "calibration_config": str(calibration_config),
        "n_rows": int(len(labeled)),
        "seed_count": int(len(seed_values)),
        "summary": summary.to_dict(orient="records"),
        "claim_boundary": "External real labels test transfer of the synthetic-trained map; they are not mixed into training.",
        "outputs": [
            "real_labeled_external_validation.csv",
            "real_labeled_external_summary.csv",
            "real_labeled_prediction_vs_observed.png",
            report_path.name,
            "real_labeled_manifest.json",
        ],
    }
    (out_dir / "real_labeled_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return labeled, path


def summarize_real_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group in _iter_groups(df, "overall"):
        rows.append(_prediction_summary_row(group_name, group))
    for source, group in df.groupby("source_name", sort=True):
        rows.append(_prediction_summary_row(f"source:{source}", group))
    for domain, group in df.groupby("real_domain", sort=True):
        rows.append(_prediction_summary_row(f"domain:{domain}", group))
    return pd.DataFrame(rows)


def summarize_real_labels(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group in _iter_groups(df, "overall"):
        rows.append(_label_summary_row(group_name, group))
    for role, group in df.groupby("real_selection_role", sort=True):
        rows.append(_label_summary_row(f"role:{role}", group))
    for domain, group in df.groupby("real_domain", sort=True):
        rows.append(_label_summary_row(f"domain:{domain}", group))
    return pd.DataFrame(rows)


def real_sources(*, include_m4: bool = True, include_nab: bool = True) -> list[RealSource]:
    sources = [
        RealSource(
            name="silso_sunspots_monthly",
            domain="solar_activity",
            description="SILSO monthly total sunspot number.",
            url="https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv",
            loader=lambda seed: [_single_series("silso_sunspots_monthly", "total", "solar_activity", "SILSO monthly total sunspot number.", "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv", _load_silso_sunspots("https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"))],
        ),
        RealSource(
            name="pjme_hourly_load",
            domain="electricity_load",
            description="PJM East hourly electricity load.",
            url="https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv",
            loader=lambda seed: [_single_series("pjme_hourly_load", "load", "electricity_load", "PJM East hourly electricity load.", "https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv", _load_csv_series("https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv", "PJME_MW"))],
        ),
        RealSource(
            name="melbourne_daily_min_temperature",
            domain="weather",
            description="Daily minimum temperatures in Melbourne.",
            url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
            loader=lambda seed: [_single_series("melbourne_daily_min_temperature", "temp", "weather", "Daily minimum temperatures in Melbourne.", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", _load_csv_series("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", "Temp"))],
        ),
        RealSource(
            name="fred_usd_eur_exchange",
            domain="exchange_rate",
            description="FRED daily USD/EUR exchange rate.",
            url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU",
            loader=lambda seed: [_single_series("fred_usd_eur_exchange", "DEXUSEU", "exchange_rate", "FRED daily USD/EUR exchange rate.", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU", _load_csv_series("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU", "DEXUSEU"))],
        ),
        RealSource(
            name="noaa_mauna_loa_co2",
            domain="environment",
            description="NOAA monthly Mauna Loa CO2.",
            url="https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
            loader=lambda seed: [_single_series("noaa_mauna_loa_co2", "co2", "environment", "NOAA monthly Mauna Loa CO2.", "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv", _load_noaa_co2("https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"))],
        ),
        RealSource(
            name="ett_hourly_etth1",
            domain="electricity_transformer",
            description="ETTh1 transformer oil temperature benchmark.",
            url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
            loader=lambda seed: [_single_series("ett_hourly_etth1", "OT", "electricity_transformer", "ETTh1 transformer oil temperature benchmark.", "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv", _load_csv_series("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv", "OT"))],
        ),
        RealSource(
            name="ett_hourly_etth2",
            domain="electricity_transformer",
            description="ETTh2 transformer oil temperature benchmark.",
            url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
            loader=lambda seed: [_single_series("ett_hourly_etth2", "OT", "electricity_transformer", "ETTh2 transformer oil temperature benchmark.", "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv", _load_csv_series("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv", "OT"))],
        ),
        RealSource(
            name="ett_minute_ettm1",
            domain="electricity_transformer",
            description="ETTm1 transformer oil temperature benchmark.",
            url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
            loader=lambda seed: [_single_series("ett_minute_ettm1", "OT", "electricity_transformer", "ETTm1 transformer oil temperature benchmark.", "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv", _load_csv_series("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv", "OT"))],
        ),
        RealSource(
            name="airline_passengers_monthly",
            domain="transport",
            description="Classic monthly international airline passengers.",
            url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
            loader=lambda seed: [_single_series("airline_passengers_monthly", "passengers", "transport", "Classic monthly international airline passengers.", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", _load_csv_series("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", "Passengers"))],
        ),
    ]
    if include_m4:
        for frequency, n_series in (
            ("Hourly", 20),
            ("Daily", 20),
            ("Weekly", 20),
            ("Monthly", 24),
            ("Quarterly", 24),
        ):
            url = f"https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/{frequency}-train.csv"
            source_name = f"m4_{frequency.lower()}_sample"
            domain = f"m4_{frequency.lower()}"
            description = f"Sampled {frequency.lower()} series from the M4 forecasting competition."
            sources.append(
                RealSource(
                    name=source_name,
                    domain=domain,
                    description=description,
                    url=url,
                    loader=lambda seed, url=url, source_name=source_name, domain=domain, description=description, n_series=n_series: _load_m4_train_sample(
                        url,
                        source_name=source_name,
                        domain=domain,
                        description=description,
                        seed=seed,
                        n_series=n_series,
                    ),
                )
            )
    if include_nab:
        for name, domain, description, url in (
            (
                "nab_temperature_system_failure",
                "industrial_sensor",
                "NAB real-known-cause ambient temperature system-failure series.",
                "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv",
            ),
            (
                "nab_machine_temperature",
                "industrial_sensor",
                "NAB real-known-cause machine-temperature system-failure series.",
                "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv",
            ),
            (
                "nab_traffic_occupancy",
                "traffic",
                "NAB real traffic occupancy sensor series.",
                "https://raw.githubusercontent.com/numenta/NAB/master/data/realTraffic/occupancy_6005.csv",
            ),
            (
                "nab_traffic_speed",
                "traffic",
                "NAB real traffic speed sensor series.",
                "https://raw.githubusercontent.com/numenta/NAB/master/data/realTraffic/speed_7578.csv",
            ),
            (
                "nab_cloud_cpu",
                "cloud_metrics",
                "NAB AWS CloudWatch EC2 CPU utilization series.",
                "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
            ),
            (
                "nab_cloud_network",
                "cloud_metrics",
                "NAB AWS CloudWatch ELB request-count series.",
                "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/elb_request_count_8c0756.csv",
            ),
        ):
            sources.append(
                RealSource(
                    name=name,
                    domain=domain,
                    description=description,
                    url=url,
                    loader=lambda seed, name=name, domain=domain, description=description, url=url: [
                        _single_series(name, "value", domain, description, url, _load_csv_series(url, "value"))
                    ],
                )
            )
    return sources


def _single_series(source_name: str, series_id: str, domain: str, description: str, url: str, values: np.ndarray) -> RealSeries:
    return RealSeries(source_name=source_name, series_id=series_id, domain=domain, description=description, url=url, values=np.asarray(values, dtype=float))


def _load_m4_train_sample(url: str, *, source_name: str, domain: str, description: str, seed: int, n_series: int) -> list[RealSeries]:
    df = pd.read_csv(io.StringIO(_url_text(url)))
    if df.empty:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.to_numpy(), size=min(int(n_series), len(df)), replace=False)
    out: list[RealSeries] = []
    for row_idx in idx:
        row = df.loc[row_idx]
        series_id = str(row.iloc[0])
        values = pd.to_numeric(row.iloc[1:], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size:
            out.append(_single_series(source_name, series_id, domain, description, url, values))
    return out


def _windows(values: np.ndarray, *, window_length: int, min_length: int, max_windows: int) -> list[tuple[int, int, np.ndarray]]:
    x = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if x.size < int(min_length):
        return []
    if x.size <= int(window_length):
        return [(0, 0, x.astype(float))]
    max_start = x.size - int(window_length)
    starts = np.linspace(0, max_start, num=max(1, int(max_windows)))
    starts = sorted(set(int(round(v)) for v in starts))
    return [(i, start, x[start : start + int(window_length)].astype(float)) for i, start in enumerate(starts)]


def _prediction_summary_row(name: str, group: pd.DataFrame) -> dict[str, Any]:
    return {
        "group": name,
        "n": int(len(group)),
        "mean_predicted_qrc_advantage": _safe_mean(group["predicted_qrc_advantage"]),
        "median_predicted_qrc_advantage": _safe_median(group["predicted_qrc_advantage"]),
        "mean_predicted_prob_qrc_useful": _safe_mean(group.get("predicted_prob_qrc_useful")),
        "mean_support_score": _safe_mean(group.get("support_score")),
        "ood_rate": _safe_mean(group.get("ood_flag")),
    }


def _label_summary_row(name: str, group: pd.DataFrame) -> dict[str, Any]:
    pred = pd.to_numeric(group.get("predicted_qrc_advantage"), errors="coerce")
    actual = pd.to_numeric(group.get("qrc_advantage_real"), errors="coerce")
    useful = actual >= 0.05
    row = {
        "group": name,
        "n": int(len(group)),
        "mean_predicted_qrc_advantage": _safe_mean(pred),
        "mean_qrc_advantage_real": _safe_mean(actual),
        "median_qrc_advantage_real": _safe_median(actual),
        "qrc_win_rate_real": _safe_mean(actual > 0.0),
        "qrc_useful_rate_real": _safe_mean(useful),
        "mean_qrc_nmae_advantage_real": _safe_mean(group.get("qrc_nmae_advantage_real")),
        "mean_qrc_vs_nvar_nrmse_advantage_real": _safe_mean(group.get("qrc_vs_nvar_nrmse_advantage_real")),
        "support_score_mean": _safe_mean(group.get("support_score")),
        "ood_rate": _safe_mean(group.get("ood_flag")),
    }
    mask = pred.notna() & actual.notna()
    if int(mask.sum()) >= 3 and float(actual[mask].std()) > 1e-12:
        row["prediction_r2"] = float(r2_score(actual[mask], pred[mask]))
        row["prediction_mae"] = float(mean_absolute_error(actual[mask], pred[mask]))
    else:
        row["prediction_r2"] = np.nan
        row["prediction_mae"] = np.nan
    y_true = useful[mask]
    prob = pd.to_numeric(group.get("predicted_prob_qrc_useful"), errors="coerce")[mask]
    if int(mask.sum()) >= 3 and len(np.unique(y_true)) == 2 and prob.notna().all():
        row["prediction_roc_auc"] = float(roc_auc_score(y_true, prob))
        row["prediction_pr_auc"] = float(average_precision_score(y_true, prob))
        row["prediction_brier"] = float(brier_score_loss(y_true, prob))
    else:
        row["prediction_roc_auc"] = np.nan
        row["prediction_pr_auc"] = np.nan
        row["prediction_brier"] = np.nan
    return row


def _safe_params(row: pd.Series) -> dict[str, Any]:
    return {
        "source_name": row.get("source_name"),
        "series_id": row.get("series_id"),
        "source_url": row.get("source_url"),
        "window_start": int(row.get("window_start", 0)),
        "window_stop": int(row.get("window_stop", row.get("length", 0))),
    }


def _iter_groups(df: pd.DataFrame, overall_name: str):
    yield overall_name, df


def _write_real_prediction_figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    x = pd.to_numeric(df["support_score"], errors="coerce")
    y = pd.to_numeric(df["predicted_qrc_advantage"], errors="coerce")
    colors = np.where(df["ood_flag"].astype(bool), "#9a4f4f", "#2f6f6f")
    ax.scatter(x, y, c=colors, s=28, alpha=0.78, edgecolors="white", linewidths=0.3)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.axhline(0.05, color="#4c6f7f", linewidth=0.8, linestyle="--")
    ax.axvline(0.05, color="#222222", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Synthetic atlas support score")
    ax.set_ylabel("Predicted QRC advantage")
    ax.set_title("Real benchmark probes scored by synthetic regime map")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_real_label_figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    x = pd.to_numeric(df["predicted_qrc_advantage"], errors="coerce")
    y = pd.to_numeric(df["qrc_advantage_real"], errors="coerce")
    colors = np.where(df["ood_flag"].astype(bool), "#9a4f4f", "#2f6f6f")
    ax.scatter(x, y, c=colors, s=36, alpha=0.82, edgecolors="white", linewidths=0.4)
    lo = float(np.nanmin([x.min(), y.min(), -0.05]))
    hi = float(np.nanmax([x.max(), y.max(), 0.05]))
    ax.plot([lo, hi], [lo, hi], color="#222222", linewidth=0.8)
    ax.axhline(0.0, color="#222222", linewidth=0.7)
    ax.axvline(0.0, color="#222222", linewidth=0.7)
    ax.set_xlabel("Synthetic-trained predicted QRC advantage")
    ax.set_ylabel("Observed real QRC advantage")
    ax.set_title("External real-label transfer check")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_real_validation_report(out_dir: Path) -> Path:
    predictions_path = out_dir / "real_window_predictions.csv"
    source_summary_path = out_dir / "real_window_source_summary.csv"
    labeled_path = out_dir / "real_labeled_external_validation.csv"
    label_summary_path = out_dir / "real_labeled_external_summary.csv"
    lines = [
        "# Real-World External Validation",
        "",
        "Main evidence remains the synthetic atlas. This report uses real-world benchmark probes only as external validation of the synthetic-trained regime map.",
        "",
    ]
    if predictions_path.exists():
        predictions = load_catalog(predictions_path)
        lines.extend(
            [
                "## Probe Atlas",
                "",
                f"- Real property windows: `{len(predictions)}`.",
                f"- Real domains: `{predictions['real_domain'].nunique()}`.",
                f"- Sources: `{predictions['source_name'].nunique()}`.",
                f"- OOD windows by synthetic support score: `{int(predictions['ood_flag'].sum())}` / `{len(predictions)}`.",
                f"- Mean predicted QRC advantage: `{_safe_mean(predictions['predicted_qrc_advantage']):.3f}`.",
                "",
            ]
        )
    if source_summary_path.exists():
        source_summary = load_catalog(source_summary_path)
        domains = source_summary[source_summary["group"].astype(str).str.startswith("domain:")].copy()
        if not domains.empty:
            domains = domains.sort_values("mean_predicted_qrc_advantage", ascending=False).head(8)
            lines.extend(["## Highest-Scored Real Domains", ""])
            for _, row in domains.iterrows():
                lines.append(
                    f"- `{row['group']}`: n `{int(row['n'])}`, predicted advantage `{float(row['mean_predicted_qrc_advantage']):.3f}`, "
                    f"support `{float(row['mean_support_score']):.3f}`, OOD rate `{float(row['ood_rate']):.3f}`."
                )
            lines.append("")
    if label_summary_path.exists():
        label_summary = load_catalog(label_summary_path)
        overall = label_summary[label_summary["group"] == "overall"].iloc[0]
        lines.extend(
            [
                "## Frozen-Protocol Real Labels",
                "",
                f"- Labeled windows: `{int(overall['n'])}`.",
                f"- Mean observed QRC advantage: `{float(overall['mean_qrc_advantage_real']):.3f}`.",
                f"- Median observed QRC advantage: `{float(overall['median_qrc_advantage_real']):.3f}`.",
                f"- QRC win rate: `{float(overall['qrc_win_rate_real']):.3f}`.",
                f"- QRC useful rate at advantage >= 0.05: `{float(overall['qrc_useful_rate_real']):.3f}`.",
                f"- Mean NMAE advantage: `{float(overall['mean_qrc_nmae_advantage_real']):.3f}`.",
                f"- Mean QRC-vs-NVAR NRMSE advantage: `{float(overall['mean_qrc_vs_nvar_nrmse_advantage_real']):.3f}`.",
                f"- Prediction R2 on labeled real probes: `{float(overall['prediction_r2']):.3f}`.",
                f"- Prediction ROC-AUC for real QRC-useful labels: `{float(overall['prediction_roc_auc']):.3f}`.",
                "",
            ]
        )
    if labeled_path.exists():
        labeled = load_catalog(labeled_path)
        top = labeled.sort_values("qrc_advantage_real", ascending=False).head(8)
        lines.extend(["## Best Observed Real Windows", ""])
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['name']}` ({row['real_domain']}, {row['real_selection_role']}): "
                f"pred `{float(row['predicted_qrc_advantage']):.3f}`, observed `{float(row['qrc_advantage_real']):.3f}`, "
                f"support `{float(row['support_score']):.3f}`, OOD `{bool(row['ood_flag'])}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "The real probes do not train the regime map and do not replace the synthetic atlas as the main evidence. In this pass they mainly serve as a conservative transfer check: the fixed QRC does not show broad real-world advantage against the frozen feature-matched ESN, and the synthetic ranking does not yet transfer reliably to these real probes.",
            "",
            "Figures: `real_window_prediction_map.png`, `real_labeled_prediction_vs_observed.png`.",
            "",
        ]
    )
    path = out_dir / "REAL_VALIDATION_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _mean_key(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _safe_mean(values: Any) -> float:
    if values is None:
        return np.nan
    series = pd.Series(values)
    if series.dtype == bool:
        series = series.astype(float)
    arr = pd.to_numeric(series, errors="coerce")
    vals = arr.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def _safe_median(values: Any) -> float:
    if values is None:
        return np.nan
    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else np.nan


def _count_dict(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    return {str(k): int(v) for k, v in series.astype(str).value_counts(dropna=False).sort_index().items()}


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
