"""Downstream robustness suite for the v4 QRC regime-atlas paper claim.

The functions here do not redefine the completed v4 atlas labels.  They read the
frozen discovery/validation artifacts, stress-test the "beyond chaos and
nonlinearity" framing, and optionally run small property-defined subsets for
metrics that require fresh prediction traces.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from qrc_dataset_profiler.analysis import load_catalog
from qrc_dataset_profiler.baselines import (
    _lagged_design,
    _protocol_splits,
    _ridge_scores,
    _standardize_train,
    build_task,
    esn_sparse_baseline,
    qrc_scores_standard,
)
from qrc_dataset_profiler.frontier import compute_support_scores, materialize_frontier_features
from qrc_dataset_profiler.generators import generate
from qrc_dataset_profiler.properties import compute_backstop, profile_dataset
from qrc_dataset_profiler.quantum_attribution import spec_from_catalog_row
from qrc_dataset_profiler.run_study import _esn_grid_from_calibration_config, _load_calibration_config, _qrc_from_calibration_config
from qrc_dataset_profiler.spec import Dataset, DatasetSpec, FRONTIER_TIER_A_FIELDS


DIRECT_PREDICTABILITY_FEATURES = (
    "r2_linear",
    "forecastability",
    "pred_nrmse_gbm",
    "predictability_gap_linear_gbm",
)
GB_N_ESTIMATORS = 60
GB_MAX_DEPTH = 2
CHAOS_NONLINEARITY_COMPLEXITY_FEATURES = (
    "nl_gain",
    "lyapunov",
    "zero_one_K",
    "perm_entropy",
    "sample_entropy",
    "ext_lz_complexity",
    "ext_recurrence_rate",
    "ext_recurrence_determinism",
)
PERSISTENCE_NONSTATIONARITY_SPECTRAL_FEATURES = (
    "ac_timescale",
    "ami_first_min",
    "dfa_alpha",
    "hurst_rs",
    "adf_p",
    "kpss_p",
    "n_diffs",
    "ext_trend_strength",
    "ext_changepoint_count",
    "ext_psd_slope",
    "spectral_entropy",
    "dom_freq",
    "spectral_flatness",
    "ext_spectral_centroid",
    "ext_volatility_ac1",
    "ext_arch_lm5",
    "snr_db",
)


def run_paper_robustness_suite(
    *,
    discovery_table: Path,
    validation_table: Path,
    out_dir: Path,
    calibration_config: Path | None = None,
    thresholds: tuple[float, ...] = (0.0, 0.025, 0.05, 0.1),
    seed: int = 0,
    metric_subset_n: int = 120,
    mechanism_rows: int = 60,
    mechanism_seeds: int = 1,
    real_probes: bool = True,
    formats: tuple[str, ...] = ("png",),
) -> dict[str, Any]:
    """Run paper-facing robustness analyses and write deterministic artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    discovery = _load_frontier(discovery_table)
    validation = _load_frontier(validation_table)
    rng = np.random.default_rng(seed)

    feature_metrics = feature_family_ablation(discovery, validation, threshold=0.05, seed=seed)
    threshold_metrics = threshold_robustness(discovery, validation, thresholds=thresholds, seed=seed)
    quantile_enrichment = feature_quantile_enrichment(validation)
    regime_enrichment = regime_enrichment_table(validation, rng=rng)

    feature_metrics.to_csv(out_dir / "feature_family_ablation.csv", index=False)
    threshold_metrics.to_csv(out_dir / "threshold_robustness.csv", index=False)
    quantile_enrichment.to_csv(out_dir / "feature_quantile_enrichment.csv", index=False)
    regime_enrichment.to_csv(out_dir / "regime_enrichment.csv", index=False)

    real_probe_df = pd.DataFrame()
    if real_probes:
        real_probe_df = score_real_world_probes(discovery, out_dir=out_dir, seed=seed)
        real_probe_df.to_csv(out_dir / "real_world_probe_predictions.csv", index=False)

    metric_df = pd.DataFrame()
    metric_summary = pd.DataFrame()
    if metric_subset_n > 0 and calibration_config is not None:
        metric_df, metric_summary = metric_and_nvar_subset(
            validation,
            out_dir=out_dir,
            calibration_config=calibration_config,
            max_rows=metric_subset_n,
            seed=seed,
        )
        metric_df.to_csv(out_dir / "metric_nmae_nvar_subset.csv", index=False)
        metric_summary.to_csv(out_dir / "metric_nmae_nvar_summary.csv", index=False)

    mechanism_df = pd.DataFrame()
    mechanism_summary = pd.DataFrame()
    if mechanism_rows > 0 and calibration_config is not None:
        mechanism_df, mechanism_summary = mechanism_guardrail_subset(
            validation,
            out_dir=out_dir,
            calibration_config=calibration_config,
            max_rows=mechanism_rows,
            seeds=mechanism_seeds,
            seed=seed,
        )
        mechanism_df.to_csv(out_dir / "mechanism_guardrail_subset.csv", index=False)
        mechanism_summary.to_csv(out_dir / "mechanism_guardrail_summary.csv", index=False)

    figures = _write_figures(
        out_dir=out_dir,
        feature_metrics=feature_metrics,
        threshold_metrics=threshold_metrics,
        regime_enrichment=regime_enrichment,
        real_probe_df=real_probe_df,
        metric_summary=metric_summary,
        mechanism_summary=mechanism_summary,
        formats=formats,
    )
    report = _write_report(
        out_dir=out_dir,
        feature_metrics=feature_metrics,
        threshold_metrics=threshold_metrics,
        regime_enrichment=regime_enrichment,
        real_probe_df=real_probe_df,
        metric_summary=metric_summary,
        mechanism_summary=mechanism_summary,
    )
    manifest = {
        "analysis_version": "paper-robustness-v1",
        "discovery_table": str(discovery_table),
        "validation_table": str(validation_table),
        "calibration_config": str(calibration_config) if calibration_config is not None else None,
        "n_discovery": int(len(discovery)),
        "n_validation": int(len(validation)),
        "thresholds": [float(v) for v in thresholds],
        "metric_subset_n_requested": int(metric_subset_n),
        "metric_subset_n_written": int(len(metric_df)),
        "mechanism_rows_requested": int(mechanism_rows),
        "mechanism_rows_written": int(len(mechanism_df)),
        "real_probes_written": int(len(real_probe_df)),
        "claim_boundary": (
            "This suite stress-tests a protocol-local regime-map claim. "
            "Real probes are interpolation/OOD diagnostics, subset NMAE/NVAR checks are robustness checks, "
            "and mechanism guardrails are not evidence for fundamental quantum advantage."
        ),
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()) + ["paper_robustness_manifest.json"],
        "figures": figures,
        "report": report.name,
    }
    (out_dir / "paper_robustness_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def feature_family_ablation(discovery: pd.DataFrame, validation: pd.DataFrame, *, threshold: float, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, description, features in _feature_sets():
        rows.append(_prospective_metrics(discovery, validation, features=features, threshold=threshold, seed=seed, feature_set=name, description=description))
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def threshold_robustness(discovery: pd.DataFrame, validation: pd.DataFrame, *, thresholds: tuple[float, ...], seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for name, description, features in _feature_sets():
            if name not in {"full_30", "without_direct_predictability", "chaos_nonlinearity_complexity_only", "persistence_nonstationarity_spectral_volatility"}:
                continue
            row = _prospective_metrics(discovery, validation, features=features, threshold=threshold, seed=seed, feature_set=name, description=description)
            row["threshold"] = float(threshold)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["threshold", "roc_auc"], ascending=[True, False]).reset_index(drop=True)


def feature_quantile_enrichment(validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = {
        "persistence_ac_timescale": "ac_timescale",
        "persistence_dfa_alpha": "dfa_alpha",
        "nonstationary_trend_strength": "ext_trend_strength",
        "low_frequency_psd_slope": "ext_psd_slope",
        "spectral_entropy": "spectral_entropy",
        "volatility_memory": "ext_volatility_ac1",
        "complexity_perm_entropy": "perm_entropy",
        "chaos_lyapunov": "lyapunov",
        "nonlinearity_nl_gain": "nl_gain",
        "dominant_frequency": "dom_freq",
        "seasonality_strength": "ext_seasonality_strength",
    }
    for label, feature in features.items():
        if feature not in validation.columns:
            continue
        x = pd.to_numeric(validation[feature], errors="coerce")
        adv = pd.to_numeric(validation["qrc_advantage"], errors="coerce")
        mask = x.notna() & adv.notna()
        if int(mask.sum()) < 20:
            continue
        q20 = float(x[mask].quantile(0.2))
        q80 = float(x[mask].quantile(0.8))
        bins = {
            "low20": mask & (x <= q20),
            "mid60": mask & (x > q20) & (x < q80),
            "high20": mask & (x >= q80),
        }
        for bin_name, bin_mask in bins.items():
            sub = validation.loc[bin_mask]
            rows.append(
                {
                    "feature_group": label,
                    "feature": feature,
                    "bin": bin_name,
                    "n": int(len(sub)),
                    "q20": q20,
                    "q80": q80,
                    "qrc_useful_rate": _safe_mean(pd.to_numeric(sub["qrc_advantage"], errors="coerce") >= 0.05),
                    "qrc_win_rate": _safe_mean(pd.to_numeric(sub["qrc_advantage"], errors="coerce") > 0.0),
                    "mean_advantage": _safe_mean(pd.to_numeric(sub["qrc_advantage"], errors="coerce")),
                    "median_advantage": _safe_median(pd.to_numeric(sub["qrc_advantage"], errors="coerce")),
                }
            )
    return pd.DataFrame(rows)


def regime_enrichment_table(validation: pd.DataFrame, *, rng: np.random.Generator, n_bootstraps: int = 1000) -> pd.DataFrame:
    df = validation.copy()
    df["qrc_advantage"] = pd.to_numeric(df["qrc_advantage"], errors="coerce")
    useful = df["qrc_advantage"] >= 0.05
    masks = _regime_masks(df)
    masks["overall"] = pd.Series(True, index=df.index)
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        sub = df.loc[mask.fillna(False)]
        if sub.empty:
            continue
        adv = pd.to_numeric(sub["qrc_advantage"], errors="coerce").dropna().to_numpy(dtype=float)
        use = (adv >= 0.05).astype(float)
        win = (adv > 0.0).astype(float)
        mean_ci = _bootstrap_ci(adv, rng=rng, n_bootstraps=n_bootstraps)
        useful_ci = _bootstrap_ci(use, rng=rng, n_bootstraps=n_bootstraps)
        rows.append(
            {
                "regime": name,
                "n": int(adv.size),
                "qrc_useful_rate": _safe_mean(use),
                "qrc_useful_ci_low": useful_ci[0],
                "qrc_useful_ci_high": useful_ci[1],
                "qrc_win_rate": _safe_mean(win),
                "mean_advantage": _safe_mean(adv),
                "mean_ci_low": mean_ci[0],
                "mean_ci_high": mean_ci[1],
                "median_advantage": _safe_median(adv),
                "enrichment_vs_overall": _safe_mean(use) / max(float(useful.mean()), 1e-12),
            }
        )
    return pd.DataFrame(rows).sort_values("qrc_useful_rate", ascending=False).reset_index(drop=True)


def score_real_world_probes(discovery: pd.DataFrame, *, out_dir: Path, seed: int = 0) -> pd.DataFrame:
    probe_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for spec in _real_probe_sources():
        try:
            series = spec["loader"]()
            row = _profile_real_series(
                np.asarray(series, dtype=float),
                name=spec["name"],
                family=spec["family"],
                url=spec["url"],
                max_points=int(spec.get("max_points", 4000)),
            )
            row["probe_description"] = spec["description"]
            probe_rows.append(row)
        except Exception as exc:  # pragma: no cover - network data can disappear
            errors.append({"name": str(spec["name"]), "url": str(spec["url"]), "error": repr(exc)})
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "real_world_probe_fetch_errors.csv", index=False)
    if not probe_rows:
        return pd.DataFrame(errors)
    probes = materialize_frontier_features(pd.DataFrame(probe_rows))
    probes["qrc_advantage"] = np.nan

    X_train, X_probe, _features_used = _feature_matrices(discovery, probes, tuple(FRONTIER_TIER_A_FIELDS))
    y_train = pd.to_numeric(discovery["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    reg = _gb_reg(seed).fit(X_train, y_train)
    probes["predicted_qrc_advantage"] = reg.predict(X_probe)
    y_train_bin = y_train >= 0.05
    if len(np.unique(y_train_bin)) == 2:
        clf = _gb_clf(seed).fit(X_train, y_train_bin)
        probes["predicted_prob_qrc_useful"] = clf.predict_proba(X_probe)[:, 1]
    else:
        probes["predicted_prob_qrc_useful"] = np.nan
    support = compute_support_scores(discovery, probes, k_values=(15, 30, 50))
    for col in ("support_score", "ood_flag", "family_entropy", "nearest_family_mixture"):
        probes[col] = support[col].to_numpy()
    probes["prediction_claim_boundary"] = "external interpolation/OOD probe; not a real-world performance label"
    keep = [
        "name",
        "family",
        "probe_description",
        "length",
        "source_url",
        "predicted_qrc_advantage",
        "predicted_prob_qrc_useful",
        "support_score",
        "ood_flag",
        "nearest_family_mixture",
        "ac_timescale",
        "dfa_alpha",
        "ext_trend_strength",
        "ext_psd_slope",
        "spectral_entropy",
        "ext_volatility_ac1",
        "nl_gain",
        "lyapunov",
        "prediction_claim_boundary",
    ]
    return probes[[c for c in keep if c in probes.columns]].sort_values("predicted_qrc_advantage", ascending=False).reset_index(drop=True)


def metric_and_nvar_subset(
    validation: pd.DataFrame,
    *,
    out_dir: Path,
    calibration_config: Path,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = _load_calibration_config(calibration_config)
    if frozen is None:
        raise ValueError("calibration_config is required")
    qrc_cfg = _qrc_from_calibration_config(frozen)
    esn_grid = _esn_grid_from_calibration_config(frozen)
    selected = _select_property_subset(validation, max_rows=max_rows, seed=seed)
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        spec = spec_from_catalog_row(row)
        ds = generate(spec)
        if ds.ground_truth.get("_unavailable"):
            continue
        qrc = qrc_scores_standard(ds, qrc_cfg, seed=0)
        esn = esn_sparse_baseline(ds, qrc_cfg=qrc_cfg, seed=0, esn_grid=esn_grid, return_details=True)
        nvar = nvar_baseline_scores(ds, lag=20, degree=2)
        rows.append(
            {
                "dataset_id": row.get("dataset_id"),
                "name": row.get("name"),
                "family": row.get("family"),
                "subset_group": row.get("subset_group"),
                "original_qrc_advantage_nrmse": _float_or_nan(row.get("qrc_advantage")),
                "nrmse_qrc_spin_rerun": qrc["test_nrmse"],
                "nrmse_esn_frozen_rerun": esn["nrmse"],
                "nrmse_advantage_rerun": esn["nrmse"] - qrc["test_nrmse"],
                "nmae_qrc_spin": qrc["test_nmae"],
                "nmae_esn_frozen": esn["nmae"],
                "nmae_advantage": esn["nmae"] - qrc["test_nmae"],
                "nrmse_nvar": nvar["test_nrmse"],
                "nmae_nvar": nvar["test_nmae"],
                "nrmse_qrc_vs_nvar_advantage": nvar["test_nrmse"] - qrc["test_nrmse"],
                "nmae_qrc_vs_nvar_advantage": nvar["test_nmae"] - qrc["test_nmae"],
            }
        )
    df = pd.DataFrame(rows)
    summary = _metric_subset_summary(df)
    return df, summary


def mechanism_guardrail_subset(
    validation: pd.DataFrame,
    *,
    out_dir: Path,
    calibration_config: Path,
    max_rows: int,
    seeds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = _load_calibration_config(calibration_config)
    if frozen is None:
        raise ValueError("calibration_config is required")
    qrc_j1 = _qrc_from_calibration_config(frozen)
    qrc_j0 = replace(qrc_j1, J=0.0)
    qrc_diss = replace(qrc_j1, amplitude_damping=0.02, dephasing=0.01, dissipation_method="trajectory")
    selected = _select_property_subset(validation, max_rows=max_rows, seed=seed)
    seed_values = tuple(range(max(1, int(seeds))))
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        spec = spec_from_catalog_row(row)
        ds = generate(spec)
        if ds.ground_truth.get("_unavailable"):
            continue
        j1 = [qrc_scores_standard(ds, qrc_j1, seed=s) for s in seed_values]
        j0 = [qrc_scores_standard(ds, qrc_j0, seed=s) for s in seed_values]
        diss = [qrc_scores_standard(ds, qrc_diss, seed=s) for s in seed_values]
        j1_nrmse = _mean_dict(j1, "test_nrmse")
        j0_nrmse = _mean_dict(j0, "test_nrmse")
        diss_nrmse = _mean_dict(diss, "test_nrmse")
        rows.append(
            {
                "dataset_id": row.get("dataset_id"),
                "name": row.get("name"),
                "family": row.get("family"),
                "subset_group": row.get("subset_group"),
                "original_qrc_advantage_nrmse": _float_or_nan(row.get("qrc_advantage")),
                "nrmse_qrc_J1": j1_nrmse,
                "nrmse_qrc_J0": j0_nrmse,
                "nrmse_qrc_J1_dissipative_exploratory": diss_nrmse,
                "paired_delta_J0_minus_J1": j0_nrmse - j1_nrmse,
                "paired_delta_diss_minus_nodiss": diss_nrmse - j1_nrmse,
                "nmae_qrc_J1": _mean_dict(j1, "test_nmae"),
                "nmae_qrc_J0": _mean_dict(j0, "test_nmae"),
                "nmae_qrc_J1_dissipative_exploratory": _mean_dict(diss, "test_nmae"),
                "feature_dim": int(qrc_j1.feature_dim),
                "seeds": ",".join(str(s) for s in seed_values),
            }
        )
    df = pd.DataFrame(rows)
    summary = _mechanism_summary(df)
    return df, summary


def nvar_baseline_scores(ds: Dataset, *, lag: int = 20, degree: int = 2) -> dict[str, float]:
    """Polynomial nonlinear vector autoregression baseline with ridge readout."""

    u, y = build_task(ds)
    X = _lagged_design(u, lag=lag)
    n = min(X.shape[0], y.size)
    splits = _protocol_splits(n)
    X = X[:n]
    y = y[:n]
    Xs, _, _ = _standardize_train(X, splits.train)
    pieces = [Xs]
    if int(degree) >= 2:
        ii, jj = np.triu_indices(Xs.shape[1])
        pieces.append(Xs[:, ii] * Xs[:, jj])
    design = np.hstack(pieces)
    return _ridge_scores(design, y, splits)


def _prospective_metrics(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    features: tuple[str, ...],
    threshold: float,
    seed: int,
    feature_set: str,
    description: str,
) -> dict[str, Any]:
    features = tuple(c for c in features if c in discovery.columns and c in validation.columns)
    X_train, X_val, features_used = _feature_matrices(discovery, validation, features)
    y_train = pd.to_numeric(discovery["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    y_val = pd.to_numeric(validation["qrc_advantage"], errors="coerce").to_numpy(dtype=float)
    if X_train.size == 0 or X_val.size == 0:
        return {
            "feature_set": feature_set,
            "description": description,
            "n_features_declared": len(features),
            "n_features_used": 0,
            "regression_r2": np.nan,
            "regression_mae": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "brier": np.nan,
        }
    reg = _gb_reg(seed).fit(X_train, y_train)
    pred = reg.predict(X_val)
    y_train_bin = y_train >= float(threshold)
    y_val_bin = y_val >= float(threshold)
    prob = np.full(len(y_val), np.nan)
    roc = ap = brier = np.nan
    if len(np.unique(y_train_bin)) == 2 and len(np.unique(y_val_bin)) == 2:
        clf = _gb_clf(seed).fit(X_train, y_train_bin)
        prob = clf.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val_bin, prob)
        ap = average_precision_score(y_val_bin, prob)
        brier = brier_score_loss(y_val_bin, prob)
    return {
        "feature_set": feature_set,
        "description": description,
        "threshold": float(threshold),
        "n_features_declared": int(len(features)),
        "n_features_used": int(len(features_used)),
        "validation_positive_rate": _safe_mean(y_val_bin.astype(float)),
        "regression_r2": _finite_or_nan(r2_score(y_val, pred) if np.var(y_val) > 1e-12 else np.nan),
        "regression_mae": _finite_or_nan(mean_absolute_error(y_val, pred)),
        "roc_auc": _finite_or_nan(roc),
        "pr_auc": _finite_or_nan(ap),
        "brier": _finite_or_nan(brier),
        "mean_predicted_advantage": _safe_mean(pred),
        "features_used": ",".join(features_used),
    }


def _feature_matrices(train: pd.DataFrame, target: pd.DataFrame, features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = tuple(c for c in features if c in train.columns and c in target.columns)
    if not features:
        return np.empty((len(train), 0)), np.empty((len(target), 0)), []
    X_train = train.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_target = target.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = X_train.median(numeric_only=True).fillna(0.0)
    X_train = X_train.fillna(med).fillna(0.0)
    X_target = X_target.fillna(med).fillna(0.0)
    keep = [c for c in X_train.columns if float(X_train[c].std()) > 1e-12]
    if not keep:
        return np.empty((len(train), 0)), np.empty((len(target), 0)), []
    scaler = StandardScaler().fit(X_train[keep].to_numpy(dtype=float))
    return scaler.transform(X_train[keep].to_numpy(dtype=float)), scaler.transform(X_target[keep].to_numpy(dtype=float)), keep


def _gb_reg(seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(n_estimators=GB_N_ESTIMATORS, max_depth=GB_MAX_DEPTH, learning_rate=0.06, random_state=seed)


def _gb_clf(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=GB_N_ESTIMATORS, max_depth=GB_MAX_DEPTH, learning_rate=0.06, random_state=seed)


def _feature_sets() -> list[tuple[str, str, tuple[str, ...]]]:
    full = tuple(FRONTIER_TIER_A_FIELDS)
    chaos = tuple(c for c in CHAOS_NONLINEARITY_COMPLEXITY_FEATURES if c in full)
    direct = tuple(c for c in DIRECT_PREDICTABILITY_FEATURES if c in full)
    return [
        ("full_30", "All pre-declared Tier-A time-series descriptors.", full),
        (
            "without_direct_predictability",
            "All Tier-A descriptors except cheap direct predictability proxies.",
            tuple(c for c in full if c not in set(direct)),
        ),
        (
            "without_chaos_nonlinearity_complexity",
            "All Tier-A descriptors except chaos, nonlinearity, and entropy/complexity descriptors.",
            tuple(c for c in full if c not in set(chaos)),
        ),
        (
            "persistence_nonstationarity_spectral_volatility",
            "Memory, stationarity, spectral, noise, and volatility descriptors only.",
            tuple(c for c in PERSISTENCE_NONSTATIONARITY_SPECTRAL_FEATURES if c in full),
        ),
        (
            "chaos_nonlinearity_complexity_only",
            "Chaos, nonlinearity, and entropy/complexity descriptors only.",
            chaos,
        ),
        (
            "direct_predictability_only",
            "Cheap direct predictability probes only.",
            direct,
        ),
    ]


def _regime_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    q = lambda col, p: pd.to_numeric(df[col], errors="coerce").quantile(p) if col in df.columns else np.nan
    high_persistence = (pd.to_numeric(df.get("dfa_alpha"), errors="coerce") >= q("dfa_alpha", 0.75)) | (
        pd.to_numeric(df.get("ac_timescale"), errors="coerce") >= q("ac_timescale", 0.75)
    )
    drift = (pd.to_numeric(df.get("ext_trend_strength"), errors="coerce") >= q("ext_trend_strength", 0.75)) | (
        pd.to_numeric(df.get("n_diffs"), errors="coerce") >= 1
    )
    low_frequency = (pd.to_numeric(df.get("dom_freq"), errors="coerce") <= q("dom_freq", 0.25)) | (
        pd.to_numeric(df.get("ext_psd_slope"), errors="coerce") <= q("ext_psd_slope", 0.25)
    )
    pe = pd.to_numeric(df.get("perm_entropy"), errors="coerce")
    se = pd.to_numeric(df.get("spectral_entropy"), errors="coerce")
    moderate_complexity = pe.between(pe.quantile(0.25), pe.quantile(0.75)) & se.between(se.quantile(0.25), se.quantile(0.75))
    clean_periodic = (
        (pd.to_numeric(df.get("spectral_entropy"), errors="coerce") <= q("spectral_entropy", 0.25))
        & (pd.to_numeric(df.get("ext_seasonality_strength"), errors="coerce") >= q("ext_seasonality_strength", 0.75))
    ) | (df.get("family", pd.Series(index=df.index, dtype=object)).astype(str) == "oscillatory_quasiperiodic")
    chaotic_family = df.get("family", pd.Series(index=df.index, dtype=object)).astype(str).isin(["chaotic_map", "chaotic_flow"])
    explicit_nonlinear = df.get("family", pd.Series(index=df.index, dtype=object)).astype(str).isin(
        ["chaotic_map", "chaotic_flow", "nonlinear_autoregressive", "delay_dynamics", "input_driven_memory"]
    )
    return {
        "high_persistence": high_persistence,
        "drift_or_nonstationary": drift,
        "low_frequency_or_steep_spectrum": low_frequency,
        "moderate_complexity": moderate_complexity,
        "persistence_and_drift": high_persistence & drift,
        "persistence_and_low_frequency": high_persistence & low_frequency,
        "persistence_drift_low_frequency": high_persistence & drift & low_frequency,
        "persistence_drift_low_frequency_moderate_complexity": high_persistence & drift & low_frequency & moderate_complexity,
        "clean_periodic_control": clean_periodic,
        "chaotic_family_control": chaotic_family,
        "explicit_nonlinear_family_control": explicit_nonlinear,
    }


def _select_property_subset(validation: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0:
        return validation.head(0).copy()
    rng = np.random.default_rng(seed)
    masks = _regime_masks(validation)
    groups = [
        ("target_pocket", masks["persistence_drift_low_frequency_moderate_complexity"]),
        ("chaotic_control", masks["chaotic_family_control"]),
        ("periodic_control", masks["clean_periodic_control"]),
        ("background_control", ~(masks["persistence_drift_low_frequency_moderate_complexity"] | masks["chaotic_family_control"] | masks["clean_periodic_control"])),
    ]
    per_group = max(1, int(math.ceil(max_rows / len(groups))))
    selected: list[pd.DataFrame] = []
    used: set[Any] = set()
    for group_name, mask in groups:
        sub = validation.loc[mask.fillna(False)].copy()
        if sub.empty:
            continue
        take = min(per_group, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        piece = validation.loc[idx].copy()
        piece["subset_group"] = group_name
        selected.append(piece)
        used.update(idx.tolist())
    out = pd.concat(selected, ignore_index=False) if selected else validation.head(0).copy()
    if len(out) < max_rows:
        remaining = validation.loc[[idx for idx in validation.index if idx not in used]].copy()
        if not remaining.empty:
            take = min(max_rows - len(out), len(remaining))
            idx = rng.choice(remaining.index.to_numpy(), size=take, replace=False)
            piece = validation.loc[idx].copy()
            piece["subset_group"] = "fill_control"
            out = pd.concat([out, piece], ignore_index=False)
    return out.head(max_rows).reset_index(drop=True)


def _profile_real_series(series: np.ndarray, *, name: str, family: str, url: str, max_points: int) -> dict[str, Any]:
    x = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if x.size > max_points:
        x = x[-max_points:]
    spec = DatasetSpec(
        name=name,
        family=family,
        source="real",
        task_type="forecast",
        params={"source_url": url, "probe_max_points": int(max_points)},
        seed=0,
        length=int(x.size),
        horizon=1,
    )
    ds = Dataset(spec, x)
    rec = profile_dataset(ds)
    row = {**rec.to_row(), **compute_backstop(x), "base_generator": "external_real_probe", "source_url": url}
    return row


def _real_probe_sources() -> list[dict[str, Any]]:
    return [
        {
            "name": "silso_monthly_sunspots",
            "family": "real_solar_activity_probe",
            "description": "SILSO monthly total sunspot number.",
            "url": "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv",
            "loader": lambda: _load_silso_sunspots("https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"),
            "max_points": 4000,
        },
        {
            "name": "pjme_hourly_load",
            "family": "real_electricity_load_probe",
            "description": "PJM East hourly electricity load.",
            "url": "https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv",
            "loader": lambda: _load_csv_series(
                "https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv",
                "PJME_MW",
            ),
            "max_points": 4000,
        },
        {
            "name": "melbourne_daily_min_temperature",
            "family": "real_weather_probe",
            "description": "Daily minimum temperatures in Melbourne.",
            "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
            "loader": lambda: _load_csv_series("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", "Temp"),
            "max_points": 4000,
        },
        {
            "name": "fred_usd_eur_exchange",
            "family": "real_exchange_rate_probe",
            "description": "FRED daily USD/EUR exchange rate.",
            "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU",
            "loader": lambda: _load_csv_series("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU", "DEXUSEU"),
            "max_points": 4000,
        },
        {
            "name": "noaa_mauna_loa_co2",
            "family": "real_environment_probe",
            "description": "NOAA monthly Mauna Loa CO2.",
            "url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
            "loader": lambda: _load_noaa_co2("https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"),
            "max_points": 4000,
        },
        {
            "name": "ett_hourly_transformer_temperature",
            "family": "real_electricity_transformer_probe",
            "description": "ETTh1 transformer oil temperature benchmark.",
            "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
            "loader": lambda: _load_csv_series("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv", "OT"),
            "max_points": 4000,
        },
        {
            "name": "airline_passengers_monthly",
            "family": "real_transport_probe",
            "description": "Classic monthly international airline passengers.",
            "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
            "loader": lambda: _load_csv_series("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", "Passengers"),
            "max_points": 4000,
        },
    ]


def _url_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "qrc-dataset-profiler/0.1"})
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _load_csv_series(url: str, column: str) -> np.ndarray:
    df = pd.read_csv(io.StringIO(_url_text(url)))
    values = pd.to_numeric(df[column], errors="coerce")
    return values.dropna().to_numpy(dtype=float)


def _load_silso_sunspots(url: str) -> np.ndarray:
    cols = ["year", "month", "date_frac", "sunspot_number", "std", "n_obs", "definitive"]
    df = pd.read_csv(io.StringIO(_url_text(url)), sep=";", header=None, names=cols)
    values = pd.to_numeric(df["sunspot_number"], errors="coerce")
    return values[values >= 0].dropna().to_numpy(dtype=float)


def _load_noaa_co2(url: str) -> np.ndarray:
    df = pd.read_csv(io.StringIO(_url_text(url)), comment="#")
    candidates = ["average", "co2", "monthly average"]
    for col in candidates:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            return values[values > 0].dropna().to_numpy(dtype=float)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    for col in numeric.columns:
        values = numeric[col].dropna()
        if len(values) > 100 and float(values.mean()) > 100:
            return values.to_numpy(dtype=float)
    raise ValueError("could not identify NOAA CO2 value column")


def _metric_subset_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group, sub in _iter_subset_groups(df):
        rows.append(
            {
                "subset_group": group,
                "n": int(len(sub)),
                "mean_original_nrmse_advantage": _safe_mean(sub["original_qrc_advantage_nrmse"]),
                "mean_nrmse_advantage_rerun": _safe_mean(sub["nrmse_advantage_rerun"]),
                "mean_nmae_advantage": _safe_mean(sub["nmae_advantage"]),
                "nrmse_useful_rate": _safe_mean(pd.to_numeric(sub["nrmse_advantage_rerun"], errors="coerce") >= 0.05),
                "nmae_useful_rate": _safe_mean(pd.to_numeric(sub["nmae_advantage"], errors="coerce") >= 0.05),
                "qrc_beats_nvar_nrmse_rate": _safe_mean(pd.to_numeric(sub["nrmse_qrc_vs_nvar_advantage"], errors="coerce") > 0.0),
                "qrc_beats_nvar_nmae_rate": _safe_mean(pd.to_numeric(sub["nmae_qrc_vs_nvar_advantage"], errors="coerce") > 0.0),
                "corr_nrmse_vs_nmae_advantage": _safe_corr(sub["nrmse_advantage_rerun"], sub["nmae_advantage"]),
            }
        )
    return pd.DataFrame(rows)


def _mechanism_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group, sub in _iter_subset_groups(df):
        delta = pd.to_numeric(sub["paired_delta_J0_minus_J1"], errors="coerce")
        diss = pd.to_numeric(sub["paired_delta_diss_minus_nodiss"], errors="coerce")
        rows.append(
            {
                "subset_group": group,
                "n": int(len(sub)),
                "mean_delta_J0_minus_J1": _safe_mean(delta),
                "median_delta_J0_minus_J1": _safe_median(delta),
                "frac_J1_better_than_J0": _safe_mean(delta > 0.0),
                "mean_delta_diss_minus_nodiss": _safe_mean(diss),
                "frac_dissipative_better_than_nodiss": _safe_mean(diss < 0.0),
                "claim_boundary": "subset guardrail only; not a mechanism proof",
            }
        )
    return pd.DataFrame(rows)


def _iter_subset_groups(df: pd.DataFrame):
    yield "overall", df
    if "subset_group" in df.columns:
        for group, sub in df.groupby("subset_group", sort=True):
            yield str(group), sub


def _write_figures(
    *,
    out_dir: Path,
    feature_metrics: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    regime_enrichment: pd.DataFrame,
    real_probe_df: pd.DataFrame,
    metric_summary: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    formats: tuple[str, ...],
) -> list[dict[str, str]]:
    figures: list[dict[str, str]] = []
    for fmt in formats:
        figures.append(_plot_feature_ablation(feature_metrics, out_dir / f"robustness_01_feature_family_ablation.{fmt}"))
        figures.append(_plot_threshold_robustness(threshold_metrics, out_dir / f"robustness_02_threshold_robustness.{fmt}"))
        figures.append(_plot_regime_enrichment(regime_enrichment, out_dir / f"robustness_03_regime_enrichment.{fmt}"))
        if not real_probe_df.empty:
            figures.append(_plot_real_probes(real_probe_df, out_dir / f"robustness_04_real_world_probes.{fmt}"))
        if not metric_summary.empty:
            figures.append(_plot_metric_summary(metric_summary, out_dir / f"robustness_05_metric_nmae_nvar.{fmt}"))
        if not mechanism_summary.empty:
            figures.append(_plot_mechanism_summary(mechanism_summary, out_dir / f"robustness_06_mechanism_guardrail.{fmt}"))
    return figures


def _plot_feature_ablation(df: pd.DataFrame, path: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    data = df.sort_values("roc_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    y = np.arange(len(data))
    ax.barh(y, data["roc_auc"], color="#4c6f7f", label="ROC-AUC")
    ax.scatter(data["pr_auc"], y, color="#9a4f4f", label="PR-AUC", zorder=3)
    ax.axvline(0.5, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["feature_set"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Prospective validation score")
    ax.set_title("Feature-family ablation")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {"path": str(path), "description": "Feature-family ablation for the QRC-useful classifier."}


def _plot_threshold_robustness(df: pd.DataFrame, path: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    for name, sub in df.groupby("feature_set", sort=False):
        ax.plot(sub["threshold"], sub["roc_auc"], marker="o", linewidth=1.3, label=name)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Useful threshold")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Threshold robustness")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {"path": str(path), "description": "Meta-model classification robustness across useful thresholds."}


def _plot_regime_enrichment(df: pd.DataFrame, path: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    data = df[df["regime"] != "overall"].sort_values("qrc_useful_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    y = np.arange(len(data))
    ax.barh(y, data["qrc_useful_rate"], color="#4c6f7f")
    ax.axvline(float(df.loc[df["regime"] == "overall", "qrc_useful_rate"].iloc[0]), color="#222222", linewidth=0.9, linestyle="--", label="overall")
    ax.set_yticks(y)
    ax.set_yticklabels(data["regime"])
    ax.set_xlabel("QRC-useful rate")
    ax.set_title("Property-regime enrichment")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {"path": str(path), "description": "Enrichment of QRC-useful rows in property-defined regimes."}


def _plot_real_probes(df: pd.DataFrame, path: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    data = df.sort_values("predicted_qrc_advantage", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    y = np.arange(len(data))
    colors = ["#9a4f4f" if bool(v) else "#2f6f6f" for v in data["ood_flag"]]
    ax.barh(y, data["predicted_qrc_advantage"], color=colors)
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["name"])
    ax.set_xlabel("Predicted QRC advantage")
    ax.set_title("Real-world probes with OOD guardrail")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {"path": str(path), "description": "External real-world probe predictions and support/OOD flags."}


def _plot_metric_summary(df: pd.DataFrame, path: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    data = df.sort_values("subset_group", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    x = np.arange(len(data))
    width = 0.35
    ax.bar(x - width / 2, data["mean_nrmse_advantage_rerun"], width, label="NRMSE", color="#4c6f7f")
    ax.bar(x + width / 2, data["mean_nmae_advantage"], width, label="NMAE", color="#7b5e2f")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(data["subset_group"], rotation=25, ha="right")
    ax.set_ylabel("Mean ESN - QRC advantage")
    ax.set_title("Metric robustness subset")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {"path": str(path), "description": "Subset rerun comparing NRMSE and standardized NMAE advantage."}


def _plot_mechanism_summary(df: pd.DataFrame, path: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    data = df.sort_values("subset_group", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    y = np.arange(len(data))
    ax.barh(y, data["mean_delta_J0_minus_J1"], color="#4c6f7f")
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["subset_group"])
    ax.set_xlabel("Mean NRMSE(J=0) - NRMSE(J=J*)")
    ax.set_title("Paired coupling guardrail subset")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {"path": str(path), "description": "Paired J=0 vs calibrated-coupling QRC guardrail on property-defined subset."}


def _write_report(
    *,
    out_dir: Path,
    feature_metrics: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    regime_enrichment: pd.DataFrame,
    real_probe_df: pd.DataFrame,
    metric_summary: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
) -> Path:
    best = feature_metrics.sort_values("roc_auc", ascending=False).iloc[0]
    chaos = feature_metrics[feature_metrics["feature_set"] == "chaos_nonlinearity_complexity_only"].iloc[0]
    no_chaos = feature_metrics[feature_metrics["feature_set"] == "without_chaos_nonlinearity_complexity"].iloc[0]
    no_proxy = feature_metrics[feature_metrics["feature_set"] == "without_direct_predictability"].iloc[0]
    enriched = regime_enrichment[regime_enrichment["regime"] == "persistence_drift_low_frequency_moderate_complexity"]
    enriched_rate = float(enriched["qrc_useful_rate"].iloc[0]) if not enriched.empty else math.nan
    overall = regime_enrichment[regime_enrichment["regime"] == "overall"].iloc[0]
    lines = [
        "# Paper Robustness Report",
        "",
        "## Core Robustness",
        "",
        f"- Best prospective feature set: `{best['feature_set']}` ROC-AUC `{float(best['roc_auc']):.3f}`, PR-AUC `{float(best['pr_auc']):.3f}`.",
        f"- Chaos/nonlinearity/complexity-only: ROC-AUC `{float(chaos['roc_auc']):.3f}`, PR-AUC `{float(chaos['pr_auc']):.3f}`.",
        f"- Without chaos/nonlinearity/complexity: ROC-AUC `{float(no_chaos['roc_auc']):.3f}`, PR-AUC `{float(no_chaos['pr_auc']):.3f}`.",
        f"- Without direct predictability proxies: ROC-AUC `{float(no_proxy['roc_auc']):.3f}`, PR-AUC `{float(no_proxy['pr_auc']):.3f}`.",
        "",
        "## Regime Enrichment",
        "",
        f"- Overall QRC-useful rate: `{float(overall['qrc_useful_rate']):.3f}`.",
        f"- Persistence + drift + low-frequency + moderate-complexity pocket useful rate: `{enriched_rate:.3f}`.",
        "",
        "## Real-World Probes",
        "",
    ]
    if real_probe_df.empty:
        lines.append("- No real-world probe predictions were written.")
    else:
        for _, row in real_probe_df.iterrows():
            lines.append(
                f"- `{row['name']}`: predicted advantage `{float(row['predicted_qrc_advantage']):.3f}`, "
                f"P(useful) `{float(row['predicted_prob_qrc_useful']):.3f}`, support `{float(row['support_score']):.3f}`, OOD `{bool(row['ood_flag'])}`."
            )
    lines.extend(["", "## Metric Robustness", ""])
    if metric_summary.empty:
        lines.append("- Metric subset rerun was not requested or produced no rows.")
    else:
        row = metric_summary[metric_summary["subset_group"] == "overall"].iloc[0]
        lines.append(
            f"- Subset NRMSE mean advantage `{float(row['mean_nrmse_advantage_rerun']):.3f}`; "
            f"NMAE mean advantage `{float(row['mean_nmae_advantage']):.3f}`; "
            f"QRC beats NVAR by NRMSE rate `{float(row['qrc_beats_nvar_nrmse_rate']):.3f}`."
        )
    lines.extend(["", "## Mechanism Guardrail", ""])
    if mechanism_summary.empty:
        lines.append("- Mechanism guardrail subset was not requested or produced no rows.")
    else:
        row = mechanism_summary[mechanism_summary["subset_group"] == "overall"].iloc[0]
        lines.append(
            f"- Mean paired delta J0-J*: `{float(row['mean_delta_J0_minus_J1']):.3f}`; "
            f"fraction J* better than J0 `{float(row['frac_J1_better_than_J0']):.3f}`."
        )
    lines.extend(
        [
            "",
            "Claim boundary: these artifacts support a protocol-local regime-map and robustness claim. They do not establish broad QRC superiority, hardware quantum advantage, or an entanglement mechanism.",
            "",
        ]
    )
    path = out_dir / "PAPER_ROBUSTNESS_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load_frontier(path: Path) -> pd.DataFrame:
    df = materialize_frontier_features(load_catalog(path))
    df["qrc_advantage"] = pd.to_numeric(df["qrc_advantage"], errors="coerce")
    return df[np.isfinite(df["qrc_advantage"].to_numpy(dtype=float))].reset_index(drop=True)


def _bootstrap_ci(values: np.ndarray, *, rng: np.random.Generator, n_bootstraps: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, arr.size, size=(int(n_bootstraps), arr.size))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _mean_dict(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _safe_mean(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def _safe_median(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def _safe_corr(a: Any, b: Any) -> float:
    x = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if math.isfinite(out) else np.nan


def _finite_or_nan(value: Any) -> float:
    out = _float_or_nan(value)
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
