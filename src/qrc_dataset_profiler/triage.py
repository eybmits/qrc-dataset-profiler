"""User-facing QRC usefulness triage from a new time series.

The triage layer is deliberately downstream-only.  It never runs QRC or ESN on
the user dataset.  It computes the same dataset descriptors used in the atlas,
fits the lightweight meta-model from the frozen discovery table, and returns a
screening recommendation with an atlas-support guardrail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from qrc_dataset_profiler.analysis import load_catalog
from qrc_dataset_profiler.frontier import compute_support_scores, materialize_frontier_features
from qrc_dataset_profiler.properties import compute_backstop, profile_dataset
from qrc_dataset_profiler.spec import Dataset, DatasetSpec, FRONTIER_TIER_A_FIELDS


DEFAULT_DISCOVERY_TABLE = Path("results_frontier_v5_discovery/frontier_discovery_evaluated_v5_multi_qrc.csv")
DEFAULT_WIN_THRESHOLD = 0.05
DEFAULT_MAX_LENGTH = 800
QRC_TRIAGE_VERSION = "qrc-triage-v1"


@dataclass(frozen=True)
class TriageModel:
    """Fitted discovery-only meta-model used for user-facing triage."""

    features: tuple[str, ...]
    medians: dict[str, float]
    training_feature_values: dict[str, np.ndarray]
    regressor: GradientBoostingRegressor
    classifier: GradientBoostingClassifier | None
    class_prior: float
    win_threshold: float
    n_discovery: int
    target_column: str


def read_series_csv(
    path: Path | str,
    *,
    column: str | int | None = None,
    sep: str = ",",
    header: bool = True,
) -> np.ndarray:
    """Read one univariate series from a CSV/TSV-like file."""

    source = Path(path)
    df = pd.read_csv(source, sep=sep, header=0 if header else None)
    if df.empty:
        raise ValueError(f"{source} is empty")
    if column is None:
        if not header:
            column = 0
        else:
            numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]
            if len(numeric_cols) != 1:
                raise ValueError(f"please pass --column; numeric-looking columns are {numeric_cols}")
            column = numeric_cols[0]
    if isinstance(column, str) and header and column not in df.columns:
        raise ValueError(f"column {column!r} not found in {source}")
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise ValueError(f"column {column!r} has no finite numeric values")
    return values.astype(float, copy=False)


def window_series(series: np.ndarray, *, max_length: int = DEFAULT_MAX_LENGTH, window: str = "tail") -> np.ndarray:
    """Return the atlas-comparable analysis window."""

    x = np.asarray(series, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("series is empty")
    if window == "full" or max_length <= 0 or x.size <= max_length:
        return x
    if window == "head":
        return x[: int(max_length)]
    if window == "tail":
        return x[-int(max_length) :]
    raise ValueError("window must be 'tail', 'head', or 'full'")


def feature_row_from_series(
    series: np.ndarray,
    *,
    name: str = "user_series",
    horizon: int = 1,
    max_length: int = DEFAULT_MAX_LENGTH,
    window: str = "tail",
) -> pd.DataFrame:
    """Compute one 30-feature atlas row for a user-provided series."""

    raw = np.asarray(series, dtype=float).reshape(-1)
    used = window_series(raw, max_length=max_length, window=window)
    spec = DatasetSpec(
        name=name,
        family="user_provided",
        source="real",
        task_type="forecast",
        params={"triage_window": window, "raw_length": int(raw.size)},
        seed=0,
        length=int(used.size),
        horizon=max(1, int(horizon)),
    )
    ds = Dataset(spec, used)
    rec = profile_dataset(ds)
    row = {**rec.to_row(), **compute_backstop(used)}
    row["base_generator"] = "user_series"
    row["triage_raw_length"] = int(raw.size)
    row["triage_used_length"] = int(used.size)
    row["triage_window"] = window
    return materialize_frontier_features(pd.DataFrame([row]))


def fit_triage_model(
    discovery_table: pd.DataFrame,
    *,
    seed: int = 0,
    win_threshold: float = DEFAULT_WIN_THRESHOLD,
    target_column: str = "best_qrc_advantage_vs_esn",
) -> TriageModel:
    """Fit the lightweight discovery-only model used for triage."""

    df = materialize_frontier_features(discovery_table.copy())
    if target_column not in df.columns:
        if "qrc_advantage" in df.columns:
            target_column = "qrc_advantage"
        else:
            raise ValueError(f"discovery table lacks target column {target_column!r}")
    y = pd.to_numeric(df[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = y.notna().to_numpy()
    if int(valid.sum()) < 20:
        raise ValueError("need at least 20 discovery rows with finite QRC-advantage labels")
    y_arr = y.loc[valid].to_numpy(dtype=float)
    X_df = df.loc[valid, list(FRONTIER_TIER_A_FIELDS)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = X_df.median(numeric_only=True).fillna(0.0)
    X_filled = X_df.fillna(med)
    variable = [c for c in X_filled.columns if float(np.nanstd(X_filled[c].to_numpy(dtype=float))) > 1e-12]
    if not variable:
        raise ValueError("no variable Tier-A features available in discovery table")
    X = X_filled[variable].to_numpy(dtype=float)
    reg = GradientBoostingRegressor(random_state=seed).fit(X, y_arr)
    y_bin = y_arr >= float(win_threshold)
    class_prior = float(np.mean(y_bin))
    clf: GradientBoostingClassifier | None = None
    if len(np.unique(y_bin)) == 2:
        clf = GradientBoostingClassifier(random_state=seed).fit(X, y_bin)
    training_values = {c: X_filled[c].to_numpy(dtype=float) for c in variable}
    return TriageModel(
        features=tuple(variable),
        medians={str(k): float(v) for k, v in med.items()},
        training_feature_values=training_values,
        regressor=reg,
        classifier=clf,
        class_prior=class_prior,
        win_threshold=float(win_threshold),
        n_discovery=int(valid.sum()),
        target_column=target_column,
    )


def triage_feature_row(
    feature_row: pd.DataFrame,
    model: TriageModel,
    *,
    discovery_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Score one precomputed feature row with the fitted triage model."""

    row = materialize_frontier_features(feature_row.copy()).reset_index(drop=True)
    if len(row) != 1:
        raise ValueError("triage_feature_row expects exactly one row")
    X = _transform_row(row, model)
    predicted_advantage = float(model.regressor.predict(X)[0])
    if model.classifier is None:
        qrc_useful_probability = model.class_prior
    else:
        classes = list(model.classifier.classes_)
        positive_idx = classes.index(True) if True in classes else int(np.argmax(classes))
        qrc_useful_probability = float(model.classifier.predict_proba(X)[0, positive_idx])
    support = _support_payload(discovery_table, row) if discovery_table is not None else {}
    feature_payload = _key_feature_payload(row, model)
    rule_pocket = _slow_stateful_rule(feature_payload)
    recommendation = _recommendation(
        qrc_useful_probability=qrc_useful_probability,
        predicted_advantage=predicted_advantage,
        support_score=support.get("support_score"),
        rule_pocket=rule_pocket,
        win_threshold=model.win_threshold,
    )
    return {
        "triage_version": QRC_TRIAGE_VERSION,
        "dataset_id": str(row.loc[0, "dataset_id"]) if "dataset_id" in row else "user_series",
        "name": str(row.loc[0, "name"]) if "name" in row else "user_series",
        "prediction": {
            "predicted_best_qrc_advantage_vs_esn": predicted_advantage,
            "qrc_useful_probability": qrc_useful_probability,
            "useful_threshold": model.win_threshold,
            "recommendation": recommendation,
        },
        "support": support,
        "key_features": feature_payload,
        "slow_stateful_rule_match": bool(rule_pocket),
        "model": {
            "trained_on": "frozen discovery atlas",
            "n_discovery": model.n_discovery,
            "target_column": model.target_column,
            "features_used": list(model.features),
            "class_prior": model.class_prior,
        },
        "claim_boundary": (
            "This is a screening estimate from measured dataset properties. It does not prove a QRC win, "
            "does not run QRC/ESN on the submitted dataset, and should be treated as a triage signal only."
        ),
    }


def triage_series(
    series: np.ndarray,
    *,
    discovery_table: pd.DataFrame,
    name: str = "user_series",
    horizon: int = 1,
    max_length: int = DEFAULT_MAX_LENGTH,
    window: str = "tail",
    seed: int = 0,
    win_threshold: float = DEFAULT_WIN_THRESHOLD,
) -> dict[str, Any]:
    """Compute features and return a QRC-usefulness triage report."""

    features = feature_row_from_series(series, name=name, horizon=horizon, max_length=max_length, window=window)
    model = fit_triage_model(discovery_table, seed=seed, win_threshold=win_threshold)
    report = triage_feature_row(features, model, discovery_table=discovery_table)
    report["dataset"] = {
        "raw_length": int(np.asarray(series).reshape(-1).size),
        "used_length": int(features.loc[0, "triage_used_length"]) if "triage_used_length" in features else int(features.loc[0, "length"]),
        "window": window,
        "horizon": int(horizon),
    }
    return report


def load_discovery_table(path: Path | str = DEFAULT_DISCOVERY_TABLE) -> pd.DataFrame:
    """Load the frozen discovery table used by the triage CLI."""

    return load_catalog(Path(path))


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(_json_safe(report), indent=2) + "\n"


def report_to_text(report: dict[str, Any]) -> str:
    pred = report["prediction"]
    support = report.get("support", {})
    lines = [
        f"QRC triage for {report.get('name', 'user_series')}",
        f"recommendation: {pred['recommendation']}",
        f"predicted best-QRC advantage vs ESN: {pred['predicted_best_qrc_advantage_vs_esn']:+.3f} NRMSE",
        f"QRC-useful probability: {pred['qrc_useful_probability']:.3f}",
    ]
    if support:
        lines.append(f"atlas support score: {support.get('support_score', float('nan')):.3f} (OOD={support.get('ood_flag')})")
        if support.get("nearest_family_mixture"):
            lines.append(f"nearest atlas families: {support['nearest_family_mixture']}")
    if report.get("slow_stateful_rule_match"):
        lines.append("rule pocket: matches slow/stateful high-usefulness pocket")
    key = report.get("key_features", {})
    if key:
        lines.append("key descriptors:")
        for name, payload in key.items():
            lines.append(f"  - {name}: value={payload['value']:.4g}, atlas_percentile={payload['atlas_percentile']:.3f}")
    lines.append("boundary: " + report["claim_boundary"])
    return "\n".join(lines) + "\n"


def _transform_row(row: pd.DataFrame, model: TriageModel) -> np.ndarray:
    values = []
    for feature in model.features:
        raw = pd.to_numeric(row.get(feature, pd.Series([np.nan])), errors="coerce").iloc[0]
        if not np.isfinite(float(raw)):
            raw = model.medians.get(feature, 0.0)
        values.append(float(raw))
    return np.asarray(values, dtype=float).reshape(1, -1)


def _support_payload(discovery_table: pd.DataFrame | None, row: pd.DataFrame) -> dict[str, Any]:
    if discovery_table is None:
        return {}
    support = compute_support_scores(materialize_frontier_features(discovery_table.copy()), row, k_values=(15, 30, 50))
    if support.empty:
        return {}
    rec = support.iloc[0].to_dict()
    keep = (
        "support_score",
        "ood_flag",
        "raw_distance_percentile",
        "pca_distance_percentile",
        "family_entropy",
        "nearest_family_mixture",
    )
    return {k: _json_safe(rec[k]) for k in keep if k in rec}


def _key_feature_payload(row: pd.DataFrame, model: TriageModel) -> dict[str, dict[str, float]]:
    keys = (
        "ext_trend_strength",
        "ext_spectral_centroid",
        "dfa_alpha",
        "ac_timescale",
        "ext_volatility_ac1",
        "snr_db",
    )
    out: dict[str, dict[str, float]] = {}
    for feature in keys:
        if feature not in row.columns:
            continue
        value = float(pd.to_numeric(row[feature], errors="coerce").iloc[0])
        if not np.isfinite(value):
            continue
        ref = model.training_feature_values.get(feature)
        if ref is None or ref.size == 0:
            continue
        out[feature] = {
            "value": value,
            "atlas_percentile": float(np.mean(ref <= value)),
        }
    return out


def _slow_stateful_rule(key_features: dict[str, dict[str, float]]) -> bool:
    trend = key_features.get("ext_trend_strength", {}).get("atlas_percentile")
    centroid = key_features.get("ext_spectral_centroid", {}).get("atlas_percentile")
    if trend is None or centroid is None:
        return False
    return bool(trend > 0.92 and 0.03 < centroid <= 0.18)


def _recommendation(
    *,
    qrc_useful_probability: float,
    predicted_advantage: float,
    support_score: float | None,
    rule_pocket: bool,
    win_threshold: float,
) -> str:
    if support_score is not None and np.isfinite(float(support_score)) and float(support_score) < 0.05:
        return "outside_atlas_support__run_direct_benchmark"
    if qrc_useful_probability >= 0.50 and predicted_advantage >= win_threshold:
        return "high_priority_qrc_test"
    if rule_pocket and predicted_advantage > 0.0:
        return "high_priority_qrc_test"
    if qrc_useful_probability >= 0.25 or predicted_advantage > 0.0:
        return "worth_testing_qrc_if_available"
    return "esn_first_qrc_low_priority"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
