"""Explanatory meta-model for QRC advantage.

The meta-model stays deliberately small and interpretable: it only uses the
frozen core axis fields, guards every small-data path, and reports feature
directions against the original QRC advantage target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from qrc_dataset_profiler.spec import CORE_AXIS_FIELDS


CORRELATION_CLUSTER_THRESHOLD = 0.9
NEAR_CONSTANT_STD = 1e-12


@dataclass
class MetaModelResult:
    """Container returned by :func:`fit_meta_model`.

    The class also supports dict-like access for lightweight downstream use in
    scripts and tests.
    """

    features_used: list[str]
    n_samples: int
    regression_cv: dict[str, Any]
    classification_cv: dict[str, Any]
    ranked_importances: pd.DataFrame
    estimators: dict[str, Any]
    collinearity_mapping: dict[str, str]
    collinearity_clusters: dict[str, list[str]]
    qrc_win_drivers: list[dict[str, Any]]
    preprocessing: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    X: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), repr=False)
    y: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "features_used": self.features_used,
            "n_samples": self.n_samples,
            "regression_cv": self.regression_cv,
            "classification_cv": self.classification_cv,
            "ranked_importances": self.ranked_importances,
            "estimators": self.estimators,
            "collinearity_mapping": self.collinearity_mapping,
            "collinearity_clusters": self.collinearity_clusters,
            "qrc_win_drivers": self.qrc_win_drivers,
            "preprocessing": self.preprocessing,
            "notes": self.notes,
        }


def fit_meta_model(
    catalog_df: pd.DataFrame,
    *,
    target: str = "qrc_advantage",
    win_threshold: float = 0.05,
    seed: int = 0,
    feature_fields: tuple[str, ...] | list[str] | None = None,
) -> MetaModelResult:
    """Fit interpretable models that explain QRC advantage from dataset properties.

    Parameters
    ----------
    catalog_df:
        Catalog with one row per dataset and Block A-E fields.
    target:
        Regression target. By default this is ``qrc_advantage``.
    win_threshold:
        Classification threshold for "QRC wins": ``target > win_threshold``.
    seed:
        Deterministic seed used for shuffled CV and ensemble estimators.
    """

    notes: list[str] = []
    empty_importances = _empty_importance_frame()
    if target not in catalog_df.columns:
        return MetaModelResult(
            features_used=[],
            n_samples=0,
            regression_cv={"note": f"missing target column: {target}"},
            classification_cv={"note": f"missing target column: {target}"},
            ranked_importances=empty_importances,
            estimators={},
            collinearity_mapping={},
            collinearity_clusters={},
            qrc_win_drivers=[],
            notes=[f"missing target column: {target}"],
        )

    y_all = pd.to_numeric(catalog_df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_y = y_all.notna().to_numpy()
    n_target = int(valid_y.sum())
    if n_target < 20:
        notes.append("insufficient samples: n<20")

    requested_features = tuple(feature_fields) if feature_fields is not None else CORE_AXIS_FIELDS
    candidate_features = [c for c in requested_features if c in catalog_df.columns]
    X_df = catalog_df.loc[valid_y, candidate_features].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    y = y_all.loc[valid_y].to_numpy(dtype=float)

    prep = _prepare_features(X_df, y)
    notes.extend(prep["notes"])
    X = prep["X"]
    features = prep["features_used"]

    if len(features) == 0 or len(y) == 0:
        note = "no usable features after preprocessing"
        return MetaModelResult(
            features_used=features,
            n_samples=len(y),
            regression_cv={"note": note},
            classification_cv={"note": note},
            ranked_importances=empty_importances,
            estimators={},
            collinearity_mapping=prep["collinearity_mapping"],
            collinearity_clusters=prep["collinearity_clusters"],
            qrc_win_drivers=[],
            preprocessing=prep["preprocessing"],
            notes=notes + [note],
            X=X,
            y=y,
        )

    reg_models = {
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
        "ridge": Ridge(alpha=1.0),
    }
    regression_cv = _evaluate_regression_cv(X, y, reg_models, seed=seed)

    fitted_estimators: dict[str, Any] = {}
    for name, model in reg_models.items():
        try:
            fitted_estimators[f"regression_{name}"] = clone(model).fit(X, y)
        except Exception as exc:  # pragma: no cover - defensive small-data path
            fitted_estimators[f"regression_{name}"] = None
            notes.append(f"failed to fit regression_{name}: {exc}")

    importances = _regression_importances(
        X,
        y,
        features,
        model=GradientBoostingRegressor(random_state=seed),
        seed=seed,
        raw_feature_frame=prep["raw_reduced_imputed"],
    )
    importances = _attach_linear_coefficients(importances, fitted_estimators.get("regression_ridge"), features)
    importances = _attach_shap_importances(importances, fitted_estimators.get("regression_gradient_boosting"), X, notes)
    importances = importances.sort_values(["importance_mean", "ridge_abs_coef"], ascending=[False, False]).reset_index(drop=True)

    y_bin = y > float(win_threshold)
    clf_models = {
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "logistic": LogisticRegression(max_iter=1000),
    }
    classification_cv = _evaluate_classification_cv(X, y_bin, clf_models, seed=seed)
    if len(np.unique(y_bin)) == 2:
        for name, model in clf_models.items():
            try:
                fitted_estimators[f"classification_{name}"] = clone(model).fit(X, y_bin)
            except Exception as exc:  # pragma: no cover - defensive small-data path
                fitted_estimators[f"classification_{name}"] = None
                notes.append(f"failed to fit classification_{name}: {exc}")
    else:
        for name in clf_models:
            fitted_estimators[f"classification_{name}"] = None

    qrc_win_drivers = _qrc_win_drivers(importances)

    return MetaModelResult(
        features_used=features,
        n_samples=len(y),
        regression_cv=regression_cv,
        classification_cv=classification_cv,
        ranked_importances=importances,
        estimators=fitted_estimators,
        collinearity_mapping=prep["collinearity_mapping"],
        collinearity_clusters=prep["collinearity_clusters"],
        qrc_win_drivers=qrc_win_drivers,
        preprocessing=prep["preprocessing"],
        notes=notes,
        X=X,
        y=y,
    )


def _prepare_features(X_df: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    notes: list[str] = []
    if X_df.empty:
        return {
            "X": np.empty((len(y), 0)),
            "features_used": [],
            "raw_reduced_imputed": pd.DataFrame(index=X_df.index),
            "collinearity_mapping": {},
            "collinearity_clusters": {},
            "preprocessing": {"medians": {}, "dropped_all_nan": [], "dropped_constant": []},
            "notes": ["no CORE_AXIS_FIELDS present"],
        }

    dropped_all_nan = [c for c in X_df.columns if X_df[c].isna().all()]
    X_non_nan = X_df.drop(columns=dropped_all_nan)
    dropped_constant: list[str] = []
    for col in X_non_nan.columns:
        vals = X_non_nan[col].dropna().to_numpy(dtype=float)
        if vals.size == 0 or np.nanstd(vals) <= NEAR_CONSTANT_STD or pd.Series(vals).nunique(dropna=True) <= 1:
            dropped_constant.append(col)
    X_variable = X_non_nan.drop(columns=dropped_constant)
    if dropped_all_nan:
        notes.append(f"dropped all-NaN features: {', '.join(dropped_all_nan)}")
    if dropped_constant:
        notes.append(f"dropped near-constant features: {', '.join(dropped_constant)}")

    if X_variable.empty:
        return {
            "X": np.empty((len(y), 0)),
            "features_used": [],
            "raw_reduced_imputed": pd.DataFrame(index=X_df.index),
            "collinearity_mapping": {},
            "collinearity_clusters": {},
            "preprocessing": {
                "medians": {},
                "dropped_all_nan": dropped_all_nan,
                "dropped_constant": dropped_constant,
            },
            "notes": notes,
        }

    medians = X_variable.median(numeric_only=True).to_dict()
    X_imputed = X_variable.fillna(medians)
    clusters = _correlation_clusters(X_imputed, threshold=CORRELATION_CLUSTER_THRESHOLD)
    representatives = [_choose_representative(cluster, X_variable, X_imputed.columns) for cluster in clusters]
    feature_mapping = {member: rep for rep, cluster in zip(representatives, clusters) for member in cluster}
    cluster_dict = {rep: list(cluster) for rep, cluster in zip(representatives, clusters)}
    X_reduced = X_imputed[representatives]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced.to_numpy(dtype=float))
    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(-1, 1)

    return {
        "X": X_scaled,
        "features_used": list(representatives),
        "raw_reduced_imputed": X_reduced,
        "collinearity_mapping": feature_mapping,
        "collinearity_clusters": cluster_dict,
        "preprocessing": {
            "medians": {k: float(v) for k, v in medians.items()},
            "dropped_all_nan": dropped_all_nan,
            "dropped_constant": dropped_constant,
            "scaler_mean": dict(zip(representatives, scaler.mean_.astype(float))),
            "scaler_scale": dict(zip(representatives, scaler.scale_.astype(float))),
            "correlation_cluster_threshold": CORRELATION_CLUSTER_THRESHOLD,
        },
        "notes": notes,
    }


def _correlation_clusters(X: pd.DataFrame, *, threshold: float) -> list[list[str]]:
    cols = list(X.columns)
    if len(cols) <= 1:
        return [cols] if cols else []
    corr = X.corr().abs().fillna(0.0)
    visited: set[str] = set()
    clusters: list[list[str]] = []
    for col in cols:
        if col in visited:
            continue
        stack = [col]
        cluster: list[str] = []
        visited.add(col)
        while stack:
            current = stack.pop()
            cluster.append(current)
            neighbors = [c for c in cols if c not in visited and float(corr.loc[current, c]) >= threshold]
            for neighbor in neighbors:
                visited.add(neighbor)
                stack.append(neighbor)
        clusters.append([c for c in cols if c in set(cluster)])
    return clusters


def _choose_representative(cluster: list[str], X_with_nan: pd.DataFrame, column_order: pd.Index) -> str:
    order = {col: idx for idx, col in enumerate(column_order)}
    return sorted(cluster, key=lambda c: (float(X_with_nan[c].isna().mean()), order[c]))[0]


def _evaluate_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    models: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"models": {}}
    k = min(5, len(y) // 4)
    if len(y) < 8 or k < 2 or X.shape[1] == 0:
        result["note"] = "insufficient samples for regression CV"
        for name in models:
            result["models"][name] = {"r2": [], "mae": [], "r2_mean": np.nan, "mae_mean": np.nan}
        return result

    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    for name, model in models.items():
        r2_vals: list[float] = []
        mae_vals: list[float] = []
        for train_idx, test_idx in splitter.split(X):
            try:
                fitted = clone(model).fit(X[train_idx], y[train_idx])
                pred = fitted.predict(X[test_idx])
                r2_vals.append(float(r2_score(y[test_idx], pred)) if np.var(y[test_idx]) > NEAR_CONSTANT_STD else np.nan)
                mae_vals.append(float(mean_absolute_error(y[test_idx], pred)))
            except Exception:
                r2_vals.append(np.nan)
                mae_vals.append(np.nan)
        result["models"][name] = {
            "r2": r2_vals,
            "mae": mae_vals,
            "r2_mean": _finite_mean(r2_vals),
            "mae_mean": _finite_mean(mae_vals),
            "n_splits": k,
        }
    result["r2_mean"] = result["models"]["gradient_boosting"]["r2_mean"]
    result["mae_mean"] = result["models"]["gradient_boosting"]["mae_mean"]
    result["n_splits"] = k
    return result


def _evaluate_classification_cv(
    X: np.ndarray,
    y_bin: np.ndarray,
    models: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"models": {}, "class_balance": _class_balance(y_bin)}
    classes, counts = np.unique(y_bin, return_counts=True)
    if len(classes) < 2:
        result["note"] = "only one class for QRC-win target"
        for name in models:
            result["models"][name] = {"roc_auc": [], "average_precision": [], "brier": [], "roc_auc_mean": np.nan, "average_precision_mean": np.nan, "brier_mean": np.nan}
        return result
    k = min(5, int(counts.min()))
    if k < 2 or X.shape[1] == 0:
        result["note"] = "insufficient class counts for classification CV"
        for name in models:
            result["models"][name] = {"roc_auc": [], "average_precision": [], "brier": [], "roc_auc_mean": np.nan, "average_precision_mean": np.nan, "brier_mean": np.nan}
        return result

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for name, model in models.items():
        auc_vals: list[float] = []
        ap_vals: list[float] = []
        brier_vals: list[float] = []
        for train_idx, test_idx in splitter.split(X, y_bin):
            if len(np.unique(y_bin[train_idx])) < 2 or len(np.unique(y_bin[test_idx])) < 2:
                auc_vals.append(np.nan)
                ap_vals.append(np.nan)
                brier_vals.append(np.nan)
                continue
            try:
                fitted = clone(model).fit(X[train_idx], y_bin[train_idx])
                if hasattr(fitted, "predict_proba"):
                    score = fitted.predict_proba(X[test_idx])[:, 1]
                else:
                    score = fitted.decision_function(X[test_idx])
                prob = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
                auc_vals.append(float(roc_auc_score(y_bin[test_idx], score)))
                ap_vals.append(float(average_precision_score(y_bin[test_idx], score)))
                brier_vals.append(float(brier_score_loss(y_bin[test_idx], prob)))
            except Exception:
                auc_vals.append(np.nan)
                ap_vals.append(np.nan)
                brier_vals.append(np.nan)
        result["models"][name] = {
            "roc_auc": auc_vals,
            "average_precision": ap_vals,
            "brier": brier_vals,
            "roc_auc_mean": _finite_mean(auc_vals),
            "average_precision_mean": _finite_mean(ap_vals),
            "brier_mean": _finite_mean(brier_vals),
            "n_splits": k,
        }
    result["roc_auc_mean"] = result["models"]["gradient_boosting"]["roc_auc_mean"]
    result["average_precision_mean"] = result["models"]["gradient_boosting"]["average_precision_mean"]
    result["brier_mean"] = result["models"]["gradient_boosting"]["brier_mean"]
    result["n_splits"] = k
    return result


def _regression_importances(
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    *,
    model: Any,
    seed: int,
    raw_feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    if len(y) < 8 or X.shape[1] == 0:
        rows = []
        for i, feature in enumerate(features):
            rows.append(_importance_row(feature, 0.0, 0.0, raw_feature_frame.iloc[:, i].to_numpy(), y))
        return pd.DataFrame(rows, columns=_importance_columns())

    k = min(5, len(y) // 4)
    importances: list[np.ndarray] = []
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        try:
            fitted = clone(model).fit(X[train_idx], y[train_idx])
            scoring = "r2" if np.var(y[test_idx]) > NEAR_CONSTANT_STD else "neg_mean_absolute_error"
            perm = permutation_importance(
                fitted,
                X[test_idx],
                y[test_idx],
                scoring=scoring,
                n_repeats=20,
                random_state=seed + fold,
            )
            importances.append(perm.importances_mean)
        except Exception:
            continue
    if importances:
        arr = np.vstack(importances)
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0)
    else:
        means = np.zeros(len(features), dtype=float)
        stds = np.zeros(len(features), dtype=float)

    rows = [
        _importance_row(feature, means[i], stds[i], raw_feature_frame[feature].to_numpy(dtype=float), y)
        for i, feature in enumerate(features)
    ]
    return pd.DataFrame(rows, columns=_importance_columns())


def _importance_row(feature: str, mean: float, std: float, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    corr = _safe_corr(x, y)
    if np.isnan(corr) or abs(corr) <= 1e-12:
        direction = "flat"
    elif corr > 0:
        direction = "positive"
    else:
        direction = "negative"
    return {
        "feature": feature,
        "importance_mean": float(mean),
        "importance_std": float(std),
        "corr_with_advantage": corr,
        "direction": direction,
    }


def _attach_linear_coefficients(df: pd.DataFrame, estimator: Any, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    coefs = np.full(len(features), np.nan)
    if estimator is not None and hasattr(estimator, "coef_"):
        coefs = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        if len(coefs) != len(features):
            coefs = np.full(len(features), np.nan)
    df["ridge_coef"] = coefs
    df["ridge_abs_coef"] = np.abs(coefs)
    return df


def _attach_shap_importances(df: pd.DataFrame, estimator: Any, X: np.ndarray, notes: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["shap_mean_abs"] = np.nan
    if estimator is None or X.size == 0:
        return df
    try:
        import shap  # type: ignore
    except Exception:
        notes.append("SHAP not available; skipped SHAP importances")
        return df
    try:  # pragma: no cover - shap is optional and absent in the target environment
        explainer = shap.Explainer(estimator)
        values = explainer(X)
        arr = np.asarray(values.values)
        if arr.ndim == 2 and arr.shape[1] == len(df):
            df["shap_mean_abs"] = np.mean(np.abs(arr), axis=0)
    except Exception as exc:
        notes.append(f"SHAP failed; skipped SHAP importances: {exc}")
    return df


def _qrc_win_drivers(importances: pd.DataFrame) -> list[dict[str, Any]]:
    if importances.empty:
        return []
    positive = importances[importances["direction"] == "positive"].head(10)
    return [
        {
            "feature": str(row.feature),
            "importance_mean": float(row.importance_mean),
            "corr_with_advantage": float(row.corr_with_advantage),
            "direction": str(row.direction),
        }
        for row in positive.itertuples(index=False)
    ]


def _finite_mean(vals: list[float] | np.ndarray) -> float:
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else np.nan


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2 or np.nanstd(x) <= NEAR_CONSTANT_STD or np.nanstd(y) <= NEAR_CONSTANT_STD:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _class_balance(y_bin: np.ndarray) -> dict[str, int]:
    return {"negative": int((~y_bin).sum()), "positive": int(y_bin.sum())}


def _importance_columns() -> list[str]:
    return ["feature", "importance_mean", "importance_std", "corr_with_advantage", "direction"]


def _empty_importance_frame() -> pd.DataFrame:
    cols = _importance_columns() + ["ridge_coef", "ridge_abs_coef", "shap_mean_abs"]
    return pd.DataFrame(columns=cols)
