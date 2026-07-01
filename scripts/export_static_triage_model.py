"""Export the lightweight browser-side QRC triage model.

The public GitHub Pages site cannot run the full Python 30-feature triage stack.
This script trains a compact model on descriptors that are practical to compute
in browser JavaScript and writes a static JSON asset consumed by docs/assets/triage.js.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score


BROWSER_FEATURES = (
    "ac_timescale",
    "dfa_alpha",
    "spectral_entropy",
    "dom_freq",
    "spectral_flatness",
    "perm_entropy",
    "sample_entropy",
    "hurst_rs",
    "forecastability",
    "ext_volatility_ac1",
    "ext_arch_lm5",
    "ext_psd_slope",
    "ext_spectral_centroid",
    "ext_trend_strength",
    "ext_changepoint_count",
    "ext_lz_complexity",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export browser-side QRC triage model JSON.")
    parser.add_argument("--discovery", default="results_frontier_v5_discovery/frontier_discovery_evaluated_v5_multi_qrc.csv")
    parser.add_argument("--validation", default="results_frontier_v5_validation/frontier_validation_evaluated_v5_multi_qrc.csv")
    parser.add_argument("--out", default="docs/assets/triage_model.json")
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    discovery = pd.read_csv(args.discovery)
    validation = pd.read_csv(args.validation)
    target = "best_qrc_advantage_vs_esn"
    missing = [f for f in BROWSER_FEATURES if f not in discovery.columns or f not in validation.columns]
    if missing:
        raise SystemExit(f"missing browser features: {missing}")
    if target not in discovery.columns or target not in validation.columns:
        raise SystemExit(f"missing target column: {target}")

    X_train_raw = discovery.loc[:, BROWSER_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_val_raw = validation.loc[:, BROWSER_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_train = pd.to_numeric(discovery[target], errors="coerce").to_numpy(dtype=float)
    y_val = pd.to_numeric(validation[target], errors="coerce").to_numpy(dtype=float)
    good_train = np.isfinite(y_train)
    good_val = np.isfinite(y_val)
    X_train_raw = X_train_raw.loc[good_train].reset_index(drop=True)
    X_val_raw = X_val_raw.loc[good_val].reset_index(drop=True)
    y_train = y_train[good_train]
    y_val = y_val[good_val]
    y_train_bin = y_train >= float(args.threshold)
    y_val_bin = y_val >= float(args.threshold)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)

    classifier = GradientBoostingClassifier(
        random_state=0,
        n_estimators=180,
        max_depth=2,
        learning_rate=0.04,
    )
    classifier.fit(X_train, y_train_bin)
    regressor = GradientBoostingRegressor(
        random_state=0,
        n_estimators=180,
        max_depth=2,
        learning_rate=0.04,
    )
    regressor.fit(X_train, y_train)

    val_prob = classifier.predict_proba(X_val)[:, 1]
    val_pred = regressor.predict(X_val)
    validation_metrics = {
        "n_discovery": int(X_train.shape[0]),
        "n_validation": int(X_val.shape[0]),
        "qrc_useful_threshold": float(args.threshold),
        "validation_positive_rate": float(np.mean(y_val_bin)),
        "browser_subset_roc_auc": float(roc_auc_score(y_val_bin, val_prob)),
        "browser_subset_pr_auc": float(average_precision_score(y_val_bin, val_prob)),
        "browser_subset_regression_r2": float(r2_score(y_val, val_pred)),
        "browser_subset_regression_corr": float(np.corrcoef(y_val, val_pred)[0, 1]),
    }

    quantile_grid = np.linspace(0.0, 1.0, 101)
    quantiles = {
        feature: _finite_quantiles(X_train_raw[feature].to_numpy(dtype=float), quantile_grid)
        for feature in BROWSER_FEATURES
    }
    payload = {
        "version": "browser-triage-v1",
        "source": {
            "discovery_table": args.discovery,
            "validation_table": args.validation,
            "target": target,
            "description": (
                "Static browser-side screening model trained on the v5 discovery atlas. "
                "It uses only descriptors computed in JavaScript and is not the full Python 30-feature triage model."
            ),
        },
        "features": list(BROWSER_FEATURES),
        "imputer_medians": {f: float(v) for f, v in zip(BROWSER_FEATURES, imputer.statistics_)},
        "quantile_grid": [float(v) for v in quantile_grid],
        "feature_quantiles": quantiles,
        "validation_metrics": validation_metrics,
        "classifier": {
            "learning_rate": float(classifier.learning_rate),
            "init_log_odds": _classifier_init_log_odds(classifier),
            "trees": [_tree_to_payload(est[0].tree_) for est in classifier.estimators_],
        },
        "regressor": {
            "learning_rate": float(regressor.learning_rate),
            "init_value": _regressor_init_value(regressor),
            "trees": [_tree_to_payload(est[0].tree_) for est in regressor.estimators_],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")
    print(json.dumps(validation_metrics, indent=2, sort_keys=True))
    return 0


def _finite_quantiles(values: np.ndarray, grid: np.ndarray) -> list[float]:
    x = values[np.isfinite(values)]
    if x.size == 0:
        return [0.0 for _ in grid]
    return [float(v) for v in np.quantile(x, grid)]


def _classifier_init_log_odds(model: GradientBoostingClassifier) -> float:
    prior = getattr(model.init_, "class_prior_", None)
    if prior is not None and len(prior) == 2:
        p = float(prior[1])
    else:
        p = float(np.mean(model.classes_ == 1))
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return float(np.log(p / (1.0 - p)))


def _regressor_init_value(model: GradientBoostingRegressor) -> float:
    constant = getattr(model.init_, "constant_", None)
    if constant is not None:
        return float(np.asarray(constant).reshape(-1)[0])
    return 0.0


def _tree_to_payload(tree: Any) -> dict[str, list[float] | list[int]]:
    return {
        "children_left": [int(v) for v in tree.children_left],
        "children_right": [int(v) for v in tree.children_right],
        "feature": [int(v) for v in tree.feature],
        "threshold": [float(v) for v in tree.threshold],
        "value": [float(v) for v in tree.value.reshape(tree.node_count, -1)[:, 0]],
    }


if __name__ == "__main__":
    raise SystemExit(main())
