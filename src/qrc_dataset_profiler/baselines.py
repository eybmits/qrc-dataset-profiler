"""Classical baselines and QRC readout evaluation for Block E targets."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from qrc_dataset_profiler.reservoir import StandardSpinV1
from qrc_dataset_profiler.spec import Dataset


FloatArray = NDArray[np.float64]


class Splits(NamedTuple):
    train: slice
    val: slice
    test: slice


_ALPHAS = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def build_task(ds: Dataset, *, horizon: int | None = None) -> tuple[FloatArray, FloatArray]:
    """Return scalar drive u and target y under the protocol task definition."""

    series = _finite_fill(np.asarray(ds.series, dtype=np.float64).reshape(-1))
    if ds.spec.task_type == "forecast":
        h = max(1, int(ds.spec.horizon if horizon is None else horizon))
        if series.size <= h:
            raise ValueError(f"{ds.spec.name} is too short for horizon={h} forecasting")
        return series[:-h], series[h:]
    if ds.spec.task_type == "input_driven":
        if ds.inputs is None:
            raise ValueError(f"{ds.spec.name} is input_driven but has no inputs")
        inputs = _finite_fill(np.asarray(ds.inputs, dtype=np.float64).reshape(-1))
        n = min(inputs.size, series.size)
        if n < 2:
            raise ValueError(f"{ds.spec.name} is too short for input-driven evaluation")
        return inputs[:n], series[:n]
    raise ValueError(f"unknown task_type {ds.spec.task_type!r}")


def evaluate_readout(features: np.ndarray, y: np.ndarray, splits: Splits | None = None) -> float:
    """Fit a Ridge readout, select alpha on validation, and return test NRMSE."""

    return _ridge_scores(features, y, splits)["test_nrmse"]


def linear_baseline(series: Any, inputs: np.ndarray | None = None, *, lag: int = 20, **kwargs: Any) -> float:
    """Ridge on a lagged scalar input window."""

    u, y = _task_from_args(series, inputs, kwargs.get("task_type"), kwargs.get("horizon"))
    X = _lagged_design(u, lag=lag)
    splits = _protocol_splits(min(X.shape[0], y.size))
    return evaluate_readout(X, y, splits)


def gbm_baseline(
    series: Any,
    inputs: np.ndarray | None = None,
    *,
    lag: int = 20,
    seed: int = 0,
    **kwargs: Any,
) -> float:
    """GradientBoostingRegressor on the same lagged scalar input window."""

    u, y = _task_from_args(series, inputs, kwargs.get("task_type"), kwargs.get("horizon"))
    X = _lagged_design(u, lag=lag)
    splits = _protocol_splits(min(X.shape[0], y.size))
    Xs, _, _ = _standardize_train(X, splits.train)
    yz, y_mu, y_sd = _standardize_vector_train(y, splits.train)
    train_val = _join_train_val_indices(splits)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=seed)
    model.fit(Xs[train_val], yz[train_val])
    pred = model.predict(Xs[splits.test]) * y_sd + y_mu
    return _nrmse(y[splits.test], pred)


def esn_matched_baseline(
    series: Any,
    inputs: np.ndarray | None = None,
    *,
    qrc_cfg: StandardSpinV1 | None = None,
    reservoir_size: int | None = None,
    seed: int = 0,
    esn_grid: dict[str, Sequence[float]] | None = None,
    return_details: bool = False,
    **kwargs: Any,
) -> float | dict[str, Any]:
    """Dimension-matched leaky ESN selected on validation NRMSE."""

    if esn_grid is None and "grid" in kwargs:
        esn_grid = kwargs["grid"]
    u, y = _task_from_args(series, inputs, kwargs.get("task_type"), kwargs.get("horizon"))
    size = int(reservoir_size if reservoir_size is not None else (qrc_cfg or StandardSpinV1()).feature_dim)
    if size < 1:
        raise ValueError("reservoir_size must be positive")
    splits = _protocol_splits(y.size)
    u_scaled, _, _ = _standardize_vector_train(u, splits.train)
    win_raw = _cycle_input(size, seed)
    w_raw = _cycle_reservoir(size)
    grid = _normalize_esn_grid(esn_grid)

    best: dict[str, Any] | None = None
    for rho in grid["rho"]:
        w = _scale_spectral_radius(w_raw, rho)
        for leak in grid["leak"]:
            for input_scale in grid["input_scale"]:
                states = _run_esn(u_scaled, w, win_raw * input_scale, leak)
                scores = _ridge_scores(states, y, splits)
                cand = {
                    "nrmse": scores["test_nrmse"],
                    "val_nrmse": scores["val_nrmse"],
                    "reservoir_size": size,
                    "washout": splits.train.start or 0,
                    "rho": rho,
                    "leak": leak,
                    "input_scale": input_scale,
                    "alpha": scores["alpha"],
                }
                if best is None or cand["val_nrmse"] < best["val_nrmse"]:
                    best = cand
    assert best is not None
    return best if return_details else float(best["nrmse"])


def _normalize_esn_grid(esn_grid: dict[str, Sequence[float]] | None) -> dict[str, tuple[float, ...]]:
    defaults = {
        "rho": (0.7, 0.9, 1.0, 1.1, 1.3),
        "leak": (0.1, 0.3, 0.6, 1.0),
        "input_scale": (0.3, 1.0, 2.0),
    }
    if esn_grid is None:
        return defaults
    out: dict[str, tuple[float, ...]] = {}
    for key, default in defaults.items():
        vals = tuple(float(v) for v in esn_grid.get(key, default))
        if not vals:
            raise ValueError(f"ESN grid entry {key!r} must not be empty")
        out[key] = vals
    return out


def qrc_nrmse(ds: Dataset, cfg: StandardSpinV1, seed: int = 0) -> float:
    """Run Standard-Spin v1 and score a Ridge readout."""

    u, y = build_task(ds)
    seeded_cfg = replace(cfg, seed=seed)
    features = seeded_cfg.transform(u)
    splits = _protocol_splits(min(features.shape[0], y.size))
    return evaluate_readout(features, y, splits)


def _task_from_args(
    series: Any,
    inputs: np.ndarray | None,
    task_type: str | None,
    horizon: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    if isinstance(series, Dataset):
        return build_task(series, horizon=horizon)
    y_raw = _finite_fill(np.asarray(series, dtype=np.float64).reshape(-1))
    if inputs is None and task_type != "input_driven":
        h = max(1, int(1 if horizon is None else horizon))
        if y_raw.size <= h:
            raise ValueError(f"series is too short for horizon={h} forecasting")
        return y_raw[:-h], y_raw[h:]
    if inputs is None:
        raise ValueError("inputs are required for input_driven raw-array evaluation")
    u_raw = _finite_fill(np.asarray(inputs, dtype=np.float64).reshape(-1))
    n = min(u_raw.size, y_raw.size)
    return u_raw[:n], y_raw[:n]


def _finite_fill(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if finite.all():
        return x.astype(np.float64, copy=False)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float64)
    idx = np.arange(x.size)
    return np.interp(idx, idx[finite], x[finite]).astype(np.float64)


def _protocol_splits(n: int) -> Splits:
    return _make_splits(n, washout=_default_washout(n))


def _default_washout(n: int) -> int:
    w = max(50, int(round(0.1 * n)))
    if n - w < 30:
        w = max(0, n - 30)
    return int(w)


def _make_splits(n: int, *, washout: int = 0) -> Splits:
    washout = max(0, int(washout))
    if n - washout < 30:
        raise ValueError("at least 30 aligned samples are required for train/val/test evaluation")
    n_eval = n - washout
    n_train = max(10, int(0.6 * n_eval))
    n_val = max(5, int(0.2 * n_eval))
    if n_train + n_val > n_eval - 5:
        n_train = max(10, n_eval - 10)
        n_val = 5
    return Splits(
        slice(washout, washout + n_train),
        slice(washout + n_train, washout + n_train + n_val),
        slice(washout + n_train + n_val, n),
    )


def _lagged_design(u: np.ndarray, lag: int) -> FloatArray:
    u = _finite_fill(np.asarray(u, dtype=np.float64).reshape(-1))
    n = u.size
    lag = min(max(1, int(lag)), max(1, n))
    X = np.empty((n, lag), dtype=np.float64)
    for t in range(n):
        start = max(0, t - lag + 1)
        hist = u[start : t + 1][::-1]
        if hist.size < lag:
            X[t, : hist.size] = hist
            X[t, hist.size :] = hist[-1] if hist.size else 0.0
        else:
            X[t] = hist
    return X


def _lagged_window(u: np.ndarray, y: np.ndarray, lag: int) -> tuple[FloatArray, FloatArray]:
    u = _finite_fill(np.asarray(u, dtype=np.float64).reshape(-1))
    y = _finite_fill(np.asarray(y, dtype=np.float64).reshape(-1))
    n = min(u.size, y.size)
    lag = min(max(1, int(lag)), max(1, n // 2))
    rows = n - lag + 1
    if rows < 30:
        raise ValueError("not enough samples for lagged baseline")
    X = np.empty((rows, lag), dtype=np.float64)
    yy = np.empty(rows, dtype=np.float64)
    for row, t in enumerate(range(lag - 1, n)):
        X[row] = u[t - lag + 1 : t + 1][::-1]
        yy[row] = y[t]
    return X, yy


def _standardize_train(X: np.ndarray, train: slice) -> tuple[FloatArray, FloatArray, FloatArray]:
    X = np.asarray(X, dtype=np.float64)
    mu = np.mean(X[train], axis=0)
    sd = np.std(X[train], axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return ((X - mu) / sd).astype(np.float64), mu.astype(np.float64), sd.astype(np.float64)


def _standardize_vector_train(y: np.ndarray, train: slice) -> tuple[FloatArray, float, float]:
    y = _finite_fill(np.asarray(y, dtype=np.float64).reshape(-1))
    mu = float(np.mean(y[train]))
    sd = float(np.std(y[train]))
    if sd < 1e-12:
        sd = 1.0
    return ((y - mu) / sd).astype(np.float64), mu, sd


def _ridge_scores(features: np.ndarray, y: np.ndarray, splits: Splits | None = None) -> dict[str, float]:
    X = np.asarray(features, dtype=np.float64)
    yy = _finite_fill(np.asarray(y, dtype=np.float64).reshape(-1))
    n = min(X.shape[0], yy.size)
    X = X[:n]
    yy = yy[:n]
    if splits is None:
        splits = _make_splits(n)
    Xs, _, _ = _standardize_train(X, splits.train)
    yz, y_mu, y_sd = _standardize_vector_train(yy, splits.train)

    best_alpha = float(_ALPHAS[0])
    best_val = np.inf
    for alpha in _ALPHAS:
        model = Ridge(alpha=float(alpha))
        model.fit(Xs[splits.train], yz[splits.train])
        pred_val = model.predict(Xs[splits.val]) * y_sd + y_mu
        val = _nrmse(yy[splits.val], pred_val)
        if val < best_val:
            best_val = val
            best_alpha = float(alpha)

    train_val = _join_train_val_indices(splits)
    model = Ridge(alpha=best_alpha)
    model.fit(Xs[train_val], yz[train_val])
    pred_test = model.predict(Xs[splits.test]) * y_sd + y_mu
    return {"test_nrmse": _nrmse(yy[splits.test], pred_test), "val_nrmse": float(best_val), "alpha": best_alpha}


def _join_train_val_indices(splits: Splits) -> np.ndarray:
    return np.arange(splits.train.start or 0, splits.val.stop, dtype=int)


# NRMSE >= ~2 means the readout is divergent/anti-predictive (e.g. extrapolation on a
# nonstationary holdout). Cap it so a single blown-up baseline (chirp ESN hit 25.8)
# cannot create a giant spurious qrc_advantage outlier that would dominate the meta-model.
NRMSE_CAP = 2.0


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    scale = float(np.std(y_true))
    nrmse = rmse / max(scale, 1e-12)
    if not np.isfinite(nrmse):
        return NRMSE_CAP
    return float(min(nrmse, NRMSE_CAP))


def _scale_spectral_radius(w: FloatArray, rho: float) -> FloatArray:
    eigvals = np.linalg.eigvals(w)
    radius = float(np.max(np.abs(eigvals)))
    if not np.isfinite(radius) or radius < 1e-12:
        return np.zeros_like(w)
    return (w * (float(rho) / radius)).astype(np.float64)


def _cycle_input(size: int, seed: int) -> FloatArray:
    win = np.zeros(int(size), dtype=np.float64)
    win[int(seed) % int(size)] = 1.0
    return win


def _cycle_reservoir(size: int) -> FloatArray:
    w = np.zeros((int(size), int(size)), dtype=np.float64)
    for i in range(int(size)):
        w[(i + 1) % int(size), i] = 1.0
    return w


def _run_esn(u: FloatArray, w: FloatArray, win: FloatArray, leak: float) -> FloatArray:
    states = np.empty((u.size, win.size), dtype=np.float64)
    x = np.zeros(win.size, dtype=np.float64)
    leak = float(leak)
    for t, u_t in enumerate(u):
        proposal = np.tanh(win * float(u_t) + w @ x)
        x = (1.0 - leak) * x + leak * proposal
        states[t] = x
    return states
