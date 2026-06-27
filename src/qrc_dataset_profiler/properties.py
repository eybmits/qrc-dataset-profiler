"""Core property estimators for schema v1."""

from __future__ import annotations

import math
import warnings
from collections import Counter
from typing import Any, Callable

import numpy as np
from scipy import signal, stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mutual_info_score, r2_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import LinearRegression, Ridge

from qrc_dataset_profiler.spec import (
    Dataset,
    DatasetRecord,
    MIN_LENGTH_DFA,
    MIN_LENGTH_LYAPUNOV,
    MIN_LENGTH_ZERO_ONE,
)


def profile_dataset(ds: Dataset) -> DatasetRecord:
    """Compute one schema-v1 catalog record for ``ds``."""

    spec = ds.spec
    y_raw = _as_1d(ds.series)
    missing_frac = float(np.mean(~np.isfinite(y_raw))) if y_raw.size else 1.0
    y_raw = _finite_fill(y_raw)
    if y_raw.size == 0:
        raise ValueError(f"{spec.name} has no available data")

    train_n = max(10, int(0.7 * y_raw.size))
    raw_train = y_raw[:train_n]
    y_train = _zscore(raw_train)
    if spec.task_type == "input_driven":
        if ds.inputs is None:
            raise ValueError(f"{spec.name} is input_driven but has no inputs")
        u_raw = _finite_fill(_as_1d(ds.inputs))[: y_raw.size]
        u_train = _zscore(u_raw[:train_n])
        source_train = u_train
        target_train = y_train
    else:
        source_train = y_train
        target_train = y_train

    rec = DatasetRecord(
        dataset_id=f"{spec.name}:{spec.seed}:{spec.length}",
        name=spec.name,
        family=spec.family,
        source=spec.source,
        task_type=spec.task_type,
        params=spec.params,
        seed=spec.seed,
        n_channels=spec.n_channels,
        length=int(y_raw.size),
        horizon=spec.horizon,
        missing_frac=missing_frac,
        irregular_sampling=False,
    )

    rec.mean = _safe("mean", lambda: float(np.mean(y_train)))
    rec.std = _safe("std", lambda: float(np.std(y_train)))
    rec.skew = _safe("skew", lambda: float(stats.skew(y_train, bias=False)))
    rec.kurtosis = _safe("kurtosis", lambda: float(stats.kurtosis(y_train, fisher=True, bias=False)))

    rec.ac_timescale = _safe("ac_timescale", lambda: _ac_timescale(source_train))
    rec.ami_first_min = _safe("ami_first_min", lambda: _ami_first_min(source_train))

    mem_value, mem_valid = _safe_pair("mem_capacity", lambda: _mem_capacity(u_train, target_train) if spec.task_type == "input_driven" else (math.nan, False))
    rec.mem_capacity = mem_value
    rec.mem_capacity_valid = mem_valid

    pred = _safe("predictive_models", lambda: _predictive_models(source_train, target_train, seed=spec.seed))
    if isinstance(pred, dict):
        rec.r2_linear = pred["r2_linear"]
        rec.nl_gain = pred["nl_gain"]
        rec.pred_nrmse_gbm = pred["pred_nrmse_gbm"]

    snr_value, snr_valid = _safe_pair("snr_db", lambda: _snr_db(source_train, target_train))
    rec.snr_db = snr_value
    rec.snr_valid = snr_valid

    lyap_value, lyap_valid = _safe_pair("lyapunov", lambda: _lyapunov_rosenstein(source_train, dt=float(spec.params.get("dt", 1.0))))
    rec.lyapunov = lyap_value
    rec.lyapunov_valid = lyap_valid

    z_value, z_valid = _safe_pair("zero_one_K", lambda: _zero_one_K(source_train, seed=spec.seed))
    rec.zero_one_K = z_value
    rec.zero_one_valid = z_valid

    freq = _safe("spectral", lambda: _spectral_features(source_train))
    if isinstance(freq, dict):
        rec.spectral_entropy = freq["spectral_entropy"]
        rec.dom_freq = freq["dom_freq"]
        rec.spectral_flatness = freq["spectral_flatness"]
        rec.forecastability = 1.0 - rec.spectral_entropy

    stat_features = _safe("stationarity", lambda: _stationarity_features(raw_train))
    if isinstance(stat_features, dict):
        rec.adf_p = stat_features["adf_p"]
        rec.kpss_p = stat_features["kpss_p"]
        rec.n_diffs = stat_features["n_diffs"]

    dfa_value, dfa_valid = _safe_pair("dfa_alpha", lambda: _dfa_alpha(source_train))
    rec.dfa_alpha = dfa_value
    rec.dfa_valid = dfa_valid

    rec.perm_entropy = _safe("perm_entropy", lambda: _perm_entropy(source_train))

    for key, value in ds.ground_truth.items():
        if hasattr(rec, key):
            setattr(rec, key, value)
    return rec


def compute_backstop(series: np.ndarray) -> dict[str, float]:
    """Return optional Tier-B features when optional packages exist."""

    out: dict[str, float] = {}
    x = _finite_fill(_as_1d(series))
    try:
        import pycatch22  # type: ignore

        values = pycatch22.catch22_all(x)
        names = values.get("names", [])
        vals = values.get("values", [])
        out.update({f"catch22_{name}": float(val) for name, val in zip(names, vals)})
    except Exception:
        pass
    try:
        from tsfresh.feature_extraction import feature_calculators as fc  # type: ignore

        out["ts_abs_energy"] = float(fc.abs_energy(x))
        out["ts_mean_abs_change"] = float(fc.mean_abs_change(x))
    except Exception:
        pass
    return out


def count_spectral_peaks(series: np.ndarray, max_peaks: int = 32) -> int:
    """Small helper used by validation output for frequency ground truth."""

    x = _zscore(_finite_fill(_as_1d(series)))
    if x.size < 16:
        return 0
    freqs, psd = signal.welch(x, nperseg=min(1024, x.size))
    if psd.size <= 3:
        return 0
    psd = psd.copy()
    psd[0] = 0.0
    peaks, props = signal.find_peaks(psd, prominence=max(float(np.max(psd)) * 0.03, 1e-12))
    if peaks.size == 0:
        return 0
    order = np.argsort(props["prominences"])[::-1]
    return int(min(max_peaks, order.size))


def _safe(name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        warnings.warn(f"{name} failed: {exc}", RuntimeWarning)
        return math.nan


def _safe_pair(name: str, fn: Callable[[], tuple[float, bool]]) -> tuple[float, bool]:
    try:
        value, valid = fn()
        return float(value), bool(valid)
    except Exception as exc:
        warnings.warn(f"{name} failed: {exc}", RuntimeWarning)
        return math.nan, False


def _as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _finite_fill(x: np.ndarray) -> np.ndarray:
    x = _as_1d(x)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if finite.all():
        return x.astype(float, copy=False)
    if not finite.any():
        return np.asarray([], dtype=float)
    idx = np.arange(x.size)
    return np.interp(idx, idx[finite], x[finite]).astype(float)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = _finite_fill(x)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-12:
        return x * 0.0
    return (x - mu) / sd


def _ac_timescale(x: np.ndarray) -> float:
    x = _zscore(x)
    n = x.size
    if n < 4:
        return math.nan
    max_lag = min(500, n // 3)
    corr = np.correlate(x, x, mode="full")[n - 1 : n + max_lag]
    corr = corr / max(abs(corr[0]), 1e-12)
    below = np.flatnonzero(np.abs(corr[1:]) < 1.0 / math.e)
    return float(below[0] + 1) if below.size else float(max_lag)


def _ami_first_min(x: np.ndarray) -> float:
    x = _finite_fill(x)
    n = x.size
    if n < 32:
        return math.nan
    max_lag = min(80, n // 5)
    bins = min(16, max(4, int(np.sqrt(n))))
    edges = np.histogram_bin_edges(x, bins=bins)
    disc = np.digitize(x, edges[1:-1])
    vals = np.array([mutual_info_score(disc[:-lag], disc[lag:]) for lag in range(1, max_lag + 1)])
    for i in range(1, vals.size - 1):
        if vals[i] < vals[i - 1] and vals[i] <= vals[i + 1]:
            return float(i + 1)
    return float(np.argmin(vals) + 1)


def _lagged_xy(source: np.ndarray, target: np.ndarray, lag: int = 30, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    source = _finite_fill(source)
    target = _finite_fill(target)
    n = min(source.size, target.size)
    lag = min(lag, max(1, n // 5))
    rows = n - lag - horizon + 1
    if rows < 20:
        raise ValueError("not enough samples for lagged design")
    X = np.empty((rows, lag), dtype=float)
    y = np.empty(rows, dtype=float)
    for i in range(rows):
        end = i + lag
        X[i] = source[i:end][::-1]
        y[i] = target[end + horizon - 1]
    return X, y


def _predictive_models(source: np.ndarray, target: np.ndarray, seed: int) -> dict[str, float]:
    X, y = _lagged_xy(source, target, lag=30)
    split = max(20, int(0.7 * X.shape[0]))
    if X.shape[0] - split < 10:
        split = X.shape[0] // 2
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    ridge = Ridge(alpha=1e-3)
    ridge.fit(Xtr, ytr)
    pred_lin = ridge.predict(Xte)
    # R^2 is unbounded below on nonstationary holdouts; clip to [-1, 1] so a single
    # pathological dataset (e.g. chirp) cannot dominate the downstream meta-model.
    r2_lin = float(np.clip(r2_score(yte, pred_lin), -1.0, 1.0))
    gbm = GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=seed)
    gbm.fit(Xtr, ytr)
    pred_gbm = gbm.predict(Xte)
    r2_gbm = float(np.clip(r2_score(yte, pred_gbm), -1.0, 1.0))
    rmse = float(np.sqrt(np.mean((pred_gbm - yte) ** 2)))
    scale = float(np.std(yte))
    return {
        "r2_linear": r2_lin,
        "nl_gain": r2_gbm - r2_lin,
        "pred_nrmse_gbm": rmse / max(scale, 1e-12),
    }


def _mem_capacity(inputs: np.ndarray, outputs: np.ndarray) -> tuple[float, bool]:
    u = _zscore(inputs)
    y = _zscore(outputs)
    n = min(u.size, y.size)
    if n < 200:
        return math.nan, False
    max_delay = min(50, n // 10)
    state_lag = min(20, n // 20)
    X_all, _ = _lagged_xy(y[:n], y[:n], lag=state_lag)
    start = state_lag
    scores = []
    for delay in range(1, max_delay + 1):
        target = u[start - delay : start - delay + X_all.shape[0]]
        if target.size != X_all.shape[0]:
            continue
        split = int(0.7 * X_all.shape[0])
        model = Ridge(alpha=1e-3).fit(X_all[:split], target[:split])
        score = r2_score(target[split:], model.predict(X_all[split:]))
        if np.isfinite(score):
            scores.append(max(0.0, float(score)))
    return float(np.sum(scores)), bool(scores)


def _snr_db(source: np.ndarray, target: np.ndarray) -> tuple[float, bool]:
    X, y = _lagged_xy(source, target, lag=10)
    if X.shape[0] < 80:
        return math.nan, False
    split = int(0.7 * X.shape[0])
    knn = KNeighborsRegressor(n_neighbors=min(7, max(2, split // 20)), weights="distance")
    knn.fit(X[:split], y[:split])
    pred = knn.predict(X[split:])
    noise = float(np.mean((y[split:] - pred) ** 2))
    signal_power = float(np.var(y[split:]))
    if noise <= 0 or signal_power <= 0:
        return math.inf, True
    return float(10.0 * np.log10(signal_power / noise)), True


def _lyapunov_rosenstein(x: np.ndarray, dt: float = 1.0) -> tuple[float, bool]:
    x = _zscore(x)
    n = x.size
    if n < MIN_LENGTH_LYAPUNOV or np.std(x) < 1e-12:
        return math.nan, False
    if dt < 1.0:
        delay = min(24, max(1, int(round(_ami_first_min(x) * 1.8))))
        fit_start = 12
        fit_count = 8
    else:
        delay = 1
        fit_start = 1
        fit_count = 6
    emb_dim = 3
    m = n - (emb_dim - 1) * delay
    if m < 200:
        return math.nan, False
    emb = np.column_stack([x[i * delay : i * delay + m] for i in range(emb_dim)])
    max_points = min(2200, emb.shape[0] - 40)
    emb = emb[:max_points]
    m = emb.shape[0]
    theiler = max(20, emb_dim * delay)
    nn = NearestNeighbors(n_neighbors=min(32, m), algorithm="auto").fit(emb)
    distances, indices = nn.kneighbors(emb)
    neigh = np.full(m, -1, dtype=int)
    for i in range(m):
        for j in indices[i, 1:]:
            if abs(int(j) - i) > theiler:
                neigh[i] = int(j)
                break
    valid_i = np.flatnonzero(neigh >= 0)
    if valid_i.size < 100:
        return math.nan, False
    kmax = min(30, m // 8)
    means = []
    ks = []
    for k in range(1, kmax + 1):
        idx = valid_i[(valid_i + k < m) & (neigh[valid_i] + k < m)]
        if idx.size < 80:
            continue
        d = np.linalg.norm(emb[idx + k] - emb[neigh[idx] + k], axis=1)
        d = d[np.isfinite(d) & (d > 1e-12)]
        if d.size:
            means.append(float(np.mean(np.log(d))))
            ks.append(k)
    if len(ks) < 6:
        return math.nan, False
    start = min(fit_start - 1, max(0, len(ks) - fit_count))
    ks_arr = np.asarray(ks[start : start + min(fit_count, len(ks) - start)], dtype=float)
    means_arr = np.asarray(means[start : start + ks_arr.size], dtype=float)
    if ks_arr.size < 4:
        return math.nan, False
    xfit = ks_arr * max(dt, 1e-12)
    coeffs = np.polyfit(xfit, means_arr, 1)
    slope = float(coeffs[0])
    resid = means_arr - np.polyval(coeffs, xfit)
    ss_tot = float(np.sum((means_arr - np.mean(means_arr)) ** 2))
    fit_r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 1e-12 else 0.0
    # A valid positive Lyapunov exponent requires a clean positive divergence slope
    # over the scaling region. Negative/flat slopes or poor linear fits (boundary
    # saturation, noise) must report invalid, NOT a fabricated valid zero.
    if not np.isfinite(slope) or slope <= 0.0 or fit_r2 < 0.6:
        return math.nan, False
    return slope, True


def _zero_one_K(x: np.ndarray, seed: int) -> tuple[float, bool]:
    x = _zscore(x)
    n0 = x.size
    if n0 < MIN_LENGTH_ZERO_ONE:
        return math.nan, False
    # Gottwald & Melbourne: the test needs decorrelated samples. Oversampled
    # continuous flows (dt < 1) must be decimated by their dominant timescale,
    # else p,q stay bounded and K collapses to ~0 for genuinely chaotic data.
    try:
        stride = int(round(_ami_first_min(x)))
    except Exception:
        stride = 1
    if not np.isfinite(stride) or stride < 1:
        stride = 1
    # Cap decimation so the decorrelated series keeps ~>=100 points; long-timescale
    # flows (e.g. Roessler, ami~20) need a larger stride than a tight cap would allow.
    stride = min(stride, max(1, n0 // 100))
    x = x[::stride]
    n = x.size
    if n < 80 or float(np.std(x)) < 1e-12:
        return math.nan, False
    rng = np.random.default_rng(seed)
    cs = rng.uniform(np.pi / 5.0, 4.0 * np.pi / 5.0, size=12)
    n_cut = min(200, n // 10)
    if n_cut < 5:
        return math.nan, False
    t = np.arange(1, n + 1, dtype=float)
    n_vec = np.arange(1, n_cut + 1, dtype=float)
    values = []
    for c in cs:
        p = np.cumsum(x * np.cos(c * t))
        q = np.cumsum(x * np.sin(c * t))
        msd = np.array([np.mean((p[j:] - p[:-j]) ** 2 + (q[j:] - q[:-j]) ** 2) for j in range(1, n_cut + 1)])
        if np.std(msd) > 1e-12:
            values.append(float(np.corrcoef(n_vec, msd)[0, 1]))
    if not values:
        return math.nan, False
    return float(np.median(values)), True


def _spectral_features(x: np.ndarray) -> dict[str, float]:
    x = _zscore(x)
    if x.size < 16:
        return {"spectral_entropy": math.nan, "dom_freq": math.nan, "spectral_flatness": math.nan}
    freqs, psd = signal.welch(x, nperseg=min(1024, x.size))
    psd = np.asarray(psd, dtype=float)
    psd = np.maximum(psd, 0.0)
    psd_sum = float(np.sum(psd))
    if psd_sum <= 0:
        return {"spectral_entropy": math.nan, "dom_freq": math.nan, "spectral_flatness": math.nan}
    p = psd / psd_sum
    entropy = -float(np.sum(p * np.log(p + 1e-15)) / np.log(p.size))
    idx = int(np.argmax(psd[1:]) + 1) if psd.size > 1 else 0
    flatness = float(np.exp(np.mean(np.log(psd + 1e-15))) / max(np.mean(psd), 1e-15))
    return {"spectral_entropy": entropy, "dom_freq": float(freqs[idx]), "spectral_flatness": flatness}


# ADF (constant, no-trend) tau -> p anchors; monotone so np.interp is continuous.
_ADF_TAU = np.array([-4.5, -3.96, -3.43, -3.12, -2.86, -2.57, -2.26, -1.95, -1.62, -1.28, -0.90, -0.40, 0.0, 0.5, 1.0])
_ADF_P = np.array([0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.975, 0.99])
# KPSS (level) statistic -> p anchors; p decreases as the statistic grows.
_KPSS_STAT = np.array([0.05, 0.10, 0.216, 0.347, 0.463, 0.574, 0.739, 1.0, 2.0])
_KPSS_P = np.array([0.95, 0.90, 0.50, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001])


def _adf_p(x: np.ndarray) -> float:
    """ADF p-value: statsmodels if importable, else continuous MacKinnon fallback."""
    x = _finite_fill(x)
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore

        return float(adfuller(x, autolag="AIC")[1])
    except Exception:
        return _adf_p_fallback(x)


def _kpss_p(x: np.ndarray) -> float:
    x = _finite_fill(x)
    try:
        import warnings as _w
        from statsmodels.tsa.stattools import kpss  # type: ignore

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            return float(kpss(x, regression="c", nlags="auto")[1])
    except Exception:
        return _kpss_p_fallback(x)


def _stationarity_features(raw: np.ndarray) -> dict[str, float | int]:
    x = _finite_fill(raw)
    adf_p = _adf_p(x)
    kpss_p = _kpss_p(x)
    n_diffs = 0
    z = x.copy()
    while n_diffs < 2 and _adf_p(z) > 0.05:  # same path as adf_p, not the discrete one
        z = np.diff(z)
        n_diffs += 1
        if z.size < 20:
            break
    return {"adf_p": adf_p, "kpss_p": kpss_p, "n_diffs": n_diffs}


def _adf_p_fallback(x: np.ndarray) -> float:
    x = _finite_fill(x)
    if x.size < 30 or np.std(x) < 1e-12:
        return 0.99
    dx = np.diff(x)
    ylag = x[:-1]
    max_lag = min(5, max(0, x.size // 100))
    rows = dx.size - max_lag
    if rows <= 10:
        return 0.99
    y = dx[max_lag:]
    cols = [np.ones(rows), ylag[max_lag:]]
    for lag in range(1, max_lag + 1):
        cols.append(dx[max_lag - lag : dx.size - lag])
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(1, rows - X.shape[1])
    sigma2 = float(np.sum(resid**2) / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(cov[1, 1], 1e-12)))
    t_stat = float(coef[1] / se)
    # Continuous p via interpolation over the tau distribution (vs the old 5-value ladder).
    return float(np.clip(np.interp(t_stat, _ADF_TAU, _ADF_P), 0.001, 0.999))


def _kpss_p_fallback(x: np.ndarray) -> float:
    x = _finite_fill(x)
    n = x.size
    if n < 30:
        return 0.99
    resid = x - np.mean(x)
    s = np.cumsum(resid)
    eta = float(np.sum(s**2) / (n**2))
    nlags = int(np.sqrt(n))
    gamma0 = float(np.mean(resid**2))
    lrv = gamma0
    for lag in range(1, nlags + 1):
        weight = 1.0 - lag / (nlags + 1.0)
        cov = float(np.mean(resid[lag:] * resid[:-lag]))
        lrv += 2.0 * weight * cov
    stat = eta / max(lrv, 1e-12)
    # Continuous interpolation over KPSS level-stationarity critical values.
    return float(np.clip(np.interp(stat, _KPSS_STAT, _KPSS_P), 0.001, 0.99))


def _dfa_alpha(x: np.ndarray) -> tuple[float, bool]:
    x = _finite_fill(x)
    n = x.size
    if n < MIN_LENGTH_DFA:
        return math.nan, False
    y = np.cumsum(x - np.mean(x))
    sizes = np.unique(np.logspace(np.log10(8), np.log10(max(16, n // 4)), num=16).astype(int))
    flucts = []
    used = []
    for size in sizes:
        if size < 4 or n // size < 4:
            continue
        rms = []
        for start in range(0, (n // size) * size, size):
            seg = y[start : start + size]
            t = np.arange(size, dtype=float)
            coef = np.polyfit(t, seg, 1)
            detrended = seg - np.polyval(coef, t)
            rms.append(float(np.sqrt(np.mean(detrended**2))))
        val = float(np.sqrt(np.mean(np.asarray(rms) ** 2)))
        if val > 0 and np.isfinite(val):
            flucts.append(val)
            used.append(size)
    if len(used) < 5:
        return math.nan, False
    alpha = float(np.polyfit(np.log(used), np.log(flucts), 1)[0])
    return alpha, np.isfinite(alpha)


def _perm_entropy(x: np.ndarray, order: int = 4, delay: int = 1) -> float:
    x = _finite_fill(x)
    n = x.size - delay * (order - 1)
    if n <= order:
        return math.nan
    patterns = []
    for i in range(n):
        window = x[i : i + delay * order : delay]
        patterns.append(tuple(np.argsort(window, kind="mergesort")))
    counts = np.array(list(Counter(patterns).values()), dtype=float)
    p = counts / np.sum(counts)
    return float(-np.sum(p * np.log(p + 1e-15)) / np.log(math.factorial(order)))
