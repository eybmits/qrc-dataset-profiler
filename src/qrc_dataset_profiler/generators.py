"""Dataset generators for the frozen protocol v1 catalog."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy import signal

from qrc_dataset_profiler.spec import Dataset, DatasetSpec


ALL_SPECS: list[DatasetSpec] = [
    DatasetSpec("mackey_glass_t17", "chaotic_flow", "synthetic", "forecast", {"tau": 17}, seed=101),
    DatasetSpec("mackey_glass_t30", "chaotic_flow", "synthetic", "forecast", {"tau": 30}, seed=102),
    DatasetSpec("lorenz63", "chaotic_flow", "synthetic", "forecast", {"rho": 28.0, "dt": 0.02}, seed=103),
    DatasetSpec("rossler", "chaotic_flow", "synthetic", "forecast", {"c": 5.7, "dt": 0.05}, seed=104),
    DatasetSpec("logistic_r4", "chaotic_map", "synthetic", "forecast", {"r": 4.0}, seed=105),
    DatasetSpec("henon", "chaotic_map", "synthetic", "forecast", {"a": 1.4, "b": 0.3}, seed=106),
    DatasetSpec("narma10", "input_driven", "synthetic", "input_driven", {"order": 10}, seed=107),
    DatasetSpec("narma20", "input_driven", "synthetic", "input_driven", {"order": 20}, seed=108),
    DatasetSpec("linear_memory", "input_driven", "synthetic", "input_driven", {"lag": 8}, seed=109),
    DatasetSpec("nonlinear_ipc", "input_driven", "synthetic", "input_driven", {"lags": [1, 3, 5], "degree": 2}, seed=110),
    DatasetSpec("mso8", "oscillatory", "synthetic", "forecast", {}, seed=111),
    DatasetSpec("quasi_periodic", "oscillatory", "synthetic", "forecast", {}, seed=112),
    DatasetSpec("ar2", "linear_stochastic", "synthetic", "forecast", {"phi": [0.6, -0.3]}, seed=113),
    DatasetSpec("garch11", "nonlinear_stochastic", "synthetic", "forecast", {"omega": 0.05, "alpha": 0.12, "beta": 0.82}, seed=114),
    DatasetSpec("fbm_h08", "long_range", "synthetic", "forecast", {"H": 0.8}, seed=115),
    DatasetSpec("pink_noise", "colored_noise", "synthetic", "forecast", {"beta": 1.0}, seed=116),
    DatasetSpec("chirp", "nonstationary", "synthetic", "forecast", {"f0": 0.01, "f1": 0.35}, seed=117),
    DatasetSpec("regime_switch_ar", "nonstationary", "synthetic", "forecast", {"p_switch": 0.015}, seed=118),
    DatasetSpec("lorenz63_noisy", "chaotic_flow", "synthetic", "forecast", {"rho": 28.0, "dt": 0.02, "snr_db": 10.0}, seed=119),
    DatasetSpec("santa_fe_laser", "real_bridge", "real", "forecast", {"dataset": "A"}, seed=120),
]


def generate(spec: DatasetSpec) -> Dataset:
    """Generate one dataset described by ``spec``."""

    dispatch: dict[str, Callable[[DatasetSpec], Dataset]] = {
        "mackey_glass_t17": _mackey_glass,
        "mackey_glass_t30": _mackey_glass,
        "lorenz63": _lorenz63,
        "rossler": _rossler,
        "logistic_r4": _logistic,
        "henon": _henon,
        "narma10": _narma,
        "narma20": _narma,
        "linear_memory": _linear_memory,
        "nonlinear_ipc": _nonlinear_ipc,
        "mso8": _mso8,
        "quasi_periodic": _quasi_periodic,
        "ar2": _ar2,
        "garch11": _garch11,
        "fbm_h08": _fbm,
        "pink_noise": _pink_noise,
        "chirp": _chirp,
        "regime_switch_ar": _regime_switch_ar,
        "lorenz63_noisy": _lorenz63_noisy,
        "santa_fe_laser": _santa_fe_laser,
    }
    if spec.name not in dispatch:
        raise ValueError(f"unknown dataset spec: {spec.name}")
    return dispatch[spec.name](spec)


def add_observation_noise(series: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white observation noise at a target signal-to-noise ratio in dB."""

    x = np.asarray(series, dtype=float)
    signal_power = float(np.var(x))
    if signal_power <= 0:
        return x.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(noise_power), size=x.shape)


def _rng(spec: DatasetSpec) -> np.random.Generator:
    return np.random.default_rng(spec.seed)


def _slice(series: np.ndarray, spec: DatasetSpec, burn: int = 0) -> np.ndarray:
    arr = np.asarray(series, dtype=float)[burn : burn + spec.length]
    if arr.size != spec.length:
        raise RuntimeError(f"{spec.name} produced {arr.size} values, expected {spec.length}")
    return arr


def _mackey_glass(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    tau = int(spec.params["tau"])
    beta, gamma, n_exp, dt = 0.2, 0.1, 10, 1.0
    total = spec.length + 1000 + tau + 2
    x = np.empty(total, dtype=float)
    x[: tau + 1] = 1.2 + 0.01 * rng.normal(size=tau + 1)
    for t in range(tau, total - 1):
        delayed = x[t - tau]
        dx = beta * delayed / (1.0 + delayed**n_exp) - gamma * x[t]
        x[t + 1] = x[t] + dt * dx
    return Dataset(spec, _slice(x, spec, burn=1000 + tau), ground_truth={"is_chaotic": 1.0})


def _lorenz_series(spec: DatasetSpec) -> np.ndarray:
    rho = float(spec.params.get("rho", 28.0))
    dt = float(spec.params.get("dt", 0.02))
    n = spec.length + 1000

    def rhs(_t: float, xyz: np.ndarray) -> tuple[float, float, float]:
        sigma, beta = 10.0, 8.0 / 3.0
        x, y, z = xyz
        return (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)

    t_eval = np.arange(n, dtype=float) * dt
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), (1.0, 1.0, 1.0), t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.y[0]


def _lorenz63(spec: DatasetSpec) -> Dataset:
    return Dataset(
        spec,
        _slice(_lorenz_series(spec), spec, burn=1000),
        ground_truth={"true_lyapunov": 0.906, "is_chaotic": 1.0},
    )


def _lorenz63_noisy(spec: DatasetSpec) -> Dataset:
    clean = _slice(_lorenz_series(spec), spec, burn=1000)
    noisy = add_observation_noise(clean, float(spec.params.get("snr_db", 10.0)), _rng(spec))
    return Dataset(spec, noisy, ground_truth={"is_chaotic": 1.0})


def _rossler(spec: DatasetSpec) -> Dataset:
    c = float(spec.params.get("c", 5.7))
    dt = float(spec.params.get("dt", 0.05))
    n = spec.length + 1000

    def rhs(_t: float, xyz: np.ndarray) -> tuple[float, float, float]:
        a, b = 0.2, 0.2
        x, y, z = xyz
        return (-y - z, x + a * y, b + z * (x - c))

    t_eval = np.arange(n, dtype=float) * dt
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), (0.1, 0.0, 0.0), t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return Dataset(spec, _slice(sol.y[0], spec, burn=1000), ground_truth={"is_chaotic": 1.0})


def _logistic(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    r = float(spec.params.get("r", 4.0))
    total = spec.length + 1000
    x = np.empty(total, dtype=float)
    x[0] = rng.uniform(0.11, 0.89)
    for t in range(total - 1):
        x[t + 1] = r * x[t] * (1.0 - x[t])
        x[t + 1] = min(max(x[t + 1], np.finfo(float).eps), 1.0 - np.finfo(float).eps)
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"true_lyapunov": float(np.log(2.0)), "is_chaotic": 1.0})


def _henon(spec: DatasetSpec) -> Dataset:
    a = float(spec.params.get("a", 1.4))
    b = float(spec.params.get("b", 0.3))
    total = spec.length + 1000
    x = np.empty(total, dtype=float)
    y = np.empty(total, dtype=float)
    x[0], y[0] = 0.1, 0.3
    for t in range(total - 1):
        x[t + 1] = 1.0 - a * x[t] ** 2 + y[t]
        y[t + 1] = b * x[t]
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"true_lyapunov": 0.419, "is_chaotic": 1.0})


def _narma(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    order = int(spec.params["order"])
    total = spec.length + 1000 + order + 2
    u = rng.uniform(0.0, 0.2, size=total)
    y = np.zeros(total, dtype=float)
    for t in range(order, total - 1):
        y_next = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - order + 1 : t + 1])
            + 1.5 * u[t - order + 1] * u[t]
            + 0.1
        )
        y[t + 1] = y_next if np.isfinite(y_next) else 0.0
    burn = 1000 + order
    return Dataset(
        spec,
        _slice(y, spec, burn=burn),
        inputs=_slice(u, spec, burn=burn),
        ground_truth={"true_memory_order": float(order)},
    )


def _linear_memory(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    lag = int(spec.params.get("lag", 8))
    total = spec.length + 1000 + lag
    u = rng.uniform(-1.0, 1.0, size=total)
    y = np.roll(u, lag)
    y[:lag] = 0.0
    y += 0.01 * rng.normal(size=total)
    burn = 1000 + lag
    return Dataset(spec, _slice(y, spec, burn=burn), inputs=_slice(u, spec, burn=burn), ground_truth={"true_memory_order": float(lag)})


def _nonlinear_ipc(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    lags = [int(v) for v in spec.params.get("lags", [1, 3, 5])]
    total = spec.length + 1000 + max(lags) + 1
    u = rng.uniform(-1.0, 1.0, size=total)
    y = np.zeros(total, dtype=float)
    for t in range(max(lags), total):
        y[t] = u[t - lags[0]] * u[t - lags[1]] + 0.5 * u[t - lags[2]] ** 2
    y += 0.01 * rng.normal(size=total)
    burn = 1000 + max(lags)
    return Dataset(spec, _slice(y, spec, burn=burn), inputs=_slice(u, spec, burn=burn), ground_truth={"true_memory_order": float(max(lags))})


def _mso8(spec: DatasetSpec) -> Dataset:
    freqs = [0.031, 0.047, 0.073, 0.109, 0.151, 0.197, 0.251, 0.317]
    t = np.arange(spec.length, dtype=float)
    y = sum(np.sin(2.0 * np.pi * f * t) for f in freqs)
    return Dataset(spec, np.asarray(y, dtype=float), ground_truth={"true_n_frequencies": 8.0, "true_frequencies": freqs})


def _quasi_periodic(spec: DatasetSpec) -> Dataset:
    freqs = [0.033, np.sqrt(2.0) / 31.0, np.sqrt(3.0) / 37.0]
    t = np.arange(spec.length, dtype=float)
    y = sum(np.sin(2.0 * np.pi * f * t + 0.3 * i) for i, f in enumerate(freqs))
    return Dataset(spec, np.asarray(y, dtype=float), ground_truth={"true_n_frequencies": 3.0, "true_frequencies": [float(f) for f in freqs]})


def _ar2(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    phi1, phi2 = [float(v) for v in spec.params.get("phi", [0.6, -0.3])]
    total = spec.length + 1000
    x = np.zeros(total, dtype=float)
    eps = rng.normal(0.0, 0.5, size=total)
    for t in range(2, total):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + eps[t]
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"true_memory_order": 2.0})


def _garch11(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    omega = float(spec.params.get("omega", 0.05))
    alpha = float(spec.params.get("alpha", 0.12))
    beta = float(spec.params.get("beta", 0.82))
    total = spec.length + 1000
    eps = np.zeros(total, dtype=float)
    var = np.full(total, omega / max(1.0 - alpha - beta, 0.05), dtype=float)
    z = rng.normal(size=total)
    for t in range(1, total):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(max(var[t], 1e-12)) * z[t]
    return Dataset(spec, _slice(eps, spec, burn=1000))


def _fbm(spec: DatasetSpec) -> Dataset:
    H = float(spec.params.get("H", 0.8))
    x = _hosking_fgn(spec.length + 1000, H, _rng(spec))
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"true_hurst": H})


def _hosking_fgn(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    k = np.arange(n, dtype=float)
    gamma = 0.5 * (np.abs(k - 1.0) ** (2.0 * H) - 2.0 * np.abs(k) ** (2.0 * H) + np.abs(k + 1.0) ** (2.0 * H))
    x = np.zeros(n, dtype=float)
    phi = np.zeros(n, dtype=float)
    v = gamma[0]
    x[0] = rng.normal(scale=np.sqrt(v))
    for i in range(1, n):
        if i == 1:
            phi_i = gamma[1] / v
        else:
            phi_i = (gamma[i] - np.dot(phi[: i - 1], gamma[1:i][::-1])) / v
        old = phi[: i - 1].copy()
        if i > 1:
            phi[: i - 1] = old - phi_i * old[::-1]
        phi[i - 1] = phi_i
        v = max(v * (1.0 - phi_i**2), 1e-12)
        x[i] = np.dot(phi[:i], x[i - 1 :: -1]) + rng.normal(scale=np.sqrt(v))
    return x


def _pink_noise(spec: DatasetSpec) -> Dataset:
    beta = float(spec.params.get("beta", 1.0))
    rng = _rng(spec)
    n = spec.length
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = freqs[1:] ** (-beta / 2.0)
    coeffs = (rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)) * scale
    coeffs[0] = 0.0
    x = np.fft.irfft(coeffs, n=n)
    return Dataset(spec, x.astype(float))


def _chirp(spec: DatasetSpec) -> Dataset:
    t = np.linspace(0.0, 1.0, spec.length)
    y = signal.chirp(t, f0=float(spec.params.get("f0", 0.01)) * spec.length, f1=float(spec.params.get("f1", 0.35)) * spec.length, t1=1.0)
    return Dataset(spec, y.astype(float))


def _regime_switch_ar(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    p_switch = float(spec.params.get("p_switch", 0.015))
    total = spec.length + 1000
    x = np.zeros(total, dtype=float)
    state = 0
    phis = [(0.85, -0.15), (-0.55, 0.25)]
    for t in range(2, total):
        if rng.random() < p_switch:
            state = 1 - state
        phi1, phi2 = phis[state]
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + rng.normal(scale=0.35 + 0.15 * state)
    return Dataset(replace(spec, params={**spec.params, "phis": phis}), _slice(x, spec, burn=1000))


def _santa_fe_laser(spec: DatasetSpec) -> Dataset:
    data_dir = Path.cwd() / "data"
    candidates = sorted(data_dir.glob("santa_fe_laser.*"))
    for path in candidates:
        try:
            if path.suffix == ".npy":
                arr = np.load(path)
            else:
                arr = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
            arr = np.asarray(arr, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size >= spec.length:
                return Dataset(spec, arr[: spec.length], ground_truth={})
        except Exception:
            continue
    return Dataset(spec, np.asarray([], dtype=float), ground_truth={"_unavailable": True})
