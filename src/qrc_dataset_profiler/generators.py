"""Dataset generators for the frozen protocol v1 catalog."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy import signal
from scipy.stats import qmc

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
        "mackey_glass": _mackey_glass,
        "lorenz63": _lorenz63,
        "rossler": _rossler,
        "logistic_r4": _logistic,
        "logistic": _logistic,
        "henon": _henon,
        "narma10": _narma,
        "narma20": _narma,
        "narma": _narma,
        "linear_memory": _linear_memory,
        "nonlinear_ipc": _nonlinear_ipc,
        "mso8": _mso8,
        "mso": _mso8,
        "quasi_periodic": _quasi_periodic,
        "ar2": _ar2,
        "ar": _ar2,
        "garch11": _garch11,
        "garch": _garch11,
        "fbm_h08": _fbm,
        "fbm": _fbm,
        "pink_noise": _pink_noise,
        "colored_noise": _pink_noise,
        "chirp": _chirp,
        "regime_switch_ar": _regime_switch_ar,
        "lorenz63_noisy": _lorenz63_noisy,
        "santa_fe_laser": _santa_fe_laser,
    }
    key = str(spec.params.get("generator", spec.name))
    if key not in dispatch:
        raise ValueError(f"unknown dataset spec: {spec.name}")
    ds = dispatch[key](spec)
    if spec.params.get("noise_overlay") and not ds.ground_truth.get("_unavailable"):
        noisy = add_observation_noise(ds.series, float(spec.params["snr_db"]), _rng(spec))
        return Dataset(spec, noisy, inputs=ds.inputs, ground_truth=dict(ds.ground_truth))
    return ds


def make_sweep_specs(n_per_family: int = 20, seed: int = 0) -> list[DatasetSpec]:
    """Build a deterministic parameterized synthetic catalog sweep.

    Latin-hypercube samples cover each generator family's meaningful knobs while
    preserving the schema-v1 ``DatasetSpec`` contract.  ``params["generator"]``
    selects the concrete generator, leaving ``name`` free to identify variants.
    """

    n = max(1, int(n_per_family))
    seed = int(seed)
    specs: list[DatasetSpec] = []
    offsets = {
        "mackey_glass": 1100,
        "lorenz63": 1200,
        "rossler": 1300,
        "logistic": 1400,
        "henon": 1500,
        "narma": 1600,
        "linear_memory": 1700,
        "nonlinear_ipc": 1800,
        "mso": 1900,
        "quasi_periodic": 2000,
        "ar": 2100,
        "garch": 2200,
        "fbm": 2300,
        "colored_noise": 2400,
        "chirp": 2500,
        "regime_switch_ar": 2600,
    }

    def add(name: str, family: str, task_type: str, params: dict[str, object], i: int, *, source: str = "synthetic") -> None:
        gen = str(params["generator"])
        spec_seed = seed * 100_000 + offsets[gen] + i
        specs.append(DatasetSpec(name, family, source, task_type, params, seed=spec_seed))

    samples = {key: _lhs(n, dim, seed + j) for j, (key, dim) in enumerate(
        [
            ("mackey_glass", 1),
            ("lorenz63", 1),
            ("rossler", 1),
            ("logistic", 1),
            ("henon", 1),
            ("nonlinear_ipc", 2),
            ("ar", 2),
            ("garch", 2),
            ("fbm", 1),
            ("colored_noise", 1),
            ("chirp", 2),
            ("regime_switch_ar", 3),
        ],
        start=1,
    )}

    narma_orders = [2, 5, 10, 15, 20, 25, 30]
    mso_counts = list(range(2, 9))
    quasi_counts = list(range(2, 6))

    for i, u in enumerate(samples["mackey_glass"][:, 0]):
        tau = int(round(_scale(u, 15, 45)))
        add(f"mackey_glass_tau{tau:02d}_s{i:03d}", "chaotic_flow", "forecast", {"generator": "mackey_glass", "tau": tau}, i)

    for i, u in enumerate(samples["lorenz63"][:, 0]):
        rho = float(_scale(u, 22.0, 46.0))
        add(f"lorenz63_rho{rho:.2f}_s{i:03d}", "chaotic_flow", "forecast", {"generator": "lorenz63", "rho": rho, "dt": 0.02}, i)

    for i, u in enumerate(samples["rossler"][:, 0]):
        c = float(_scale(u, 4.0, 9.0))
        add(f"rossler_c{c:.2f}_s{i:03d}", "chaotic_flow", "forecast", {"generator": "rossler", "c": c, "dt": 0.05}, i)

    for i, u in enumerate(samples["logistic"][:, 0]):
        r = float(_scale(u, 3.6, 4.0))
        add(f"logistic_r{r:.4f}_s{i:03d}", "chaotic_map", "forecast", {"generator": "logistic", "r": r}, i)

    for i, u in enumerate(samples["henon"][:, 0]):
        a = float(_scale(u, 1.0, 1.4))
        add(f"henon_a{a:.3f}_s{i:03d}", "chaotic_map", "forecast", {"generator": "henon", "a": a, "b": 0.3}, i)

    for i in range(n):
        order = narma_orders[i % len(narma_orders)]
        add(f"narma{order:02d}_s{i:03d}", "input_driven", "input_driven", {"generator": "narma", "order": order}, i)

    for i in range(n):
        lag = 1 + ((i * 7 + seed) % 25)
        add(f"linear_memory_lag{lag:02d}_s{i:03d}", "input_driven", "input_driven", {"generator": "linear_memory", "lag": lag}, i)

    for i, row in enumerate(samples["nonlinear_ipc"]):
        degree = 2 + int(np.floor(row[0] * 3.0))
        degree = min(degree, 4)
        max_lag = int(round(_scale(row[1], 3, 25)))
        lags = sorted({1, max(2, max_lag // 2), max_lag})
        while len(lags) < 3:
            lags.append(lags[-1] + 1)
        add(
            f"nonlinear_ipc_d{degree}_l{max(lags):02d}_s{i:03d}",
            "input_driven",
            "input_driven",
            {"generator": "nonlinear_ipc", "degree": degree, "lags": lags[:3]},
            i,
        )

    for i in range(n):
        n_freqs = mso_counts[i % len(mso_counts)]
        add(f"mso{n_freqs}_s{i:03d}", "oscillatory", "forecast", {"generator": "mso", "n_freqs": n_freqs}, i)

    for i in range(n):
        n_freqs = quasi_counts[i % len(quasi_counts)]
        add(f"quasi_periodic{n_freqs}_s{i:03d}", "oscillatory", "forecast", {"generator": "quasi_periodic", "n_freqs": n_freqs}, i)

    for i, row in enumerate(samples["ar"]):
        p = 1 + int(np.floor(row[0] * 3.0))
        p = min(p, 3)
        phi = _stable_ar_coeffs(p, seed * 10_000 + i)
        add(f"ar{p}_s{i:03d}", "linear_stochastic", "forecast", {"generator": "ar", "phi": phi}, i)

    for i, row in enumerate(samples["garch"]):
        alpha = float(_scale(row[0], 0.03, 0.24))
        beta_max = max(0.35, 0.98 - alpha)
        beta = float(_scale(row[1], 0.35, beta_max))
        add(f"garch_a{alpha:.3f}_b{beta:.3f}_s{i:03d}", "nonlinear_stochastic", "forecast", {"generator": "garch", "omega": 0.05, "alpha": alpha, "beta": beta}, i)

    for i, u in enumerate(samples["fbm"][:, 0]):
        H = float(_scale(u, 0.5, 0.95))
        add(f"fbm_h{H:.3f}_s{i:03d}", "long_range", "forecast", {"generator": "fbm", "H": H}, i)

    for i, u in enumerate(samples["colored_noise"][:, 0]):
        beta = float(_scale(u, 0.5, 2.0))
        add(f"colored_noise_b{beta:.3f}_s{i:03d}", "colored_noise", "forecast", {"generator": "colored_noise", "beta": beta}, i)

    for i, row in enumerate(samples["chirp"]):
        f0 = float(_scale(row[0], 0.005, 0.08))
        f1 = float(_scale(row[1], max(f0 + 0.05, 0.12), 0.45))
        add(f"chirp_f{f0:.3f}_{f1:.3f}_s{i:03d}", "nonstationary", "forecast", {"generator": "chirp", "f0": f0, "f1": f1}, i)

    for i, row in enumerate(samples["regime_switch_ar"]):
        p_switch = float(_scale(row[0], 0.003, 0.05))
        phi_a = (float(_scale(row[1], 0.55, 0.92)), float(_scale(row[2], -0.25, 0.05)))
        phi_b = (float(_scale(row[1], -0.85, -0.35)), float(_scale(row[2], 0.05, 0.35)))
        add(
            f"regime_switch_ar_p{p_switch:.3f}_s{i:03d}",
            "nonstationary",
            "forecast",
            {"generator": "regime_switch_ar", "p_switch": p_switch, "phis": [phi_a, phi_b], "sigmas": [0.25, 0.5]},
            i,
        )

    specs.extend(_noise_overlay_specs(specs, n_per_family=n, seed=seed))
    return specs


def add_observation_noise(series: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white observation noise at a target signal-to-noise ratio in dB."""

    x = np.asarray(series, dtype=float)
    signal_power = float(np.var(x))
    if signal_power <= 0:
        return x.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(noise_power), size=x.shape)


def _lhs(n: int, d: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=int(d), seed=int(seed))
    return sampler.random(n=int(n))


def _scale(u: float, lo: float, hi: float) -> float:
    return float(lo + float(u) * (hi - lo))


def _stable_ar_coeffs(p: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    p = int(p)
    if p == 1:
        return [float(rng.uniform(-0.9, 0.9))]
    for _ in range(10_000):
        phi = rng.uniform(-0.85, 0.85, size=p)
        if _is_stable_ar(phi):
            return [float(v) for v in phi]
    return [0.45] + [0.0] * (p - 1)


def _is_stable_ar(phi: np.ndarray | list[float]) -> bool:
    coeffs = np.r_[1.0, -np.asarray(phi, dtype=float)]
    roots = np.roots(coeffs)
    return bool(np.all(np.abs(roots) < 0.98))


def _noise_overlay_specs(specs: list[DatasetSpec], *, n_per_family: int, seed: int) -> list[DatasetSpec]:
    overlay_generators = {"lorenz63", "logistic", "linear_memory", "mso", "fbm", "chirp"}
    selected = [s for s in specs if s.params.get("generator") in overlay_generators]
    selected = selected[: min(len(selected), max(6, n_per_family * 2))]
    out: list[DatasetSpec] = []
    snrs = (5.0, 10.0, 20.0)
    for j, base in enumerate(selected):
        snr = snrs[j % len(snrs)]
        params = {**base.params, "noise_overlay": True, "snr_db": snr}
        out.append(
            replace(
                base,
                name=f"{base.name}_snr{int(snr)}db",
                params=params,
                seed=seed * 100_000 + 9000 + j,
            )
        )
    return out


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
    rho = float(spec.params.get("rho", 28.0))
    gt = {"is_chaotic": 1.0}
    if abs(rho - 28.0) < 1e-12:
        gt["true_lyapunov"] = 0.906
    return Dataset(
        spec,
        _slice(_lorenz_series(spec), spec, burn=1000),
        ground_truth=gt,
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
    lyap = _logistic_lyapunov_from_orbit(x[1000:], r)
    gt = {"is_chaotic": float(lyap > 0.0) if np.isfinite(lyap) else 1.0}
    if np.isfinite(lyap):
        gt["true_lyapunov"] = float(lyap)
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth=gt)


def _logistic_lyapunov_from_orbit(x: np.ndarray, r: float) -> float:
    deriv = np.abs(float(r) * (1.0 - 2.0 * np.asarray(x, dtype=float)))
    deriv = np.maximum(deriv, np.finfo(float).tiny)
    val = float(np.mean(np.log(deriv)))
    return val if np.isfinite(val) else float("nan")


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
    lyap = _henon_largest_lyapunov(a, b, n=total, burn=1000)
    gt = {"is_chaotic": float(lyap > 0.0) if np.isfinite(lyap) else 1.0}
    if np.isfinite(lyap):
        gt["true_lyapunov"] = float(lyap)
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth=gt)


def _henon_largest_lyapunov(a: float, b: float, *, n: int, burn: int) -> float:
    x, y = 0.1, 0.3
    v = np.array([1.0, 0.0], dtype=float)
    acc = 0.0
    count = 0
    for t in range(max(n, burn + 1)):
        jac = np.array([[-2.0 * a * x, 1.0], [b, 0.0]], dtype=float)
        v = jac @ v
        norm = float(np.linalg.norm(v))
        if norm < 1e-12 or not np.isfinite(norm):
            return float("nan")
        v /= norm
        if t >= burn:
            acc += np.log(norm)
            count += 1
        x, y = 1.0 - a * x**2 + y, b * x
    return float(acc / max(count, 1))


def _narma(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    order = int(spec.params["order"])
    total = spec.length + 1000 + order + 2
    u = rng.uniform(0.0, 0.2, size=total)
    y = np.zeros(total, dtype=float)
    for t in range(order, total - 1):
        with np.errstate(over="ignore", invalid="ignore"):
            y_next = (
                0.3 * y[t]
                + 0.05 * y[t] * np.sum(y[t - order + 1 : t + 1])
                + 1.5 * u[t - order + 1] * u[t]
                + 0.1
            )
        y[t + 1] = float(np.clip(y_next, -1e6, 1e6)) if np.isfinite(y_next) else 0.0
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
    degree = max(2, int(spec.params.get("degree", 2)))
    total = spec.length + 1000 + max(lags) + 1
    u = rng.uniform(-1.0, 1.0, size=total)
    y = np.zeros(total, dtype=float)
    for t in range(max(lags), total):
        taps = np.array([u[t - lag] for lag in lags], dtype=float)
        weights = np.linspace(1.0, 0.5, num=taps.size)
        y[t] = float(np.sum(weights * taps**degree) + 0.5 * np.prod(taps[: min(3, taps.size)]))
    y += 0.01 * rng.normal(size=total)
    burn = 1000 + max(lags)
    return Dataset(spec, _slice(y, spec, burn=burn), inputs=_slice(u, spec, burn=burn), ground_truth={"true_memory_order": float(max(lags))})


def _mso8(spec: DatasetSpec) -> Dataset:
    all_freqs = [0.031, 0.047, 0.073, 0.109, 0.151, 0.197, 0.251, 0.317]
    n_freqs = int(spec.params.get("n_freqs", 8))
    freqs = all_freqs[: max(1, min(n_freqs, len(all_freqs)))]
    t = np.arange(spec.length, dtype=float)
    y = sum(np.sin(2.0 * np.pi * f * t) for f in freqs)
    return Dataset(spec, np.asarray(y, dtype=float), ground_truth={"true_n_frequencies": float(len(freqs)), "true_frequencies": freqs})


def _quasi_periodic(spec: DatasetSpec) -> Dataset:
    all_freqs = [0.033, np.sqrt(2.0) / 31.0, np.sqrt(3.0) / 37.0, np.sqrt(5.0) / 43.0, np.sqrt(7.0) / 47.0]
    n_freqs = int(spec.params.get("n_freqs", 3))
    freqs = all_freqs[: max(1, min(n_freqs, len(all_freqs)))]
    t = np.arange(spec.length, dtype=float)
    y = sum(np.sin(2.0 * np.pi * f * t + 0.3 * i) for i, f in enumerate(freqs))
    return Dataset(spec, np.asarray(y, dtype=float), ground_truth={"true_n_frequencies": float(len(freqs)), "true_frequencies": [float(f) for f in freqs]})


def _ar2(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    phi = [float(v) for v in spec.params.get("phi", [0.6, -0.3])]
    p = len(phi)
    total = spec.length + 1000
    x = np.zeros(total, dtype=float)
    eps = rng.normal(0.0, 0.5, size=total)
    for t in range(p, total):
        x[t] = float(np.dot(phi, x[t - np.arange(1, p + 1)])) + eps[t]
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"true_memory_order": float(p)})


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
    return Dataset(spec, _slice(eps, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


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
    return Dataset(spec, x.astype(float), ground_truth={"is_chaotic": 0.0})


def _chirp(spec: DatasetSpec) -> Dataset:
    t = np.linspace(0.0, 1.0, spec.length)
    y = signal.chirp(t, f0=float(spec.params.get("f0", 0.01)) * spec.length, f1=float(spec.params.get("f1", 0.35)) * spec.length, t1=1.0)
    return Dataset(spec, y.astype(float), ground_truth={"is_chaotic": 0.0})


def _regime_switch_ar(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    p_switch = float(spec.params.get("p_switch", 0.015))
    total = spec.length + 1000
    x = np.zeros(total, dtype=float)
    state = 0
    phis = [tuple(float(v) for v in pair) for pair in spec.params.get("phis", [(0.85, -0.15), (-0.55, 0.25)])]
    sigmas = [float(v) for v in spec.params.get("sigmas", [0.35, 0.5])]
    for t in range(2, total):
        if rng.random() < p_switch:
            state = 1 - state
        phi1, phi2 = phis[state]
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + rng.normal(scale=sigmas[state])
    return Dataset(replace(spec, params={**spec.params, "phis": phis, "sigmas": sigmas}), _slice(x, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


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
