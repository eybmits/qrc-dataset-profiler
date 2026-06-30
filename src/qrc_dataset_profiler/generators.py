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
    DatasetSpec("henon_heiles", "chaotic_flow", "synthetic", "forecast", {"generator": "henon_heiles", "energy": 0.145, "dt": 0.04}, seed=120),
    DatasetSpec("narma2", "input_driven", "synthetic", "input_driven", {"generator": "narma", "order": 2}, seed=121),
    DatasetSpec("narma5", "input_driven", "synthetic", "input_driven", {"generator": "narma", "order": 5}, seed=122),
    DatasetSpec("narma30", "input_driven", "synthetic", "input_driven", {"generator": "narma", "order": 30}, seed=123),
    DatasetSpec("channel_equalization", "input_driven", "synthetic", "input_driven", {"generator": "channel_equalization"}, seed=124),
    DatasetSpec("ikeda_map", "chaotic_map", "synthetic", "forecast", {"generator": "ikeda_map", "u": 0.9}, seed=125),
    DatasetSpec("tent_map", "chaotic_map", "synthetic", "forecast", {"generator": "tent_map", "mu": 1.99}, seed=126),
    DatasetSpec("sine_map", "chaotic_map", "synthetic", "forecast", {"generator": "sine_map", "a": 0.99}, seed=127),
    DatasetSpec("circle_map", "chaotic_map", "synthetic", "forecast", {"generator": "circle_map", "omega": 0.31, "K": 1.1}, seed=128),
    DatasetSpec("lozi_map", "chaotic_map", "synthetic", "forecast", {"generator": "lozi_map", "a": 1.7, "b": 0.5}, seed=129),
    DatasetSpec("standard_map", "chaotic_map", "synthetic", "forecast", {"generator": "standard_map", "K": 1.2}, seed=130),
    DatasetSpec("quadratic_map", "chaotic_map", "synthetic", "forecast", {"generator": "quadratic_map", "a": 1.9}, seed=131),
    DatasetSpec("duffing", "chaotic_flow", "synthetic", "forecast", {"generator": "duffing", "gamma": 0.35, "dt": 0.05}, seed=132),
    DatasetSpec("van_der_pol", "oscillatory", "synthetic", "forecast", {"generator": "van_der_pol", "mu": 2.0, "dt": 0.05}, seed=133),
    DatasetSpec("lorenz96", "chaotic_flow", "synthetic", "forecast", {"generator": "lorenz96", "F": 8.0, "K": 8, "dt": 0.02}, seed=134),
    DatasetSpec("chua_circuit", "chaotic_flow", "synthetic", "forecast", {"generator": "chua_circuit", "dt": 0.01}, seed=135),
    DatasetSpec("arma22", "linear_stochastic", "synthetic", "forecast", {"generator": "arma", "phi": [0.5, -0.2], "theta": [0.4, 0.25]}, seed=136),
    DatasetSpec("arima_random_walk", "nonstationary", "synthetic", "forecast", {"generator": "arima_random_walk", "drift": 0.003}, seed=137),
    DatasetSpec("seasonal_ar", "linear_stochastic", "synthetic", "forecast", {"generator": "seasonal_ar", "season": 24}, seed=138),
    DatasetSpec("setar", "nonlinear_stochastic", "synthetic", "forecast", {"generator": "setar"}, seed=139),
    DatasetSpec("egarch", "nonlinear_stochastic", "synthetic", "forecast", {"generator": "egarch"}, seed=140),
    DatasetSpec("stochastic_volatility", "nonlinear_stochastic", "synthetic", "forecast", {"generator": "stochastic_volatility"}, seed=141),
    DatasetSpec("bilinear", "nonlinear_stochastic", "synthetic", "forecast", {"generator": "bilinear"}, seed=142),
    DatasetSpec("arch", "nonlinear_stochastic", "synthetic", "forecast", {"generator": "arch"}, seed=143),
    DatasetSpec("brown_noise", "colored_noise", "synthetic", "forecast", {"generator": "colored_noise", "beta": 2.0}, seed=144),
    DatasetSpec("blue_noise", "colored_noise", "synthetic", "forecast", {"generator": "colored_noise", "beta": -1.0}, seed=145),
    DatasetSpec("amplitude_modulated", "oscillatory", "synthetic", "forecast", {"generator": "amplitude_modulated"}, seed=146),
    DatasetSpec("damped_oscillator", "oscillatory", "synthetic", "forecast", {"generator": "damped_oscillator"}, seed=147),
    DatasetSpec("level_shift", "nonstationary", "synthetic", "forecast", {"generator": "level_shift"}, seed=148),
    DatasetSpec("intermittent_demand", "nonstationary", "synthetic", "forecast", {"generator": "intermittent_demand"}, seed=149),
    DatasetSpec("trend_seasonal", "nonstationary", "synthetic", "forecast", {"generator": "trend_seasonal"}, seed=150),
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
        "channel_equalization": _channel_equalization,
        "ikeda_map": _ikeda_map,
        "tent_map": _tent_map,
        "sine_map": _sine_map,
        "circle_map": _circle_map,
        "lozi_map": _lozi_map,
        "standard_map": _standard_map,
        "quadratic_map": _quadratic_map,
        "duffing": _duffing,
        "van_der_pol": _van_der_pol,
        "lorenz96": _lorenz96,
        "chua_circuit": _chua_circuit,
        "henon_heiles": _henon_heiles,
        "arma": _arma,
        "arima_random_walk": _arima_random_walk,
        "seasonal_ar": _seasonal_ar,
        "setar": _setar,
        "egarch": _egarch,
        "stochastic_volatility": _stochastic_volatility,
        "bilinear": _bilinear,
        "arch": _arch,
        "amplitude_modulated": _amplitude_modulated,
        "damped_oscillator": _damped_oscillator,
        "level_shift": _level_shift,
        "intermittent_demand": _intermittent_demand,
        "trend_seasonal": _trend_seasonal,
        "multiscale_composite": _multiscale_composite,
        "student_t_ar": _student_t_ar,
        "jump_ar": _jump_ar,
    }
    key = str(spec.params.get("generator", spec.name))
    if key not in dispatch:
        raise ValueError(f"unknown dataset spec: {spec.name}")
    ds = dispatch[key](spec)
    if spec.params.get("noise_overlay") and not ds.ground_truth.get("_unavailable"):
        noisy = add_observation_noise(ds.series, float(spec.params["snr_db"]), _rng(spec))
        ds = Dataset(spec, noisy, inputs=ds.inputs, ground_truth=dict(ds.ground_truth))
    if spec.params.get("perturbation_axes") and not ds.ground_truth.get("_unavailable"):
        ds = _apply_perturbation_axes(ds)
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
        "henon_heiles": 2700,
        "channel_equalization": 2800,
        "ikeda_map": 2900,
        "tent_map": 3000,
        "sine_map": 3100,
        "circle_map": 3200,
        "lozi_map": 3300,
        "standard_map": 3400,
        "quadratic_map": 3500,
        "duffing": 3600,
        "van_der_pol": 3700,
        "lorenz96": 3800,
        "chua_circuit": 3900,
        "arma": 4000,
        "arima_random_walk": 4100,
        "seasonal_ar": 4200,
        "setar": 4300,
        "egarch": 4400,
        "stochastic_volatility": 4500,
        "bilinear": 4600,
        "arch": 4700,
        "amplitude_modulated": 4800,
        "damped_oscillator": 4900,
        "level_shift": 5000,
        "intermittent_demand": 5100,
        "trend_seasonal": 5200,
        "santa_fe_laser": 5300,
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
            ("channel_equalization", 2),
            ("ikeda_map", 1),
            ("tent_map", 1),
            ("sine_map", 1),
            ("circle_map", 2),
            ("lozi_map", 2),
            ("standard_map", 1),
            ("quadratic_map", 1),
            ("duffing", 2),
            ("van_der_pol", 1),
            ("lorenz96", 1),
            ("chua_circuit", 1),
            ("henon_heiles", 2),
            ("arma", 2),
            ("arima_random_walk", 2),
            ("seasonal_ar", 2),
            ("setar", 2),
            ("egarch", 2),
            ("stochastic_volatility", 2),
            ("bilinear", 2),
            ("arch", 1),
            ("amplitude_modulated", 2),
            ("damped_oscillator", 2),
            ("level_shift", 2),
            ("intermittent_demand", 2),
            ("trend_seasonal", 2),
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

    for i, row in enumerate(samples["channel_equalization"]):
        noise_std = float(_scale(row[0], 0.005, 0.08))
        drive_scale = float(_scale(row[1], 0.7, 1.3))
        add(
            f"channel_equalization_n{noise_std:.3f}_s{i:03d}",
            "input_driven",
            "input_driven",
            {"generator": "channel_equalization", "noise_std": noise_std, "drive_scale": drive_scale},
            i,
        )

    for i, u in enumerate(samples["ikeda_map"][:, 0]):
        val = float(_scale(u, 0.70, 0.90))
        add(f"ikeda_u{val:.3f}_s{i:03d}", "chaotic_map", "forecast", {"generator": "ikeda_map", "u": val}, i)

    for i, u in enumerate(samples["tent_map"][:, 0]):
        mu = float(_scale(u, 1.55, 1.999))
        add(f"tent_mu{mu:.3f}_s{i:03d}", "chaotic_map", "forecast", {"generator": "tent_map", "mu": mu}, i)

    for i, u in enumerate(samples["sine_map"][:, 0]):
        a = float(_scale(u, 0.78, 0.999))
        add(f"sine_map_a{a:.3f}_s{i:03d}", "chaotic_map", "forecast", {"generator": "sine_map", "a": a}, i)

    for i, row in enumerate(samples["circle_map"]):
        omega = float(_scale(row[0], 0.27, 0.48))
        kval = float(_scale(row[1], 0.3, 1.7))
        add(f"circle_map_k{kval:.2f}_s{i:03d}", "chaotic_map", "forecast", {"generator": "circle_map", "omega": omega, "K": kval}, i)

    for i, row in enumerate(samples["lozi_map"]):
        a = float(_scale(row[0], 1.45, 1.85))
        b = float(_scale(row[1], 0.25, 0.65))
        add(f"lozi_a{a:.2f}_b{b:.2f}_s{i:03d}", "chaotic_map", "forecast", {"generator": "lozi_map", "a": a, "b": b}, i)

    for i, u in enumerate(samples["standard_map"][:, 0]):
        kval = float(_scale(u, 0.7, 2.2))
        add(f"standard_map_k{kval:.2f}_s{i:03d}", "chaotic_map", "forecast", {"generator": "standard_map", "K": kval}, i)

    for i, u in enumerate(samples["quadratic_map"][:, 0]):
        a = float(_scale(u, 1.35, 1.99))
        add(f"quadratic_map_a{a:.3f}_s{i:03d}", "chaotic_map", "forecast", {"generator": "quadratic_map", "a": a}, i)

    for i, row in enumerate(samples["duffing"]):
        gamma = float(_scale(row[0], 0.22, 0.48))
        omega = float(_scale(row[1], 0.9, 1.4))
        add(f"duffing_g{gamma:.2f}_w{omega:.2f}_s{i:03d}", "chaotic_flow", "forecast", {"generator": "duffing", "gamma": gamma, "omega": omega, "dt": 0.05}, i)

    for i, u in enumerate(samples["van_der_pol"][:, 0]):
        mu = float(_scale(u, 0.5, 5.0))
        add(f"van_der_pol_mu{mu:.2f}_s{i:03d}", "oscillatory", "forecast", {"generator": "van_der_pol", "mu": mu, "dt": 0.05}, i)

    for i, u in enumerate(samples["lorenz96"][:, 0]):
        F = float(_scale(u, 5.0, 12.0))
        add(f"lorenz96_F{F:.2f}_s{i:03d}", "chaotic_flow", "forecast", {"generator": "lorenz96", "F": F, "K": 8, "dt": 0.02}, i)

    for i, u in enumerate(samples["chua_circuit"][:, 0]):
        alpha = float(_scale(u, 8.0, 12.0))
        add(f"chua_alpha{alpha:.2f}_s{i:03d}", "chaotic_flow", "forecast", {"generator": "chua_circuit", "alpha": alpha, "dt": 0.01}, i)

    for i, row in enumerate(_lhs(n * 2, 2, seed + 91)):
        energy = float(_scale(row[0], 0.105, 0.162))
        px0 = float(_scale(row[1], -0.12, 0.12))
        add(
            f"henon_heiles_E{energy:.3f}_s{i:03d}",
            "chaotic_flow",
            "forecast",
            {"generator": "henon_heiles", "energy": energy, "px0": px0, "dt": 0.04},
            i,
        )

    for i, row in enumerate(samples["arma"]):
        phi = _stable_ar_coeffs(2, seed * 20_000 + i)
        theta = [float(_scale(row[0], -0.65, 0.65)), float(_scale(row[1], -0.45, 0.45))]
        add(f"arma22_s{i:03d}", "linear_stochastic", "forecast", {"generator": "arma", "phi": phi, "theta": theta}, i)

    for i, row in enumerate(samples["arima_random_walk"]):
        drift = float(_scale(row[0], -0.01, 0.01))
        sigma = float(_scale(row[1], 0.2, 0.9))
        add(f"arima_rw_d{drift:.3f}_s{i:03d}", "nonstationary", "forecast", {"generator": "arima_random_walk", "drift": drift, "sigma": sigma}, i)

    for i, row in enumerate(samples["seasonal_ar"]):
        season = int(round(_scale(row[0], 12, 48)))
        seasonal_phi = float(_scale(row[1], 0.35, 0.9))
        add(f"seasonal_ar_s{season:02d}_{i:03d}", "linear_stochastic", "forecast", {"generator": "seasonal_ar", "season": season, "seasonal_phi": seasonal_phi}, i)

    for i, row in enumerate(samples["setar"]):
        threshold = float(_scale(row[0], -0.4, 0.4))
        high_phi = float(_scale(row[1], 0.45, 0.9))
        add(f"setar_t{threshold:.2f}_s{i:03d}", "nonlinear_stochastic", "forecast", {"generator": "setar", "threshold": threshold, "high_phi": high_phi}, i)

    for i, row in enumerate(samples["egarch"]):
        alpha = float(_scale(row[0], 0.04, 0.22))
        beta = float(_scale(row[1], 0.65, 0.95))
        add(f"egarch_a{alpha:.2f}_b{beta:.2f}_s{i:03d}", "nonlinear_stochastic", "forecast", {"generator": "egarch", "alpha": alpha, "beta": beta}, i)

    for i, row in enumerate(samples["stochastic_volatility"]):
        phi = float(_scale(row[0], 0.82, 0.985))
        sigma_eta = float(_scale(row[1], 0.08, 0.35))
        add(f"stoch_vol_p{phi:.3f}_s{i:03d}", "nonlinear_stochastic", "forecast", {"generator": "stochastic_volatility", "phi": phi, "sigma_eta": sigma_eta}, i)

    for i, row in enumerate(samples["bilinear"]):
        a = float(_scale(row[0], 0.1, 0.65))
        b = float(_scale(row[1], -0.7, 0.7))
        add(f"bilinear_a{a:.2f}_b{b:.2f}_s{i:03d}", "nonlinear_stochastic", "forecast", {"generator": "bilinear", "a": a, "b": b}, i)

    for i, u in enumerate(samples["arch"][:, 0]):
        alpha = float(_scale(u, 0.2, 0.85))
        add(f"arch_a{alpha:.2f}_s{i:03d}", "nonlinear_stochastic", "forecast", {"generator": "arch", "alpha": alpha}, i)

    for i, row in enumerate(samples["amplitude_modulated"]):
        carrier = float(_scale(row[0], 0.025, 0.12))
        mod = float(_scale(row[1], 0.002, 0.02))
        add(f"am_mod_c{carrier:.3f}_m{mod:.3f}_s{i:03d}", "oscillatory", "forecast", {"generator": "amplitude_modulated", "carrier_freq": carrier, "mod_freq": mod}, i)

    for i, row in enumerate(samples["damped_oscillator"]):
        freq = float(_scale(row[0], 0.015, 0.10))
        damping = float(_scale(row[1], 0.0002, 0.006))
        add(f"damped_f{freq:.3f}_d{damping:.4f}_s{i:03d}", "oscillatory", "forecast", {"generator": "damped_oscillator", "freq": freq, "damping": damping}, i)

    for i, row in enumerate(samples["level_shift"]):
        n_shifts = 1 + int(np.floor(row[0] * 4.0))
        n_shifts = min(n_shifts, 4)
        magnitude = float(_scale(row[1], 0.7, 2.5))
        add(f"level_shift_k{n_shifts}_s{i:03d}", "nonstationary", "forecast", {"generator": "level_shift", "n_shifts": n_shifts, "magnitude": magnitude}, i)

    for i, row in enumerate(samples["intermittent_demand"]):
        p_event = float(_scale(row[0], 0.02, 0.22))
        burst_scale = float(_scale(row[1], 0.7, 3.0))
        add(f"intermittent_p{p_event:.2f}_s{i:03d}", "nonstationary", "forecast", {"generator": "intermittent_demand", "p_event": p_event, "burst_scale": burst_scale}, i)

    for i, row in enumerate(samples["trend_seasonal"]):
        slope = float(_scale(row[0], -0.002, 0.004))
        season = int(round(_scale(row[1], 16, 72)))
        add(f"trend_seasonal_s{season:02d}_{i:03d}", "nonstationary", "forecast", {"generator": "trend_seasonal", "slope": slope, "season": season}, i)

    specs.extend(_noise_overlay_specs(specs, n_per_family=n, seed=seed))
    return specs


V4_PROCESS_FAMILIES: tuple[str, ...] = (
    "chaotic_flow",
    "chaotic_map",
    "delay_dynamics",
    "input_driven_memory",
    "linear_stochastic",
    "unit_root_trend",
    "seasonal_calendar",
    "oscillatory_quasiperiodic",
    "multiscale_composite",
    "long_range",
    "colored_noise",
    "nonlinear_autoregressive",
    "volatility_heteroskedastic",
    "regime_switching",
    "heavy_tail_jump",
    "intermittent_sparse",
)

V4_PERTURBATION_AXES: tuple[str, ...] = (
    "observation_noise",
    "missing_irregular",
    "quantized_clipped_saturated",
    "outlier_spike",
    "downsampled_aliased",
    "time_warped",
    "window_length_horizon",
)


def make_sweep_specs_v4(n_per_template: int = 500, seed: int = 0) -> list[DatasetSpec]:
    """Build the support-aware v4 synthetic taxonomy.

    The v4 atlas keeps the existing 50-template continuity core but relabels it into
    the paper-facing 16 process families. It then adds extra standard regimes and
    cross-family perturbation-axis rows. ``n_per_template=500`` yields roughly 50k
    property candidates.
    """

    n = max(1, int(n_per_template))
    seed = int(seed)
    specs: list[DatasetSpec] = [_v4_relabel_spec(spec) for spec in make_sweep_specs(n, seed=seed)]
    specs.extend(_v4_extra_specs(n_per_template=n, seed=seed))
    specs.extend(_v4_perturbation_specs(specs, n_per_template=n, seed=seed))
    return specs


def _v4_relabel_spec(spec: DatasetSpec) -> DatasetSpec:
    family = _v4_family_for_params(spec.params, spec.family)
    params = {**spec.params, "taxonomy": "v4", "v4_source": "continuity_core"}
    return replace(spec, family=family, params=params)


def _v4_family_for_params(params: dict[str, object], fallback: str) -> str:
    generator = str(params.get("generator", ""))
    if generator == "mackey_glass":
        return "delay_dynamics"
    if generator in {"narma", "linear_memory", "nonlinear_ipc", "channel_equalization"}:
        return "input_driven_memory"
    if generator in {"mso", "quasi_periodic", "van_der_pol", "amplitude_modulated", "damped_oscillator", "chirp"}:
        return "oscillatory_quasiperiodic"
    if generator == "arima_random_walk":
        return "unit_root_trend"
    if generator == "seasonal_ar":
        return "seasonal_calendar"
    if generator == "trend_seasonal":
        return "multiscale_composite"
    if generator in {"setar", "bilinear"}:
        return "nonlinear_autoregressive"
    if generator in {"garch", "egarch", "stochastic_volatility", "arch"}:
        return "volatility_heteroskedastic"
    if generator in {"regime_switch_ar", "level_shift"}:
        return "regime_switching"
    if generator == "intermittent_demand":
        return "intermittent_sparse"
    if fallback == "input_driven":
        return "input_driven_memory"
    if fallback == "oscillatory":
        return "oscillatory_quasiperiodic"
    if fallback == "nonstationary":
        return "regime_switching"
    if fallback == "nonlinear_stochastic":
        return "nonlinear_autoregressive"
    return fallback


def _v4_extra_specs(*, n_per_template: int, seed: int) -> list[DatasetSpec]:
    n = max(1, int(n_per_template))
    out: list[DatasetSpec] = []

    def add(name: str, family: str, task_type: str, params: dict[str, object], i: int, *, offset: int) -> None:
        out.append(
            DatasetSpec(
                name,
                family,
                "synthetic",
                task_type,
                {**params, "taxonomy": "v4", "v4_source": "expanded_template"},
                seed=seed * 100_000 + offset + i,
            )
        )

    delayed = _lhs(n * 2, 2, seed + 601)
    for i, row in enumerate(delayed):
        tau = int(round(_scale(row[0], 8, 70)))
        dt = float(_scale(row[1], 0.7, 1.3))
        add(f"v4_mackey_glass_tau{tau:02d}_dt{dt:.2f}_s{i:03d}", "delay_dynamics", "forecast", {"generator": "mackey_glass", "tau": tau, "dt": dt}, i, offset=61_000)

    seasonal = _lhs(n * 2, 2, seed + 602)
    for i, row in enumerate(seasonal):
        season = int(round(_scale(row[0], 7, 168)))
        seasonal_phi = float(_scale(row[1], 0.25, 0.93))
        add(f"v4_seasonal_ar_s{season:03d}_p{seasonal_phi:.2f}_{i:03d}", "seasonal_calendar", "forecast", {"generator": "seasonal_ar", "season": season, "seasonal_phi": seasonal_phi}, i, offset=62_000)

    multiscale = _lhs(n * 3, 4, seed + 603)
    for i, row in enumerate(multiscale):
        params = {
            "generator": "multiscale_composite",
            "slow_period": int(round(_scale(row[0], 96, 512))),
            "fast_period": int(round(_scale(row[1], 8, 48))),
            "burst_rate": float(_scale(row[2], 0.005, 0.07)),
            "trend": float(_scale(row[3], -0.0025, 0.004)),
        }
        add(f"v4_multiscale_p{params['slow_period']}_{params['fast_period']}_{i:03d}", "multiscale_composite", "forecast", params, i, offset=63_000)

    unit_root = _lhs(n * 2, 2, seed + 604)
    for i, row in enumerate(unit_root):
        drift = float(_scale(row[0], -0.02, 0.02))
        sigma = float(_scale(row[1], 0.1, 1.2))
        add(f"v4_unit_root_d{drift:.3f}_s{i:03d}", "unit_root_trend", "forecast", {"generator": "arima_random_walk", "drift": drift, "sigma": sigma}, i, offset=64_000)

    heavy = _lhs(n * 3, 3, seed + 605)
    for i, row in enumerate(heavy):
        if i % 2 == 0:
            params = {
                "generator": "student_t_ar",
                "phi": float(_scale(row[0], -0.75, 0.92)),
                "df": float(_scale(row[1], 2.2, 8.0)),
                "sigma": float(_scale(row[2], 0.2, 0.9)),
            }
        else:
            params = {
                "generator": "jump_ar",
                "phi": float(_scale(row[0], -0.5, 0.9)),
                "jump_prob": float(_scale(row[1], 0.003, 0.08)),
                "jump_scale": float(_scale(row[2], 1.0, 6.0)),
            }
        add(f"v4_{params['generator']}_{i:03d}", "heavy_tail_jump", "forecast", params, i, offset=65_000)

    intermittent = _lhs(n, 2, seed + 606)
    for i, row in enumerate(intermittent):
        p_event = float(_scale(row[0], 0.005, 0.18))
        burst_scale = float(_scale(row[1], 0.5, 5.0))
        add(f"v4_intermittent_p{p_event:.3f}_{i:03d}", "intermittent_sparse", "forecast", {"generator": "intermittent_demand", "p_event": p_event, "burst_scale": burst_scale}, i, offset=66_000)

    colored = _lhs(n * 2, 1, seed + 607)
    for i, row in enumerate(colored):
        beta = float(_scale(row[0], -1.5, 2.5))
        add(f"v4_colored_noise_b{beta:.3f}_{i:03d}", "colored_noise", "forecast", {"generator": "colored_noise", "beta": beta}, i, offset=67_000)

    long_range = _lhs(n * 2, 1, seed + 608)
    for i, row in enumerate(long_range):
        H = float(_scale(row[0], 0.52, 0.97))
        add(f"v4_fbm_h{H:.3f}_{i:03d}", "long_range", "forecast", {"generator": "fbm", "H": H}, i, offset=68_000)

    nonlinear = _lhs(n * 2, 2, seed + 609)
    for i, row in enumerate(nonlinear):
        if i % 2 == 0:
            params = {"generator": "setar", "threshold": float(_scale(row[0], -0.6, 0.6)), "high_phi": float(_scale(row[1], 0.35, 0.95))}
        else:
            params = {"generator": "bilinear", "a": float(_scale(row[0], 0.05, 0.75)), "b": float(_scale(row[1], -0.9, 0.9))}
        add(f"v4_{params['generator']}_{i:03d}", "nonlinear_autoregressive", "forecast", params, i, offset=69_000)

    volatility = _lhs(n * 2, 2, seed + 610)
    for i, row in enumerate(volatility):
        if i % 3 == 0:
            params = {"generator": "garch", "omega": 0.05, "alpha": float(_scale(row[0], 0.02, 0.25)), "beta": float(_scale(row[1], 0.45, 0.92))}
        elif i % 3 == 1:
            params = {"generator": "egarch", "alpha": float(_scale(row[0], 0.03, 0.24)), "beta": float(_scale(row[1], 0.60, 0.96))}
        else:
            params = {"generator": "arch", "alpha": float(_scale(row[0], 0.15, 0.9))}
        add(f"v4_{params['generator']}_vol_{i:03d}", "volatility_heteroskedastic", "forecast", params, i, offset=71_000)

    linear = _lhs(n, 2, seed + 611)
    for i, row in enumerate(linear):
        phi = _stable_ar_coeffs(2, seed * 30_000 + i)
        theta = [float(_scale(row[0], -0.7, 0.7)), float(_scale(row[1], -0.55, 0.55))]
        add(f"v4_arma22_{i:03d}", "linear_stochastic", "forecast", {"generator": "arma", "phi": phi, "theta": theta}, i, offset=72_000)

    return out


def _v4_perturbation_specs(specs: list[DatasetSpec], *, n_per_template: int, seed: int) -> list[DatasetSpec]:
    n = max(1, int(n_per_template))
    axes = list(V4_PERTURBATION_AXES)
    base = [spec for spec in specs if spec.source == "synthetic" and not spec.params.get("perturbation_axes")]
    if not base:
        return []
    limit = min(len(base), n * len(axes) * 4)
    selected = base[:limit]
    out: list[DatasetSpec] = []
    for j, spec in enumerate(selected):
        axis = axes[j % len(axes)]
        strength = 0.25 + 0.75 * ((j % n) / max(n - 1, 1))
        params = {**spec.params, "perturbation_axes": [axis], "perturbation_strength": float(strength), "v4_source": "perturbation_axis"}
        if axis == "observation_noise":
            params["snr_db"] = float(_scale(strength, 4.0, 24.0))
        if axis == "window_length_horizon":
            horizon = 1 + int(round(_scale(strength, 1, 12)))
            out.append(replace(spec, name=f"{spec.name}_v4_{axis}", params=params, horizon=horizon, seed=seed * 100_000 + 70_000 + j))
        else:
            out.append(replace(spec, name=f"{spec.name}_v4_{axis}", params=params, seed=seed * 100_000 + 70_000 + j))
    return out


def make_real_bridge_specs(n_windows: int = 40, seed: int = 0) -> list[DatasetSpec]:
    """Build optional real-data validation windows kept outside the synthetic atlas."""

    return _santa_fe_sweep_specs(n_windows=n_windows, seed=seed, offset=5300)


def add_observation_noise(series: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white observation noise at a target signal-to-noise ratio in dB."""

    x = np.asarray(series, dtype=float)
    signal_power = float(np.var(x))
    if signal_power <= 0:
        return x.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(noise_power), size=x.shape)


def _apply_perturbation_axes(ds: Dataset) -> Dataset:
    axes = tuple(str(axis) for axis in ds.spec.params.get("perturbation_axes", ()))
    if not axes:
        return ds
    rng = _rng(ds.spec)
    strength = float(ds.spec.params.get("perturbation_strength", 0.5))

    def transform(arr: np.ndarray, *, target: bool) -> np.ndarray:
        out = np.asarray(arr, dtype=float).copy()
        for axis in axes:
            if axis == "observation_noise":
                snr = float(ds.spec.params.get("snr_db", _scale(strength, 4.0, 24.0)))
                out = add_observation_noise(out, snr, rng)
            elif axis == "missing_irregular":
                out = _impute_missing_irregular(out, frac=float(_scale(strength, 0.02, 0.18)), rng=rng)
            elif axis == "quantized_clipped_saturated":
                out = _quantize_clip(out, levels=max(8, int(round(_scale(1.0 - strength, 10, 96)))))
            elif axis == "outlier_spike":
                out = _add_spikes(out, rate=float(_scale(strength, 0.002, 0.035)), scale=float(_scale(strength, 2.0, 8.0)), rng=rng)
            elif axis == "downsampled_aliased":
                out = _downsample_alias(out, factor=max(2, int(round(_scale(strength, 2, 8)))))
            elif axis == "time_warped":
                out = _time_warp(out, strength=float(_scale(strength, 0.03, 0.25)))
            elif axis == "window_length_horizon":
                out = out
            else:
                raise ValueError(f"unknown perturbation axis: {axis}")
        return out if target else out[: len(arr)]

    series = transform(ds.series, target=True)
    inputs = transform(ds.inputs, target=False) if ds.inputs is not None else None
    return Dataset(ds.spec, series, inputs=inputs, ground_truth=dict(ds.ground_truth))


def _impute_missing_irregular(x: np.ndarray, *, frac: float, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    if y.size < 4:
        return y
    mask = rng.random(y.size) < float(np.clip(frac, 0.0, 0.8))
    if mask.all():
        mask[rng.integers(0, y.size)] = False
    idx = np.arange(y.size)
    y[mask] = np.interp(idx[mask], idx[~mask], y[~mask])
    return y


def _quantize_clip(x: np.ndarray, *, levels: int) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    lo, hi = np.nanquantile(y, [0.02, 0.98])
    if not np.isfinite(lo + hi) or hi <= lo:
        return y
    y = np.clip(y, lo, hi)
    q = max(2, int(levels))
    return np.round((y - lo) / (hi - lo) * (q - 1)) / (q - 1) * (hi - lo) + lo


def _add_spikes(x: np.ndarray, *, rate: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    if y.size == 0:
        return y
    mask = rng.random(y.size) < float(np.clip(rate, 0.0, 0.5))
    amp = float(np.nanstd(y))
    if amp <= 0 or not np.isfinite(amp):
        amp = 1.0
    y[mask] += rng.choice([-1.0, 1.0], size=int(mask.sum())) * amp * float(scale)
    return y


def _downsample_alias(x: np.ndarray, *, factor: int) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    if y.size < 4:
        return y.copy()
    step = max(2, int(factor))
    coarse_idx = np.arange(0, y.size, step)
    if coarse_idx[-1] != y.size - 1:
        coarse_idx = np.r_[coarse_idx, y.size - 1]
    idx = np.arange(y.size)
    return np.interp(idx, coarse_idx, y[coarse_idx])


def _time_warp(x: np.ndarray, *, strength: float) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    if y.size < 4:
        return y.copy()
    t = np.linspace(0.0, 1.0, y.size)
    warp = t + float(strength) * np.sin(2.0 * np.pi * t)
    warp = np.maximum.accumulate(np.clip(warp, 0.0, 1.0))
    if warp[-1] <= warp[0]:
        return y.copy()
    warp = (warp - warp[0]) / (warp[-1] - warp[0])
    return np.interp(t, warp, y)


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
    overlay_generators = {"mackey_glass", "lorenz63", "logistic", "linear_memory", "mso", "fbm", "chirp"}
    selected = [s for s in specs if s.params.get("generator") in overlay_generators]
    selected = selected[: min(len(selected), max(6, n_per_family * 7))]
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


def _santa_fe_sweep_specs(*, n_windows: int, seed: int, offset: int) -> list[DatasetSpec]:
    """Optional real-bridge Santa Fe windows for external validation."""

    count = max(1, int(n_windows))
    max_start = 10093 - 800
    starts = np.linspace(0, max_start, num=count, dtype=int)
    out: list[DatasetSpec] = []
    for i, start in enumerate(starts):
        out.append(
            DatasetSpec(
                f"santa_fe_laser_w{i:03d}",
                "real_bridge",
                "real",
                "forecast",
                {"generator": "santa_fe_laser", "dataset": "A", "window_start": int(start)},
                seed=seed * 100_000 + offset + i,
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


def _rk4(rhs: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, *, dt: float, n: int) -> np.ndarray:
    out = np.empty((n, y0.size), dtype=float)
    state = np.asarray(y0, dtype=float).copy()
    t = 0.0
    for i in range(n):
        out[i] = state
        k1 = rhs(t, state)
        k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = rhs(t + dt, state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e8:
            state = y0.astype(float).copy()
        t += dt
    return out


def _duffing(spec: DatasetSpec) -> Dataset:
    delta = float(spec.params.get("delta", 0.2))
    alpha = float(spec.params.get("alpha", -1.0))
    beta = float(spec.params.get("beta", 1.0))
    gamma = float(spec.params.get("gamma", 0.35))
    omega = float(spec.params.get("omega", 1.2))
    dt = float(spec.params.get("dt", 0.05))
    n = spec.length + 1000

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        x, v = state
        return np.array([v, -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)], dtype=float)

    series = _rk4(rhs, np.array([0.1, 0.0]), dt=dt, n=n)[:, 0]
    return Dataset(spec, _slice(series, spec, burn=1000), ground_truth={"is_chaotic": float(gamma > 0.3)})


def _van_der_pol(spec: DatasetSpec) -> Dataset:
    mu = float(spec.params.get("mu", 2.0))
    dt = float(spec.params.get("dt", 0.05))
    n = spec.length + 1000

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        x, v = state
        return np.array([v, mu * (1.0 - x**2) * v - x], dtype=float)

    series = _rk4(rhs, np.array([0.2, 0.0]), dt=dt, n=n)[:, 0]
    return Dataset(spec, _slice(series, spec, burn=1000), ground_truth={"is_chaotic": 0.0, "true_n_frequencies": 1.0})


def _lorenz96(spec: DatasetSpec) -> Dataset:
    F = float(spec.params.get("F", 8.0))
    K = int(spec.params.get("K", 8))
    dt = float(spec.params.get("dt", 0.02))
    n = spec.length + 1000
    y0 = np.full(K, F, dtype=float)
    y0[0] += 0.01

    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

    series = _rk4(rhs, y0, dt=dt, n=n)[:, 0]
    return Dataset(spec, _slice(series, spec, burn=1000), ground_truth={"is_chaotic": float(F > 5.5)})


def _chua_circuit(spec: DatasetSpec) -> Dataset:
    alpha = float(spec.params.get("alpha", 10.0))
    beta = float(spec.params.get("beta", 14.87))
    m0 = float(spec.params.get("m0", -1.143))
    m1 = float(spec.params.get("m1", -0.714))
    dt = float(spec.params.get("dt", 0.01))
    n = spec.length + 1000

    def h(x: float) -> float:
        return m1 * x + 0.5 * (m0 - m1) * (abs(x + 1.0) - abs(x - 1.0))

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array([alpha * (y - x - h(x)), x - y + z, -beta * y], dtype=float)

    series = _rk4(rhs, np.array([0.1, 0.0, 0.0]), dt=dt, n=n)[:, 0]
    return Dataset(spec, _slice(series, spec, burn=1000), ground_truth={"is_chaotic": 1.0})


def _henon_heiles(spec: DatasetSpec) -> Dataset:
    energy = float(spec.params.get("energy", 0.145))
    dt = float(spec.params.get("dt", 0.04))
    x0 = float(spec.params.get("x0", 0.0))
    y0 = float(spec.params.get("y0", 0.1))
    px0 = float(spec.params.get("px0", 0.03))
    potential = 0.5 * (x0**2 + y0**2) + x0**2 * y0 - (y0**3) / 3.0
    py_sq = max(2.0 * (energy - potential - 0.5 * px0**2), 1e-8)
    py0 = float(np.sqrt(py_sq))
    n = spec.length + 1000

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        x, y, px, py = state
        return np.array([px, py, -x - 2.0 * x * y, -y - x**2 + y**2], dtype=float)

    series = _rk4(rhs, np.array([x0, y0, px0, py0], dtype=float), dt=dt, n=n)[:, 0]
    return Dataset(spec, _slice(series, spec, burn=1000), ground_truth={"is_chaotic": float(energy >= 0.12)})


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


def _ikeda_map(spec: DatasetSpec) -> Dataset:
    u = float(spec.params.get("u", 0.9))
    total = spec.length + 1000
    x = np.empty(total, dtype=float)
    y = np.empty(total, dtype=float)
    x[0], y[0] = 0.1, 0.1
    for t in range(total - 1):
        tau = 0.4 - 6.0 / (1.0 + x[t] ** 2 + y[t] ** 2)
        x[t + 1] = 1.0 + u * (x[t] * np.cos(tau) - y[t] * np.sin(tau))
        y[t + 1] = u * (x[t] * np.sin(tau) + y[t] * np.cos(tau))
        if not np.isfinite(x[t + 1] + y[t + 1]):
            x[t + 1], y[t + 1] = 0.1, 0.1
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"is_chaotic": 1.0})


def _tent_map(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    mu = float(spec.params.get("mu", 1.99))
    total = spec.length + 1000
    x = np.empty(total, dtype=float)
    x[0] = rng.uniform(0.11, 0.89)
    for t in range(total - 1):
        val = mu * min(x[t], 1.0 - x[t])
        x[t + 1] = float(np.clip(val, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
    gt = {"is_chaotic": float(mu > 1.0)}
    if mu > 0:
        gt["true_lyapunov"] = float(np.log(mu))
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth=gt)


def _sine_map(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    a = float(spec.params.get("a", 0.99))
    total = spec.length + 1000
    x = np.empty(total, dtype=float)
    x[0] = rng.uniform(0.11, 0.89)
    for t in range(total - 1):
        x[t + 1] = float(np.clip(a * np.sin(np.pi * x[t]), np.finfo(float).eps, 1.0 - np.finfo(float).eps))
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"is_chaotic": float(a > 0.86)})


def _circle_map(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    omega = float(spec.params.get("omega", 0.31))
    K = float(spec.params.get("K", 1.1))
    total = spec.length + 1000
    theta = np.empty(total, dtype=float)
    theta[0] = rng.uniform(0.0, 1.0)
    for t in range(total - 1):
        theta[t + 1] = (theta[t] + omega - (K / (2.0 * np.pi)) * np.sin(2.0 * np.pi * theta[t])) % 1.0
    return Dataset(spec, _slice(theta, spec, burn=1000), ground_truth={"is_chaotic": float(K > 1.0)})


def _lozi_map(spec: DatasetSpec) -> Dataset:
    a = float(spec.params.get("a", 1.7))
    b = float(spec.params.get("b", 0.5))
    total = spec.length + 1000
    x = np.empty(total, dtype=float)
    y = np.empty(total, dtype=float)
    x[0], y[0] = 0.1, 0.1
    for t in range(total - 1):
        x[t + 1] = 1.0 - a * abs(x[t]) + b * y[t]
        y[t + 1] = x[t]
        if not np.isfinite(x[t + 1] + y[t + 1]) or abs(x[t + 1]) > 1e6:
            x[t + 1], y[t + 1] = 0.1, 0.1
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"is_chaotic": 1.0})


def _standard_map(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    K = float(spec.params.get("K", 1.2))
    total = spec.length + 1000
    theta = np.empty(total, dtype=float)
    p = np.empty(total, dtype=float)
    theta[0] = rng.uniform(0.0, 2.0 * np.pi)
    p[0] = rng.uniform(-0.1, 0.1)
    for t in range(total - 1):
        p[t + 1] = (p[t] + K * np.sin(theta[t])) % (2.0 * np.pi)
        theta[t + 1] = (theta[t] + p[t + 1]) % (2.0 * np.pi)
    return Dataset(spec, _slice(np.sin(theta), spec, burn=1000), ground_truth={"is_chaotic": float(K > 0.9)})


def _quadratic_map(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    a = float(spec.params.get("a", 1.9))
    total = spec.length + 1000
    x = np.empty(total, dtype=float)
    x[0] = rng.uniform(-0.4, 0.4)
    for t in range(total - 1):
        val = 1.0 - a * x[t] ** 2
        x[t + 1] = float(np.clip(val, -1.5, 1.5))
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"is_chaotic": float(a > 1.4)})


def _narma(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    order = int(spec.params["order"])
    nonlinear_coeff = 0.05 if order <= 10 else (0.01 if order <= 20 else 0.004)
    total = spec.length + 1000 + order + 2
    u = rng.uniform(0.0, 0.2, size=total)
    y = np.zeros(total, dtype=float)
    for t in range(order, total - 1):
        with np.errstate(over="ignore", invalid="ignore"):
            y_next = (
                0.3 * y[t]
                + nonlinear_coeff * y[t] * np.sum(y[t - order + 1 : t + 1])
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


def _channel_equalization(spec: DatasetSpec) -> Dataset:
    """Classic nonlinear channel-equalization style input-driven benchmark."""

    rng = _rng(spec)
    total = spec.length + 1000 + 10
    drive_scale = float(spec.params.get("drive_scale", 1.0))
    noise_std = float(spec.params.get("noise_std", 0.02))
    symbols = rng.choice(np.array([-3.0, -1.0, 1.0, 3.0]), size=total) * drive_scale
    q = np.zeros(total, dtype=float)
    for t in range(2, total):
        q[t] = (
            0.08 * symbols[t]
            - 0.12 * symbols[t - 1]
            + 1.00 * symbols[t - 2]
            + 0.18 * symbols[t - 1] ** 2
            - 0.10 * symbols[t - 2] ** 3
        )
    y = q + 0.036 * q**2 - 0.011 * q**3 + noise_std * rng.normal(size=total)
    burn = 1000 + 10
    return Dataset(spec, _slice(y, spec, burn=burn), inputs=_slice(symbols, spec, burn=burn), ground_truth={"true_memory_order": 2.0})


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


def _amplitude_modulated(spec: DatasetSpec) -> Dataset:
    carrier = float(spec.params.get("carrier_freq", 0.05))
    mod = float(spec.params.get("mod_freq", 0.006))
    depth = float(spec.params.get("depth", 0.55))
    t = np.arange(spec.length, dtype=float)
    envelope = 1.0 + depth * np.sin(2.0 * np.pi * mod * t)
    y = envelope * np.sin(2.0 * np.pi * carrier * t)
    return Dataset(spec, np.asarray(y, dtype=float), ground_truth={"true_n_frequencies": 2.0, "is_chaotic": 0.0})


def _damped_oscillator(spec: DatasetSpec) -> Dataset:
    freq = float(spec.params.get("freq", 0.04))
    damping = float(spec.params.get("damping", 0.002))
    t = np.arange(spec.length, dtype=float)
    y = np.exp(-damping * t) * np.sin(2.0 * np.pi * freq * t)
    return Dataset(spec, np.asarray(y, dtype=float), ground_truth={"true_n_frequencies": 1.0, "is_chaotic": 0.0})


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


def _arma(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    phi = [float(v) for v in spec.params.get("phi", [0.5, -0.2])]
    theta = [float(v) for v in spec.params.get("theta", [0.4, 0.25])]
    p = len(phi)
    q = len(theta)
    total = spec.length + 1000
    x = np.zeros(total, dtype=float)
    eps = rng.normal(0.0, float(spec.params.get("sigma", 0.5)), size=total)
    for t in range(max(p, q), total):
        ar = float(np.dot(phi, x[t - np.arange(1, p + 1)])) if p else 0.0
        ma = float(np.dot(theta, eps[t - np.arange(1, q + 1)])) if q else 0.0
        x[t] = ar + ma + eps[t]
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"true_memory_order": float(max(p, q)), "is_chaotic": 0.0})


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


def _arch(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    omega = float(spec.params.get("omega", 0.1))
    alpha = float(spec.params.get("alpha", 0.5))
    total = spec.length + 1000
    eps = np.zeros(total, dtype=float)
    z = rng.normal(size=total)
    var = np.full(total, omega / max(1.0 - min(alpha, 0.95), 0.05))
    for t in range(1, total):
        var[t] = omega + alpha * eps[t - 1] ** 2
        eps[t] = np.sqrt(max(var[t], 1e-12)) * z[t]
    return Dataset(spec, _slice(eps, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


def _egarch(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    omega = float(spec.params.get("omega", -0.2))
    alpha = float(spec.params.get("alpha", 0.12))
    beta = float(spec.params.get("beta", 0.85))
    gamma = float(spec.params.get("gamma", -0.08))
    total = spec.length + 1000
    z = rng.normal(size=total)
    log_var = np.zeros(total, dtype=float)
    eps = np.zeros(total, dtype=float)
    ez_abs = np.sqrt(2.0 / np.pi)
    for t in range(1, total):
        log_var[t] = omega + beta * log_var[t - 1] + alpha * (abs(z[t - 1]) - ez_abs) + gamma * z[t - 1]
        log_var[t] = float(np.clip(log_var[t], -12.0, 8.0))
        eps[t] = np.exp(0.5 * log_var[t]) * z[t]
    return Dataset(spec, _slice(eps, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


def _stochastic_volatility(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    phi = float(spec.params.get("phi", 0.95))
    sigma_eta = float(spec.params.get("sigma_eta", 0.2))
    total = spec.length + 1000
    h = np.zeros(total, dtype=float)
    y = np.zeros(total, dtype=float)
    for t in range(1, total):
        h[t] = phi * h[t - 1] + sigma_eta * rng.normal()
        y[t] = np.exp(0.5 * np.clip(h[t], -12.0, 8.0)) * rng.normal()
    return Dataset(spec, _slice(y, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


def _bilinear(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    a = float(spec.params.get("a", 0.35))
    b = float(spec.params.get("b", 0.3))
    total = spec.length + 1000
    x = np.zeros(total, dtype=float)
    eps = rng.normal(0.0, 0.5, size=total)
    for t in range(1, total):
        x[t] = a * x[t - 1] + b * x[t - 1] * eps[t - 1] + eps[t]
        x[t] = float(np.clip(x[t], -1e4, 1e4))
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


def _setar(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    threshold = float(spec.params.get("threshold", 0.0))
    low_phi = float(spec.params.get("low_phi", -0.55))
    high_phi = float(spec.params.get("high_phi", 0.75))
    total = spec.length + 1000
    x = np.zeros(total, dtype=float)
    for t in range(1, total):
        phi = high_phi if x[t - 1] > threshold else low_phi
        x[t] = phi * x[t - 1] + rng.normal(scale=0.4)
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


def _arima_random_walk(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    drift = float(spec.params.get("drift", 0.0))
    sigma = float(spec.params.get("sigma", 0.5))
    total = spec.length + 1000
    increments = drift + rng.normal(scale=sigma, size=total)
    x = np.cumsum(increments)
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


def _seasonal_ar(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    season = max(2, int(spec.params.get("season", 24)))
    phi = float(spec.params.get("phi", 0.35))
    seasonal_phi = float(spec.params.get("seasonal_phi", 0.65))
    total = spec.length + 1000 + season
    x = rng.normal(scale=0.2, size=total)
    for t in range(season, total):
        x[t] = phi * x[t - 1] + seasonal_phi * x[t - season] + rng.normal(scale=0.35)
    return Dataset(spec, _slice(x, spec, burn=1000 + season), ground_truth={"true_memory_order": float(season), "is_chaotic": 0.0})


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


def _level_shift(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    n_shifts = max(1, int(spec.params.get("n_shifts", 2)))
    magnitude = float(spec.params.get("magnitude", 1.5))
    total = spec.length + 1000
    x = rng.normal(scale=0.25, size=total)
    levels = np.zeros(total, dtype=float)
    shift_points = np.linspace(1000, total - 1, num=n_shifts + 2, dtype=int)[1:-1]
    level = 0.0
    sign = 1.0
    for point in shift_points:
        level += sign * magnitude * rng.uniform(0.7, 1.3)
        sign *= -1.0
        levels[point:] = level
    return Dataset(spec, _slice(x + levels, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


def _intermittent_demand(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    p_event = float(spec.params.get("p_event", 0.08))
    burst_scale = float(spec.params.get("burst_scale", 1.5))
    total = spec.length + 1000
    events = rng.random(total) < p_event
    sizes = rng.gamma(shape=2.0, scale=burst_scale, size=total)
    x = np.where(events, sizes, 0.0)
    kernel = np.array([1.0, 0.45, 0.2], dtype=float)
    y = np.convolve(x, kernel, mode="full")[:total] + 0.02 * rng.normal(size=total)
    return Dataset(spec, _slice(y, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


def _trend_seasonal(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    slope = float(spec.params.get("slope", 0.001))
    season = max(2, int(spec.params.get("season", 48)))
    total = spec.length + 1000
    t = np.arange(total, dtype=float)
    y = slope * t + np.sin(2.0 * np.pi * t / season) + 0.35 * np.sin(4.0 * np.pi * t / season)
    y += rng.normal(scale=0.15, size=total)
    return Dataset(spec, _slice(y, spec, burn=1000), ground_truth={"true_n_frequencies": 2.0, "is_chaotic": 0.0})


def _multiscale_composite(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    total = spec.length + 1000
    t = np.arange(total, dtype=float)
    slow = max(8, int(spec.params.get("slow_period", 256)))
    fast = max(2, int(spec.params.get("fast_period", 24)))
    trend = float(spec.params.get("trend", 0.001))
    burst_rate = float(spec.params.get("burst_rate", 0.02))
    envelope = 1.0 + 0.4 * np.sin(2.0 * np.pi * t / max(slow, 1))
    y = (
        trend * t
        + envelope * np.sin(2.0 * np.pi * t / fast)
        + 0.55 * np.sin(2.0 * np.pi * t / slow)
        + 0.25 * np.sin(2.0 * np.pi * t / max(3, int(np.sqrt(slow * fast))))
    )
    events = rng.random(total) < burst_rate
    if np.any(events):
        kernel = np.exp(-np.arange(32, dtype=float) / 7.0)
        bursts = np.convolve(events.astype(float) * rng.gamma(2.0, 1.0, size=total), kernel, mode="full")[:total]
        y += bursts
    y += rng.normal(scale=0.12, size=total)
    return Dataset(spec, _slice(y, spec, burn=1000), ground_truth={"true_n_frequencies": 3.0, "is_chaotic": 0.0})


def _student_t_ar(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    phi = float(spec.params.get("phi", 0.6))
    df = max(2.05, float(spec.params.get("df", 3.5)))
    sigma = float(spec.params.get("sigma", 0.5))
    total = spec.length + 1000
    x = np.zeros(total, dtype=float)
    eps = rng.standard_t(df=df, size=total) * sigma / np.sqrt(df / max(df - 2.0, 1e-6))
    for t in range(1, total):
        x[t] = phi * x[t - 1] + eps[t]
        x[t] = float(np.clip(x[t], -1e5, 1e5))
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


def _jump_ar(spec: DatasetSpec) -> Dataset:
    rng = _rng(spec)
    phi = float(spec.params.get("phi", 0.55))
    jump_prob = float(spec.params.get("jump_prob", 0.02))
    jump_scale = float(spec.params.get("jump_scale", 3.0))
    total = spec.length + 1000
    x = np.zeros(total, dtype=float)
    base = rng.normal(scale=0.35, size=total)
    jumps = (rng.random(total) < jump_prob) * rng.normal(scale=jump_scale, size=total)
    for t in range(1, total):
        x[t] = phi * x[t - 1] + base[t] + jumps[t]
        x[t] = float(np.clip(x[t], -1e5, 1e5))
    return Dataset(spec, _slice(x, spec, burn=1000), ground_truth={"is_chaotic": 0.0})


def _santa_fe_laser(spec: DatasetSpec) -> Dataset:
    data_dir = Path.cwd() / "data"
    candidates = sorted(data_dir.glob("santa_fe_laser.*"))
    canonical_parts = [data_dir / "SantaFeA.dat", data_dir / "SantaFeA2.dat"]
    start = max(0, int(spec.params.get("window_start", spec.params.get("start", 0))))
    if not candidates and all(path.exists() for path in canonical_parts):
        try:
            parts = [np.loadtxt(path) for path in canonical_parts]
            arr = np.concatenate([np.asarray(part, dtype=float).reshape(-1) for part in parts])
            arr = arr[np.isfinite(arr)]
            if arr.size >= start + spec.length:
                return Dataset(spec, arr[start : start + spec.length], ground_truth={})
        except Exception:
            pass
    for path in candidates:
        try:
            if path.suffix == ".npy":
                arr = np.load(path)
            else:
                arr = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
            arr = np.asarray(arr, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size >= start + spec.length:
                return Dataset(spec, arr[start : start + spec.length], ground_truth={})
        except Exception:
            continue
    return Dataset(spec, np.asarray([], dtype=float), ground_truth={"_unavailable": True})
