from dataclasses import replace
import math

import numpy as np

from qrc_dataset_profiler.generators import ALL_SPECS, generate
from qrc_dataset_profiler.properties import (
    _adf_p,
    _lyapunov_rosenstein,
    _zero_one_K,
    profile_dataset,
)
from qrc_dataset_profiler.spec import Dataset, DatasetSpec


def test_profile_dataset_runs_on_every_available_spec():
    for spec in ALL_SPECS:
        ds = generate(replace(spec, length=min(spec.length, 1200)))
        if ds.ground_truth.get("_unavailable"):
            continue
        rec = profile_dataset(ds)
        assert rec.name == spec.name
        assert rec.length == min(spec.length, 1200)


def test_white_noise_has_short_memory_and_dfa_near_half():
    rng = np.random.default_rng(123)
    spec = DatasetSpec("white_noise", "test", "synthetic", "forecast", seed=123, length=3000)
    ds = Dataset(spec, rng.normal(size=spec.length))
    rec = profile_dataset(ds)

    assert rec.ac_timescale <= 3
    assert rec.dfa_valid
    assert abs(rec.dfa_alpha - 0.5) <= 0.15


def test_ar2_is_mostly_linear():
    spec = next(s for s in ALL_SPECS if s.name == "ar2")
    rec = profile_dataset(generate(spec))

    assert rec.r2_linear > 0.2
    assert abs(rec.nl_gain) < 0.15


def test_logistic_r4_lyapunov_close_to_ln2():
    spec = next(s for s in ALL_SPECS if s.name == "logistic_r4")
    rec = profile_dataset(generate(spec))

    assert rec.lyapunov_valid
    assert abs(rec.lyapunov - math.log(2.0)) < 0.2


def test_clean_sine_high_forecastability_low_spectral_entropy():
    t = np.arange(3000, dtype=float)
    spec = DatasetSpec("sine", "test", "synthetic", "forecast", seed=7, length=t.size)
    ds = Dataset(spec, np.sin(2 * np.pi * 0.04 * t))
    rec = profile_dataset(ds)

    assert rec.forecastability > 0.75
    assert rec.spectral_entropy < 0.25


# --- regression tests for the four review fixes -------------------------------

def test_lyapunov_no_false_positive_on_pure_sine():
    # C1: a regular (non-chaotic) signal must NOT yield a valid positive exponent.
    t = np.arange(5000, dtype=float)
    val, valid = _lyapunov_rosenstein(np.sin(2 * np.pi * 0.03 * t), dt=1.0)
    assert (not valid) or not (val > 0.0)


def test_chirp_predictive_outputs_bounded():
    # C2: R^2 / nl_gain must be clipped, not exploding to +/-100s on nonstationary data.
    spec = next(s for s in ALL_SPECS if s.name == "chirp")
    rec = profile_dataset(generate(spec))
    assert -1.0 <= rec.r2_linear <= 1.0
    assert -2.0 <= rec.nl_gain <= 2.0


def test_zero_one_chaos_vs_regular():
    # C3: continuous chaotic flow -> K near 1; clean periodic signal -> K near 0.
    for name in ("lorenz63", "rossler"):
        spec = next(s for s in ALL_SPECS if s.name == name)
        k_chaos, valid_chaos = _zero_one_K(np.asarray(generate(spec).series, dtype=float), seed=0)
        assert valid_chaos and k_chaos > 0.5, f"{name} K={k_chaos}"
    t = np.arange(4000, dtype=float)
    k_reg, _ = _zero_one_K(np.sin(2 * np.pi * 0.02 * t), seed=0)
    assert k_reg < 0.1


def test_stationarity_discriminates():
    # C4: a unit-root random walk must be ranked clearly less stationary than AR(2).
    rng = np.random.default_rng(0)
    random_walk = np.cumsum(rng.normal(size=2000))
    ar2 = next(s for s in ALL_SPECS if s.name == "ar2")
    ar_series = np.asarray(generate(ar2).series, dtype=float)
    assert _adf_p(random_walk) > _adf_p(ar_series)
