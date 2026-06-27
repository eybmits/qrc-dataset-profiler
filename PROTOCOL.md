# QRC Dataset Profiler — Protocol v1 (FROZEN)

Goal: a **unified, scientifically justified protocol** that characterizes *any* univariate
real-valued time series with the **same fields**, so arbitrarily many datasets (synthetic
*and* real) flow into one growable catalog. We then test a fixed **Standard-Spin v1**
quantum reservoir against classical baselines on each dataset and learn which dataset
properties explain when Spin-QRC wins.

This file is the **contract**. `src/qrc_dataset_profiler/spec.py` is its machine-readable form.
Do **not** silently change field names or semantics — add fields and bump `SCHEMA_VERSION`.

---

## 1. Inclusion criterion (why these properties and not more)
A property is in the **core schema** only if it is: (1) **universal** — defined on any finite
univariate real series, no required seasonality/input-channel; (2) **robust** — stably
estimable from moderate length, no embedding/threshold free-parameters; (3) **distinct &
interpretable**; (4) **established** (standard estimator). Everything that fails this lives in
the **Tier-B backstop** (`catch22` + `tsfeatures`), computed wholesale as a completeness check,
**not** fed to the explanatory meta-model directly.

## 2. Governance contract
- `schema_version` on every record. New features → append + bump version; never renumber. Old rows keep `NaN`.
- Every fragile estimator carries a `*_valid` bool + obeys a `min_length`. Invalid → `NaN`, never silent garbage.
- All estimators computed on the **train split only**, on the **train-z-standardized** series (mean/σ from train), unless explicitly "on raw" (stationarity block uses raw).
- For `task_type="forecast"`: memory/linearity/nonlinearity/difficulty estimated from **past observations**. For `task_type="input_driven"`: from **past inputs u** (property of the u→y map). Same columns, task-dependent source.
- NaN policy: ground-truth block = `NaN` for real data; `mem_capacity` = `NaN` unless input-driven.
- Determinism: every generator + estimator is seeded.

## 3. The 20 datasets (first catalog)
Synthetic unless marked. Each synthetic generator must expose ground truth where known.

| # | name | family | task_type | key params | ground truth |
|---|------|--------|-----------|-----------|--------------|
| 1 | mackey_glass_t17 | chaotic_flow | forecast | tau=17 | is_chaotic |
| 2 | mackey_glass_t30 | chaotic_flow | forecast | tau=30 | is_chaotic |
| 3 | lorenz63 | chaotic_flow | forecast | rho=28 | is_chaotic, true_lyapunov≈0.906 |
| 4 | rossler | chaotic_flow | forecast | c=5.7 | is_chaotic |
| 5 | logistic_r4 | chaotic_map | forecast | r=4.0 | true_lyapunov=ln2≈0.693, is_chaotic |
| 6 | henon | chaotic_map | forecast | a=1.4,b=0.3 | true_lyapunov≈0.419, is_chaotic |
| 7 | narma10 | input_driven | input_driven | order=10 | true_memory_order=10 |
| 8 | narma20 | input_driven | input_driven | order=20 | true_memory_order=20 |
| 9 | linear_memory | input_driven | input_driven | lag k | true_memory_order=k |
| 10 | nonlinear_ipc | input_driven | input_driven | lags, degree | true_memory_order |
| 11 | mso8 | oscillatory | forecast | 8 incommensurate freqs | true_n_frequencies=8, true_frequencies |
| 12 | quasi_periodic | oscillatory | forecast | 3 freqs | true_n_frequencies, true_frequencies |
| 13 | ar2 | linear_stochastic | forecast | phi=(0.6,-0.3) | true_memory_order=2 |
| 14 | garch11 | nonlinear_stochastic | forecast | alpha,beta | — |
| 15 | fbm_h08 | long_range | forecast | H=0.8 | true_hurst=0.8 |
| 16 | pink_noise | colored_noise | forecast | beta=1 | — |
| 17 | chirp | nonstationary | forecast | f0,f1 | — |
| 18 | regime_switch_ar | nonstationary | forecast | markov | — |
| 19 | lorenz63_noisy | chaotic_flow | forecast | rho=28, snr=10dB | is_chaotic (noise overlay demo) |
| 20 | santa_fe_laser | real_bridge | forecast | dataset A | — (real; loader, may be optional/offline-stub) |

Decorators (apply on any base, used for #19 and future rows): observation/dynamic noise at
target SNR; nonstationarity (drift/regime/chirp-warp); downsample/embed.

## 4. schema v1 — the core property catalog (per record)
All groups → columns of one row. See `spec.py` for exact names/defaults.

- **A Identity**: dataset_id, name, family, source, task_type, params, seed, schema_version, n_channels, length, horizon, missing_frac, irregular_sampling
- **B Basic stats**: mean, std, skew, kurtosis
- **C Axes (10)**
  1. Memory: `ac_timescale`, `ami_first_min`, `mem_capacity`(+`mem_capacity_valid`, input-driven only)
  2. Linearity: `r2_linear`
  3. Nonlinearity: `nl_gain` (= R²_nonlinear − R²_linear, nonlinear = GBM)
  4. Noise: `snr_db` (+`snr_valid`)
  5. Chaos: `lyapunov` (Rosenstein, +`lyapunov_valid`), `zero_one_K` (Gottwald–Melbourne, +`zero_one_valid`)
  6. Frequency: `spectral_entropy`, `dom_freq`, `spectral_flatness`
  7. Stationarity (on raw): `adf_p`, `kpss_p`, `n_diffs`
  8. Long-range: `dfa_alpha` (+`dfa_valid`)
  9. Complexity: `perm_entropy`
  10. Predictability: `forecastability` (=1−normalized spectral entropy), `pred_nrmse_gbm`
- **D Ground truth** (synthetic): `true_lyapunov`, `true_memory_order`, `true_n_frequencies`, `true_hurst`, `is_chaotic`
- **E Targets** (filled in Increment 2): `nrmse_linear`, `nrmse_esn_matched`, `nrmse_qrc_spin`, `nrmse_gbm`, `qrc_advantage`

**Tier-B backstop** (separate table, not in the core record): `catch22` (22 feats) + `tsfeatures`,
computed wholesale; used only as a canonical-correlation completeness check vs the 10 axes.

## 5. Standard-Spin v1 — the fixed quantum reservoir (Increment 2)
Persistent, input-driven **transverse-field Ising** reservoir (genuine statevector, NOT the
mean-field surrogate). H = J·Σ_⟨i,j⟩ Z_i Z_j + h·Σ_i X_i, ring topology.

- N qubits = 8 (dev) / 10 (full); statevector exact.
- Regime: **J = h = 1** (thermal/ergodic, fading memory; avoid integrable & MBL), Trotter dt≈0.25, depth D=5.
- Encoding: RZ(π·u) per qubit + re-uploading per Trotter layer.
- Memory: **persistent state**, re-inject only the input qubit each step (Fujii–Nakajima fading-memory mechanism). NOT windowed/kernel mode.
- Time-multiplexing: V=5–10 virtual nodes (measure at substeps) → features = #observables·V.
- Readout observables: Z + nearest/next-nearest ZZ (+optional ZZZ) → ridge regression (only readout trained).
- Two modes: exact statevector (ideal upper bound) + finite-shot Pauli noise (1024–8192 shots).
- Refs: Fujii & Nakajima, PRApplied 8 024030 (2017); Nakajima et al., PRApplied 11 034021 (2019); Martínez-Peña et al., PRL 127 100502 (2021).

Baselines per dataset: linear/Ridge (AR), **dim-matched leaky ESN** (key control), GBM/MLP.
`qrc_advantage = nrmse_esn_matched − nrmse_qrc_spin`, multi-seed mean + CI.

## 6. Increments
- **Increment 1 (this delegation):** package + 20 generators (Block D ground truth) + schema-v1
  estimators (Block A–C) + smoke `run_profile` → `catalog.parquet` + correlation/coverage report
  + **ground-truth validation** of estimators. Reservoir/baselines/meta-model = stubs.
- **Increment 2:** Standard-Spin v1 reservoir + baselines + targets (Block E) + meta-model (GBM+SHAP).
