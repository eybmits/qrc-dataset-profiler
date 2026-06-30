# QRC Dataset Profiler — Schema Protocol v2 (FROZEN)

Goal: a **unified, scientifically justified protocol** that characterizes *any* univariate
real-valued time series with the **same fields**, so arbitrarily many datasets (synthetic
*and* real) flow into one growable catalog. We then test a fixed **Standard-Spin v1**
quantum reservoir against classical baselines on each dataset and learn which dataset
properties explain when Spin-QRC wins.

This file is the **schema and dataset contract**. `src/qrc_dataset_profiler/spec.py` is its
machine-readable form. Do **not** silently change field names or semantics — add fields and
bump `SCHEMA_VERSION`.

The publication-facing QRC-vs-classical comparison is now frozen separately in
`COMPARISON_PROTOCOL.md` as **Frozen Comparison Protocol v3**. Current legacy/v2 artifacts
were generated before that symmetric v3 freeze; rerun the atlas with
`--comparison-protocol standard_v3` and a held-out calibration config before making
standard-comparison paper claims.

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

## 3. The 50 datasets (first catalog)
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
| 20 | henon_heiles | chaotic_flow | forecast | energy≈0.145 | is_chaotic regime flag |
| 21 | narma2 | input_driven | input_driven | order=2 | true_memory_order=2 |
| 22 | narma5 | input_driven | input_driven | order=5 | true_memory_order=5 |
| 23 | narma30 | input_driven | input_driven | order=30 | true_memory_order=30 |
| 24 | channel_equalization | input_driven | input_driven | nonlinear channel | true_memory_order≈2 |
| 25 | ikeda_map | chaotic_map | forecast | u=0.9 | is_chaotic |
| 26 | tent_map | chaotic_map | forecast | mu≈2 | true_lyapunov, is_chaotic |
| 27 | sine_map | chaotic_map | forecast | a≈1 | is_chaotic |
| 28 | circle_map | chaotic_map | forecast | omega,K | is_chaotic regime flag |
| 29 | lozi_map | chaotic_map | forecast | a,b | is_chaotic |
| 30 | standard_map | chaotic_map | forecast | K | is_chaotic regime flag |
| 31 | quadratic_map | chaotic_map | forecast | a | is_chaotic regime flag |
| 32 | duffing | chaotic_flow | forecast | forced Duffing | is_chaotic regime flag |
| 33 | van_der_pol | oscillatory | forecast | mu | true_n_frequencies≈1 |
| 34 | lorenz96 | chaotic_flow | forecast | F,K | is_chaotic regime flag |
| 35 | chua_circuit | chaotic_flow | forecast | double-scroll circuit | is_chaotic |
| 36 | arma22 | linear_stochastic | forecast | ARMA(2,2) | true_memory_order=2 |
| 37 | arima_random_walk | nonstationary | forecast | unit-root drift | is_chaotic=0 |
| 38 | seasonal_ar | linear_stochastic | forecast | seasonal lag | true_memory_order=season |
| 39 | setar | nonlinear_stochastic | forecast | threshold AR | is_chaotic=0 |
| 40 | egarch | nonlinear_stochastic | forecast | EGARCH volatility | is_chaotic=0 |
| 41 | stochastic_volatility | nonlinear_stochastic | forecast | latent volatility | is_chaotic=0 |
| 42 | bilinear | nonlinear_stochastic | forecast | bilinear AR | is_chaotic=0 |
| 43 | arch | nonlinear_stochastic | forecast | ARCH(1) | is_chaotic=0 |
| 44 | brown_noise | colored_noise | forecast | beta=2 | is_chaotic=0 |
| 45 | blue_noise | colored_noise | forecast | beta=-1 | is_chaotic=0 |
| 46 | amplitude_modulated | oscillatory | forecast | carrier/modulation | true_n_frequencies≈2 |
| 47 | damped_oscillator | oscillatory | forecast | damping,freq | true_n_frequencies≈1 |
| 48 | level_shift | nonstationary | forecast | piecewise level | is_chaotic=0 |
| 49 | intermittent_demand | nonstationary | forecast | sparse bursts | is_chaotic=0 |
| 50 | trend_seasonal | nonstationary | forecast | trend + seasonality | true_n_frequencies≈2 |

Decorators (apply on any base, used for #19 and future rows): observation/dynamic noise at
target SNR; nonstationarity (drift/regime/chirp-warp); downsample/embed.

External validation bridge: Santa Fe laser remains available via `make_real_bridge_specs`
and local `data/SantaFeA.dat` + `data/SantaFeA2.dat` or `data/santa_fe_laser.*`, but it is
not part of the 50 synthetic primary catalog or the default 1000-row synthetic sweep.

## 4. schema v2 — the core property catalog (per record)
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
  8. Long-range: `dfa_alpha` (+`dfa_valid`), `hurst_rs`
  9. Complexity: `perm_entropy`, `sample_entropy`
  10. Predictability: `forecastability` (=1−normalized spectral entropy), `pred_nrmse_gbm`
- **D Ground truth** (synthetic): `true_lyapunov`, `true_memory_order`, `true_n_frequencies`, `true_hurst`, `is_chaotic`
- **E Targets** (filled in Increment 2): `nrmse_linear`, `nrmse_esn_matched`, `nrmse_qrc_spin`, `nrmse_gbm`, `qrc_advantage`

## 4.1 Frontier Feature Tiers

The 1000-row v3 atlas uses the 20 schema-v2 core fields as the primary explanatory
feature set. The planned frontier atlas upgrades the headline feature set to **30 Tier-A
features**: the 20 core fields plus 10 predeclared additions that are universal,
interpretable, and directly relevant to reservoir usefulness. These are frozen before the
frontier run and must not be selected after seeing frontier QRC outcomes.

**Tier-A core fields (20):** `ac_timescale`, `ami_first_min`, `mem_capacity`, `r2_linear`,
`nl_gain`, `snr_db`, `lyapunov`, `zero_one_K`, `spectral_entropy`, `dom_freq`,
`spectral_flatness`, `adf_p`, `kpss_p`, `n_diffs`, `dfa_alpha`, `perm_entropy`,
`sample_entropy`, `hurst_rs`, `forecastability`, `pred_nrmse_gbm`.

**Tier-A frontier additions (10):**

- `predictability_gap_linear_gbm`: nonlinear predictability gap between linear and GBM
  baselines.
- `ext_volatility_ac1`: lag-1 autocorrelation of local volatility.
- `ext_arch_lm5`: ARCH-like volatility clustering statistic.
- `ext_recurrence_rate`: recurrence density in an embedded phase portrait.
- `ext_recurrence_determinism`: diagonal-line determinism of recurrences.
- `ext_psd_slope`: power-spectrum slope.
- `ext_spectral_centroid`: spectral center of mass.
- `ext_trend_strength`: deterministic trend strength.
- `ext_changepoint_count`: simple regime-change count.
- `ext_lz_complexity`: symbolic Lempel-Ziv complexity.

**Tier-B extended descriptors:** `ext_approx_entropy_m2`, `ext_spectral_bandwidth`,
`ext_spectral_rolloff85`, `ext_zero_crossing_rate`, `ext_turning_point_rate`,
`ext_outlier_rate_3sigma`, `ext_spike_rate_mad6`, `ext_seasonality_strength`,
`ext_fnn_fraction`, `ext_corr_dim_approx`, `ext_bds_like`, `ext_zero_fraction`,
`ext_cv2_positive`. Tier-B remains robustness/discovery material only and is not the
headline explanatory feature set.

## 5. Standard-Spin v1 — legacy fixed quantum reservoir (Increment 2)
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

Legacy note: Increment 2-5 artifacts used the original Standard-Spin v1 path with input
injection plus per-layer RZ reuploading and a dimension-matched simple-cycle ESN. The
frozen publication comparison v3 changes the primary comparison to held-out calibrated
fixed QRC versus held-out calibrated fixed sparse random leaky ESN, both with the same
ridge readout protocol. See `COMPARISON_PROTOCOL.md`.

## 6. Increments
- **Increment 1:** package + initial 20 generators (Block D ground truth) + schema-v1
  estimators (Block A–C) + smoke `run_profile` → `catalog.parquet` + correlation/coverage report
  + **ground-truth validation** of estimators. Reservoir/baselines/meta-model = stubs.
- **Increment 2:** Standard-Spin v1 reservoir + baselines + targets (Block E) + meta-model (GBM+SHAP).
- **Increment 5:** expanded to 50 named benchmarks and a 1000-row default sweep; added a
  separate deterministic Tier-B feature table without changing schema-v1 core fields.
