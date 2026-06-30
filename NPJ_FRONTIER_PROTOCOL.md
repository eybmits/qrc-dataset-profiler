# NPJ Frontier Protocol v4

Date drafted: 2026-06-29.

Working paper target:

> A regime map of conditional quantum reservoir usefulness across canonical time-series
> process families.

This protocol is the next, higher-evidence design after the completed frontier v3.1 atlas.
It keeps the scientific boundary intact:

- primary comparison: fixed globally calibrated Spin-QRC versus fixed globally calibrated,
  dimension-matched sparse ESN;
- primary claim: conditional, protocol-local QRC usefulness is structured, measurable, and
  predictable from dataset properties;
- not claimed: broad quantum advantage, broad average QRC superiority, or an
  entanglement/coupling mechanism.

## 1. Motivation

The completed v3.1 frontier atlas is already informative:

- 20,000 synthetic property candidates;
- 10,000 evaluated rows, split into 5,000 discovery and 5,000 validation rows;
- 30 Tier-A measured properties;
- prospective validation ROC-AUC `0.8563` and R2 `0.4930`;
- QRC useful in `258 / 5000` validation rows at `qrc_advantage >= 0.05`.

The main weakness is not row-level prediction inside the atlas. The main weakness is
coverage and externality:

- current taxonomy has 9 broad families;
- family-held-out generalization is weak;
- no atlas-support score is reported for new datasets;
- real-world probes are not yet embedded as frozen external interpolation tests;
- rule pockets and top features need formal stability testing.

This v4 protocol fixes those weaknesses without changing the primary result into a broad
quantum-advantage claim.

## 2. Literature-facing justification

The taxonomy should be framed against standard forecasting benchmark practice:

- M4 shows that broad forecasting claims need large, heterogeneous collections of time
  series, not one or two hand-picked tasks: 100,000 series and 61 forecasting methods.
- M5 adds hierarchical retail demand with intermittent, calendar, and sparse-sales
  structure, which motivates explicit sparse-event and seasonal/calendar families.
- The Monash Time Series Forecasting Archive motivates cross-domain real-world external
  probes and feature-based characterization across varied frequencies, lengths, and
  missingness.
- `catch22` motivates a compact, interpretable property-space representation for
  time-series structure; our 30 Tier-A features serve the same role for QRC usefulness.

References:

- M4 Competition: https://doi.org/10.1016/j.ijforecast.2019.04.014
- M5 Accuracy Competition: https://doi.org/10.1016/j.ijforecast.2021.11.013
- Monash Time Series Forecasting Archive: https://arxiv.org/abs/2105.06643
- catch22 canonical features: https://arxiv.org/abs/1901.10200

## 3. Expanded synthetic taxonomy

The v4 atlas uses 16 broad process families plus separate perturbation axes. These are
deliberately broader than concrete generator names, but narrower than vague labels like
"real-world". Boundedness, clipping, missingness, irregular sampling, and corruption are
data-condition axes unless they change the underlying generating mechanism.

| v4 process family | Purpose | Current coverage | New/updated coverage |
|---|---|---|---|
| `chaotic_flow` | continuous deterministic chaos | Lorenz, Rossler, Mackey-Glass, Duffing, Lorenz96, Chua, Henon-Heiles | keep |
| `chaotic_map` | discrete deterministic chaos | logistic, Henon, Ikeda, tent, sine, circle, Lozi, standard, quadratic | keep |
| `delay_dynamics` | delayed feedback and high-memory deterministic systems | Mackey-Glass currently inside `chaotic_flow` | split Mackey-Glass variants; add delayed logistic / delayed AR feedback |
| `input_driven_memory` | reservoir-computing memory benchmarks | NARMA, linear memory, nonlinear IPC, channel equalization | rename from `input_driven`; keep |
| `linear_stochastic` | AR/MA/ARMA baseline stochastic dynamics | AR, ARMA | keep |
| `unit_root_trend` | random walks, drift, integrated processes | ARIMA random walk currently in `nonstationary` | split from `nonstationary` |
| `seasonal_calendar` | seasonal, multiple seasonal, calendar-like structure | seasonal AR and trend-seasonal currently mixed | split; add multiple seasonality and holiday pulses |
| `oscillatory_quasiperiodic` | clean oscillatory and quasi-periodic dynamics | MSO, quasi-periodic, Van der Pol, amplitude modulation, damped oscillator | rename from `oscillatory`; keep |
| `multiscale_composite` | mixtures of slow/fast components and cascades | partial via amplitude modulation/trend-seasonal | add trend + cycles + bursts, wavelet-like composites |
| `long_range` | long memory and scaling | fBM/Hurst variants | keep; add FARIMA-like synthetic variants if cheap |
| `colored_noise` | spectral-noise families | pink, brown, blue noise | keep; extend beta range |
| `nonlinear_autoregressive` | nonlinear stochastic conditional mean | SETAR, bilinear | split from `nonlinear_stochastic`; keep |
| `volatility_heteroskedastic` | conditional variance dynamics | GARCH, EGARCH, stochastic volatility, ARCH | split from `nonlinear_stochastic` |
| `regime_switching` | switching dynamics and structural regimes | regime-switch AR, level shift currently in `nonstationary` | split; add Markov-switch AR |
| `heavy_tail_jump` | rare shocks and jump processes | not explicit | add Student-t AR, compound Poisson jumps, Levy-like innovations |
| `intermittent_sparse` | zero-heavy count/sparse-event sequences | intermittent demand currently in `nonstationary` | split; add hurdle/count variants |

Cross-family perturbation axes:

| v4 perturbation axis | Purpose |
|---|---|
| `observation_noise` | white/colored observation noise at controlled SNR levels |
| `missing_irregular` | missing values, irregular sampling, short/long gaps |
| `quantized_clipped_saturated` | finite precision, physical bounds, clipping, saturation |
| `outlier_spike` | rare additive spikes and measurement faults |
| `downsampled_aliased` | lower sampling rate, aliasing, anti-aliasing controls |
| `time_warped` | nonuniform time deformation and local speed changes |
| `window_length_horizon` | length and forecast-horizon perturbations |

Recommended implementation:

- Keep the current 50 templates as the v3 continuity core.
- Expand to approximately 90-120 synthetic templates across the 16 process families.
- Treat missingness, observation noise, quantization, clipping, downsampling, time-warping,
  and outlier contamination as cross-family perturbation axes.
- Add a new family only when it changes the generating mechanism; otherwise treat the
  variation as a perturbation axis. More labels are not automatically better if they only
  duplicate the same mechanism.
- Do not include Santa Fe or other real data in the primary synthetic taxonomy.

## 3.1. v4 global calibration

The v4 final run must recalibrate QRC and ESN once on a held-out v4 calibration set sampled
from the expanded taxonomy. Do not reuse `results_calibration_v3/frozen_config.json` for
the final NPJ v4 labels, because that calibration was selected on the older 9-family
taxonomy.

Calibration rules:

- use the same held-out calibration rows for QRC and ESN;
- exclude calibration rows from discovery, validation, and real-probe reporting;
- select configurations by mean validation NRMSE only, not by QRC advantage;
- freeze the selected QRC and ESN before generating any v4 discovery/validation labels;
- report the full calibration table and capacity audit, including non-winning QRC variants.

Suggested QRC calibration grid:

- `n_qubits in {6, 8, 10}`;
- `depth in {5, 7}`;
- `virtual_nodes in {5, 7}`, with `7` only when runtime is acceptable;
- `J in {0.8, 1.0, 1.2, 1.5}`;
- `dt in {0.15, 0.20, 0.25}`;
- weak dissipation in `{none, amplitude_damping=0.005, dephasing=0.005}`;
- ring topology;
- no reuploading in the primary protocol.

Suggested ESN calibration grid:

- reservoir size matched to the selected QRC feature dimension;
- `spectral_radius in {0.7, 0.9, 1.0, 1.1, 1.3}`;
- `leak in {0.1, 0.3, 0.6, 1.0}`;
- `input_scale in {0.3, 1.0, 2.0}`;
- density and bias fixed or selected once on the same held-out calibration rows.

This prevents a low QRC win rate from being dismissed as an underpowered-QRC artifact while
keeping the primary comparison fixed-vs-fixed and non-adaptive per dataset.

## 4. Dataset scale

The v4 atlas should have a property-only candidate space large enough to support density
and OOD estimation, and an evaluated subset large enough to stabilize rare useful pockets.

### Recommended NPJ plan

- `50,000` synthetic property candidates.
- `20,000` evaluated synthetic rows:
  - `10,000` discovery rows;
  - `10,000` prospective validation rows.
- Optional camera-ready stress run:
  - evaluate all `50,000` rows only if the 20,000-row results are stable and runtime is
    acceptable.

Why this is better than immediately labeling all 50,000:

- property-only generation is cheap and gives a strong support model;
- 20,000 evaluated rows roughly doubles the current evidence and should yield about
  1,000 useful examples if the useful base rate remains near 5%;
- the remaining 30,000 property-only rows can still define atlas support and sampling
  density;
- full 50,000 labeling can be reserved for final confirmation.

### Aggressive all-in plan

If compute time is acceptable, use:

- `50,000` property candidates;
- `50,000` evaluated labels;
- split frozen as `25,000` discovery and `25,000` validation.

This is scientifically clean because it minimizes selection-weighting concerns. It is not
strictly necessary for the main claim, but it is the strongest local-compute version.

Approximate runtime from the completed v3.1 run:

- 5,000 evaluated rows took about 70 minutes on this machine with 3 seeds and fast lengths.
- 20,000 evaluated rows: roughly 4.5-5.5 hours.
- 50,000 evaluated rows: roughly 11-14 hours.

Every long run must use:

- `--checkpoint-every 100`;
- `caffeinate -disu`;
- separate discovery/validation output folders;
- manifest completion checks before analysis.

## 5. Target-free selection

Selection must be frozen before QRC/ESN labels are computed.

For the 20,000 evaluated-row plan:

- 7,000 broad-balanced rows:
  approximately balanced over v4 families and concrete templates.
- 5,000 feature-space coverage rows:
  maximize coverage in the 30-feature PCA/UMAP/kNN support space.
- 4,000 frontier-enriched rows:
  target-free enrichment around high uncertainty and previously interesting regions:
  high spectral-slope contrast, nonlinear-predictability gaps, volatility persistence,
  trend/nonstationarity boundaries, high entropy, and low/high SNR boundaries.
- 2,000 v3.1-informed stress controls:
  regions predicted to be ESN-dominant or near-tie by the frozen v3.1 map. These are
  predeclared stress rows, not unbiased population rows.
- 2,000 perturbation-axis rows:
  missingness, quantization, clipping, downsampling, outlier contamination, and time warp.

All population-style claims must report both:

- unweighted selected-atlas summaries;
- sampling-weighted summaries using the 50,000-row candidate distribution.

Primary population-style estimates should be based on the broad-balanced and feature-space
coverage rows. Frontier-enriched and v3.1-informed stress rows are for rule discovery,
stress testing, and boundary characterization, not for estimating the unconditional
population rate of QRC usefulness.

## 6. Feature contract

Primary explanatory features remain the 30 Tier-A fields already used in v3.1.

Do not expand the headline feature set just because new descriptors are available. For an
NPJ-level paper, the strongest choice is a frozen, interpretable, compact property space.

Use additional descriptors only as secondary diagnostics:

- catch22-like descriptors as Tier-B appendix features;
- missingness/corruption descriptors for OOD/support diagnostics;
- no target-derived quantities in selection or support scoring.

## 7. Atlas-support and OOD score

Every evaluated or real-world probe dataset must receive an atlas-support score.

Fit support model on discovery rows only:

1. Robust-scale the 30 Tier-A features using discovery medians/IQRs.
2. Remove or regularize near-duplicate / highly collinear directions before distance
   estimation.
3. Compute support distances in both robust-scaled feature space and PCA space, then report
   the more conservative percentile.
4. Compute k-nearest-neighbor distances to discovery rows with `k in {15, 30, 50}`.
5. Estimate local density as inverse mean kNN distance.
6. Cross-fit discovery support distances to define support percentiles.
7. Compute a family-mixture vector from nearest-neighbor family votes.
8. Report:
   - `support_score = 1 - percentile_knn_distance`;
   - `ood_flag = support_score < 0.05`;
   - `family_entropy`;
   - nearest-family mixture;
   - distance to nearest rule pocket.

Prediction confidence should combine:

- support score;
- ensemble prediction variance across bootstrapped meta-models;
- classifier calibration / Brier reliability;
- distance to the nearest stable rule pocket.

New datasets can be interpreted only under these rules:

- high support: interpolation claim allowed;
- medium support: exploratory prediction;
- low support/OOD: report as outside atlas support, no strong prediction.

Report validation performance by support decile. The support score is a validity guardrail,
not proof that a prediction is correct.

## 8. Real-world external probes

Real data are not used to train, select, or recalibrate the synthetic regime map. They are
external interpolation probes after the synthetic model is frozen.

Recommended probe sources:

- Santa Fe laser: nonlinear/chaotic real bridge.
- Monash Forecasting Archive samples: electricity, traffic, solar, weather, exchange,
  tourism, NN5, COVID/deaths where licensing permits.
- M4 samples: micro/macro/industry/demographic/finance/other, multiple frequencies.
- M5 samples: intermittent retail demand with hierarchy/calendar effects.

Probe protocol:

1. Pre-register dataset source, series/window sampling, horizon, and preprocessing.
2. Compute the same 30 features and support score.
3. Embed probes into the synthetic atlas figure.
4. Evaluate frozen QRC and frozen ESN without recalibration.
5. Report only as:
   - in-support interpolation check;
   - boundary/OOD warning;
   - external qualitative validation.

Do not use real probes to tune the QRC, ESN, features, rule thresholds, support score, or
family taxonomy. Real probes strengthen the paper only as post-freeze support-stratified
checks: they are interpolation checks when they fall inside atlas support, and OOD warnings
otherwise.

## 9. Rule-pocket and feature-stability testing

The current v3.1 rule pockets are promising but need formal stability.

Discovery-only procedure:

- Fit meta-models on `B=500` bootstrap resamples for iteration, `B=1000` for final.
- Store feature-rank frequency:
  - top-5 frequency;
  - top-10 frequency;
  - sign consistency;
  - permutation-importance confidence interval;
  - Kendall rank stability across bootstraps.
- Mine shallow rules/rule lists on discovery bootstraps.
- Cluster rules by overlapping thresholds/features.
- Select stable rule pockets before seeing validation labels.

Validation-only reporting:

- For each frozen rule pocket:
  - support `n`;
  - QRC-useful rate and bootstrap CI;
  - enrichment over validation base rate and CI;
  - QRC win rate;
  - mean/median advantage and CI;
  - family composition;
  - support-score distribution;
  - false-discovery correction across rule pockets.

Allowed rule claim:

> These measured-property pockets enrich QRC-useful cases under the frozen protocol.

Not allowed:

> The rule identifies a universal mechanism or guarantees QRC advantage on arbitrary new
> datasets.

## 10. Meta-model suite

Primary:

- gradient-boosted regression for continuous `qrc_advantage`;
- calibrated gradient-boosted classifier for `qrc_useful`;
- EBM/GA2M if available for explainable main effects and pairwise interactions;
- shallow decision trees/rule lists for paper-readable pockets;
- sparse linear/logistic sanity baselines.

Required validation:

- prospective discovery-to-validation score;
- row-stratified CV;
- template/base-generator holdout;
- leave-family-out holdout as a stress test, not the central acceptance criterion;
- support-stratified performance;
- real-probe support-stratified performance.

The central generalization claim is support-aware interpolation in measured property space.
Leave-family-out results reveal extrapolation fragility and should be reported, but they
should not be framed as the main success criterion.

Metrics:

- regression: R2, MAE, decile calibration;
- classification: ROC-AUC, PR-AUC, Brier, calibration curve, precision/recall at selected
  operating points;
- rules: useful-rate enrichment, mean advantage, support, stability frequency.

## 11. Classical robustness baselines

Primary target remains QRC versus matched frozen ESN.

For reviewers, add secondary robustness tables:

- frozen QRC versus validation-tuned ESN;
- frozen QRC versus NVAR/NG-RC with matched lag/feature budget;
- frozen QRC versus linear ridge and GBM lag baselines;
- optional encoding-enhanced QRC versus encoding-matched classical feature maps.

NVAR/NG-RC is required for an NPJ-facing package, not optional. These are not the primary
fairness comparison, but they prevent overclaiming against a single classical reservoir.

## 12. Claim ladder

Minimum claim after v4:

> Across a 50,000-row synthetic candidate atlas spanning canonical time-series regimes,
> QRC usefulness over a matched ESN is rare but structured and prospectively predictable
> from measured dataset properties.

Stronger claim if v4 validates:

> The learned regime map identifies stable, support-aware pockets where fixed Spin-QRC is
> enriched for useful advantage over a matched ESN, and external real-world probes are
> interpretable as interpolation checks when they fall inside atlas support.

Do not claim:

- broad quantum advantage;
- broad average QRC superiority;
- QRC superiority over all classical forecasting models;
- entanglement/coupling mechanism;
- confident extrapolation outside atlas support.

## 13. Acceptance criteria

Before manuscript drafting:

- 50,000 property candidates generated and manifest-verified.
- At least 20,000 evaluated labels completed with checkpoints.
- Expanded v4 taxonomy manifest lists families, templates, and perturbation axes.
- v4 global QRC/ESN calibration is completed on held-out calibration rows and frozen before
  any discovery/validation labels are generated.
- Support/OOD score exists for every synthetic and real probe row.
- Discovery-trained model is scored once on prospective validation.
- Support-stratified validation performance is reported.
- Feature/rule stability bootstrap artifacts are generated.
- NVAR/NG-RC robustness results are reported.
- Real probes are embedded after freezing and reported as support-stratified external checks.
- QRC power audit remains visible.
- Full tests pass.
