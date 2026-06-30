# Frontier Atlas Protocol v3.1

Date drafted: 2026-06-29.

This protocol defines the next paper-grade atlas. It extends the current `standard_v3`
fixed-vs-fixed fairness run without changing the scientific boundary:

> The goal is to learn where a fixed, defensible spin-QRC protocol is relatively useful
> compared with matched classical reservoirs. The goal is not to claim broad quantum
> advantage or an entanglement mechanism.

`COMPARISON_PROTOCOL.md` freezes the model-comparison logic. `PROTOCOL.md` freezes the
dataset/schema contract. This file freezes the expanded frontier atlas design: dataset
scale, 30-feature explanatory schema, validation layers, and runtime staging.

## 1. Current anchor state

The existing v3 anchor run is a valid fast fairness atlas:

- `results_sweep_v3/sweep_catalog.csv`: 1000 synthetic rows.
- Primary synthetic base library: 50 named synthetic generators.
- Santa Fe laser: external real-data bridge only; not part of the primary synthetic atlas.
- Existing feature set: 20 schema-v2 core dataset-property fields.
- Existing extended table: 24 exploratory descriptors in
  `results_features_v3/extended_features_sweep.csv`.
- Existing model comparison: globally calibrated fixed QRC versus globally calibrated fixed
  sparse random leaky ESN, no per-dataset reservoir tuning for either.
- Existing fast frozen config: 6 qubits, feature dimension 60, 3 seeds, no RZ reuploading.

The fast v3 anchor is suitable for method development and for estimating signal strength.
The frontier atlas below is the next paper-facing expansion.

## 2. Primary question

For a fixed reservoir-computing protocol, which measured properties of a univariate time
series predict:

1. positive QRC advantage over the matched ESN,
2. practically useful QRC advantage, and
3. membership in stable, interpretable high-usefulness regions of dataset-property space?

The primary target remains:

`qrc_advantage = nrmse_esn_matched - nrmse_qrc_spin`

Use the same thresholds unless a later preregistered sensitivity analysis changes them:

- `qrc_win`: `qrc_advantage > 0.0`
- `qrc_useful`: `qrc_advantage >= 0.05`
- `near_tie`: `abs(qrc_advantage) < 0.05`
- `esn_preferred`: `qrc_advantage <= -0.05`

## 3. Model-comparison layers

No single comparison is fair under every possible reviewer framing, so the frontier atlas
uses three explicitly separated layers.

### Layer 1: primary fixed-vs-fixed fairness

This is the headline comparison.

- QRC is calibrated once on held-out synthetic calibration datasets and then frozen.
- ESN is calibrated once on the same held-out calibration datasets and then frozen.
- Both models use the same train/validation/test splits.
- Both models use the same ridge-readout protocol.
- No per-dataset reservoir hyperparameter tuning is allowed for either model.
- Reservoir feature dimensions are matched exactly.

Recommended publication configuration:

- QRC: transverse-field Ising spin reservoir, persistent state, input-qubit injection,
  no per-layer RZ reuploading in the primary model.
- QRC size: 8 qubits, 5 virtual nodes, single-Z plus nearest-neighbor ZZ observables,
  feature dimension 80.
- QRC dissipation: fixed weak damping/dephasing only if selected once on the calibration
  set; otherwise coherent dynamics is reported as the frozen selected protocol.
- ESN: sparse random leaky ESN with reservoir size 80, density 0.1, ridge readout.
- Seeds: 5 seeds for final paper tables if runtime permits; 3 seeds for iteration.

The current fast v3 anchor uses the same fairness logic at 6 qubits and feature dimension
60. It remains the fast iteration target.

### Layer 2: strong-classical stress test

This answers whether QRC remains useful when the classical reservoir receives a stronger
budget.

- Frozen QRC from Layer 1.
- Validation-tuned sparse random leaky ESN on each dataset.
- Same feature dimension as QRC.
- Report as conservative against QRC, not as the symmetric fairness result.

### Layer 3: matched tuning-budget robustness

This answers whether the conclusion depends on the no-tuning freeze.

- Either both QRC and ESN receive no per-dataset reservoir tuning, or both receive the same
  small validation-grid budget.
- The matched grid must be declared before seeing test outcomes.
- Report the result as sensitivity to tuning budget.

## 4. Encoding and mechanism controls

The primary model keeps the input encoding simple so that any learned relationship is easier
to attribute to reservoir dynamics rather than to an engineered nonlinear encoding.

Required controls:

- Coupling attribution: paired coupled-QRC versus `J=0`, identical feature dimensions,
  matched seeds, same readout protocol, family-stratified confidence intervals.
- Encoding control: primary input injection versus input injection plus RZ reuploading.
- Encoding-matched classical control: add NVAR/NG-RC and/or Fourier-feature ridge/ESN when
  reporting reuploading results, because repeated encoding can itself create nonlinear
  feature maps.
- Dissipation control: coherent QRC versus fixed weak dissipation when the final QRC uses
  dissipation.
- Null visibility: negative and null family-level mechanism results must remain visible.

Allowed claim if positive:

> Coupling/dissipation/encoding contributes positively inside this frozen protocol.

Not allowed:

> We have shown broad quantum advantage, entanglement advantage, or a universal quantum
> mechanism for time-series forecasting.

## 5. Dataset expansion

The expanded atlas remains synthetic-only in the primary analysis.

Base library:

- 50 named synthetic generators from `PROTOCOL.md`.
- No Santa Fe rows in the primary atlas.
- Santa Fe may be evaluated later as an external validation bridge.

Candidate pool:

- Generate 10000 to 20000 synthetic candidate datasets from the 50 base generators.
- Candidate generation varies seeds and generator parameters, plus declared decorators:
  noise, drift/regime changes, chirp/time-warping, downsampling, and embedding where
  scientifically appropriate.
- Compute dataset properties for candidates before QRC/ESN target evaluation.
- Selection into the expensive evaluated atlas must depend only on generator identity and
  measured properties, never on `qrc_advantage`.

Evaluated atlas sizes:

- Fast iteration: 2500 evaluated rows.
- Paper frontier: 5000 evaluated rows.
- Optional final stress run: 10000 evaluated rows only after the 5000-row analysis is stable.

Recommended 5000-row paper-frontier allocation:

- 1000 anchor rows: reproduce the existing v3 synthetic design for continuity.
- 2000 broad-balanced rows: approximately balanced over the 50 base generators and families.
- 1250 frontier rows: enriched around property regions suggested by v3, especially high
  persistence/long-range scaling, nonlinear-predictability gaps, entropy/noise boundaries,
  and stationarity boundaries.
- 750 negative/boundary controls: regions where v3 predicts ESN dominance or near-ties.

Because frontier rows are enriched, every aggregate table must report:

- raw atlas frequencies,
- family-stratified summaries,
- sampling-weighted estimates when making population-style claims, and
- separate performance on broad-balanced versus frontier-enriched subsets.

## 6. Thirty Tier-A ground-rule features

The headline meta-model uses exactly 30 Tier-A ground-rule features: 20 core schema-v2
fields plus 10 predeclared upgrades. This feature set is frozen before the frontier
outcomes are computed and should be treated as the primary explanatory coordinate system
for the atlas.

Core 20:

- `ac_timescale`
- `ami_first_min`
- `mem_capacity`
- `r2_linear`
- `nl_gain`
- `snr_db`
- `lyapunov`
- `zero_one_K`
- `spectral_entropy`
- `dom_freq`
- `spectral_flatness`
- `adf_p`
- `kpss_p`
- `n_diffs`
- `dfa_alpha`
- `perm_entropy`
- `sample_entropy`
- `hurst_rs`
- `forecastability`
- `pred_nrmse_gbm`

Tier-A upgrades:

- `predictability_gap_linear_gbm`
- `ext_volatility_ac1`
- `ext_arch_lm5`
- `ext_recurrence_rate`
- `ext_recurrence_determinism`
- `ext_psd_slope`
- `ext_spectral_centroid`
- `ext_trend_strength`
- `ext_changepoint_count`
- `ext_lz_complexity`

Tier-B descriptors remain exploratory only. They may be used for appendix discovery, but
they are not the headline explanatory model unless a future protocol explicitly refreezes
the feature set.

Implementation note: `predictability_gap_linear_gbm` must be materialized as a deterministic
column before the frontier run. The current code already declares the field, but the joined
frontier feature table must compute and export it.

## 7. Meta-models and validation

The meta-model is explanatory and predictive. It should identify stable regions of
dataset-property space, not merely maximize leaderboard accuracy.

Primary model suite:

- Explainable Boosting Machine / GA2M if available: primary explanatory model for additive
  effects plus selected pairwise interactions among the 30 features.
- Gradient-boosted tree regressor: primary predictive model for continuous
  `qrc_advantage`.
- Calibrated gradient-boosted classifier: primary predictive model for `qrc_useful`.
- Explainable shallow decision tree or rule list: human-readable useful-region extraction.
- Sparse linear/logistic model: sanity-check baseline for whether the signal requires
  nonlinear interactions.

Transformer-style tabular models are not the headline meta-model for this atlas. With 30
hand-engineered numerical features and 5000 to 10000 evaluated rows, boosted trees and
EBM/GA2M models are usually stronger, easier to validate, and more interpretable. A tabular
transformer, TabNet, or MLP may be added only as an appendix robustness benchmark after the
explanatory models are frozen.

Required metrics:

- Regression: grouped-CV R2, MAE, calibration of predicted advantage by decile.
- Classification: ROC-AUC, PR-AUC, precision/recall at fixed useful-rate thresholds, Brier
  score, reliability curve.
- Rule quality: support, useful rate, win rate, mean advantage, bootstrap confidence
  intervals, and leave-generator-family stability.

Required validation splits:

- Row-stratified CV for comparability to current v3.
- Grouped CV holding out base generator names.
- Leave-family-out validation.
- Broad-balanced holdout separate from frontier-enriched rows.
- External bridge only after the synthetic model is frozen.

Anti-circularity suites:

- all 30 Tier-A features,
- without `r2_linear`,
- without direct predictability proxies (`r2_linear`, `forecastability`, `pred_nrmse_gbm`,
  `predictability_gap_linear_gbm`),
- dynamics/complexity-only subset,
- no-target-baseline subset excluding any feature derived from fitted forecast models.

## 8. Claim gates

The following gates must pass before writing strong paper language about dataset
categorization:

- The useful-region signal persists under grouped CV and leave-family-out validation.
- Useful-region lower bootstrap confidence bound is clearly above the global useful rate.
- PR-AUC improves meaningfully over the base-rate classifier, since useful cases are rare.
- Important features are stable across bootstrap resamples and anti-circularity suites.
- The same qualitative rules appear in broad-balanced rows, not only enriched frontier rows.
- Family-level conclusions include confidence intervals and do not hide ESN-dominant
  families.

Mechanism language has stricter gates:

- Coupled-vs-`J=0` paired effect positive overall and positive in the relevant useful region.
- Mechanism effect survives matched seeds, identical feature dimensions, and family
  stratification.
- Encoding and dissipation controls do not explain away the effect.

Even if these pass, the claim is still protocol-local, not a broad quantum-advantage claim.

## 9. Runtime staging on the current machine

Current machine observed by the local checkout:

- CPU: Apple M4.
- Logical CPUs: 10.
- Existing v3 anchor shapes: 1000 sweep rows, 1000 extended-feature rows, 1000 paired
  attribution rows.

Empirical anchor for rough planning:

- 1000-row fast fixed-vs-fixed v3 sweep at 6 qubits and 3 seeds: about 20 to 30 minutes.
- 1000-row extended-feature table: about 1 to 3 minutes.
- 1000-row analysis, meta-model, maps, and plots: about 2 to 5 minutes.
- 1000-row paired coupled-vs-`J=0` attribution at 6 qubits and 1 seed: about 7 to 10 minutes.

Projected iteration costs:

| Stage | Scope | Expected wall time |
|---|---:|---:|
| Code smoke | 100 to 200 rows | 3 to 8 min |
| Fast atlas | 1000 evaluated rows | 25 to 40 min |
| Frontier iteration | 2500 evaluated rows | 60 to 100 min |
| Paper frontier | 5000 evaluated rows | 2 to 3.5 h |
| 5000-row attribution | coupled-vs-`J=0`, 1 seed | 35 to 60 min |
| 5000-row full plots/meta | 30-feature analysis | 10 to 25 min |
| Candidate pool | 10000 to 20000 property-only rows | 1 to 4 h, benchmark first |

The 8-qubit publication rerun is expected to be substantially slower than the 6-qubit fast
run. A conservative planning multiplier is 3x to 5x for QRC-heavy parts:

| Stage | 8-qubit publication mode |
|---|---:|
| 1000 evaluated rows | 1.5 to 2.5 h |
| 2500 evaluated rows | 4 to 6 h |
| 5000 evaluated rows | 8 to 12 h |
| 5000-row attribution | 2 to 4 h |

Therefore the recommended workflow is:

1. Implement the 30-feature joined table and candidate-selection code.
2. Benchmark 200 rows end-to-end.
3. Run a 2500-row 6-qubit frontier iteration.
4. Inspect rules, grouped-CV stability, and plots.
5. Run the 5000-row 6-qubit paper-frontier atlas.
6. Only then launch the 8-qubit publication confirmation run, ideally overnight.

## 10. Immediate implementation checklist

- Materialize `predictability_gap_linear_gbm`.
- Add a joined 30-feature frontier table from the sweep catalog and extended features.
- Add candidate-pool generation without target evaluation.
- Add property-only stratified selection into broad-balanced, frontier, and negative-control
  subsets.
- Extend analysis CLI to accept `FRONTIER_TIER_A_FIELDS`.
- Add PR-AUC, calibration curves, grouped-CV, leave-family-out, and rule extraction outputs.
- Add sampling-weighted tables for enriched atlas results.
- Add NVAR/NG-RC or Fourier-feature ridge as encoding-control classical baselines before
  making claims about reuploading.
- Keep all null and negative attribution results in the publication figures.
