# Current Status

Last updated: 2026-06-28.

## Commit Chain

- `fbce2ce` - Increment 3: explanatory meta-model (property -> QRC advantage)
- `591038e` - Sweep: parameterized dataset expansion (360 datasets) + scalable study
- `01853d2` - Increment 2: Standard-Spin v1 reservoir + fair baselines + advantage targets
- `d2d248c` - Increment 1: 20 generators + schema-v1 estimators + smoke profiler
- `64f5c13` - Increment 1 scaffold: frozen PROTOCOL.md + schema v1 package skeleton

## Artifact Inventory

- `results_full/full_catalog.csv`: 19 rows, 50 columns.
- `results_sweep/sweep_catalog.csv`: 360 rows, 50 columns, 9 families.
- `results_sweep_tiny/sweep_catalog.csv`: 54-row sanity sweep.
- `results_meta/importances.csv`: 16 ranked meta-model feature rows.
- `results_meta/importance_bar.png`: feature-importance figure.
- `results_meta/partial_dependence.png`: partial-dependence figure.
- `results_meta/quantum_ablation.csv`: 140-row ad-hoc coupling ablation.

Current generated artifact footprint is small, about 432 KB across the versioned result
directories.

## 360-Row Sweep Summary

Family counts in `results_sweep/sweep_catalog.csv`:

| family | rows |
|---|---:|
| chaotic_flow | 80 |
| chaotic_map | 60 |
| colored_noise | 20 |
| input_driven | 60 |
| linear_stochastic | 20 |
| long_range | 20 |
| nonlinear_stochastic | 20 |
| nonstationary | 40 |
| oscillatory | 40 |

QRC advantage summary, defined as `nrmse_esn_matched - nrmse_qrc_spin`:

- Mean advantage: `-0.1755`
- Median advantage: `-0.0512`
- QRC wins with advantage `> 0`: `126 / 360`
- QRC wins with advantage `> 0.05`: `79 / 360`

The strongest mean family-level advantage is currently in `chaotic_map`; broad average
advantage is not established across all families.

## Meta-Model Robustness

Gradient-boosting CV metrics from the current 360-row sweep:

| feature set | CV R2 | ROC-AUC | top drivers |
|---|---:|---:|---|
| all features | 0.6954 | 0.8677 | `r2_linear`, `ac_timescale`, `pred_nrmse_gbm`, `snr_db` |
| without `r2_linear` | 0.6405 | 0.8601 | `nl_gain`, `pred_nrmse_gbm`, `mem_capacity`, `spectral_entropy` |
| without predictability proxies | 0.6307 | 0.8713 | `nl_gain`, `spectral_entropy`, `mem_capacity`, `spectral_flatness` |
| chaos/nonlinearity/complexity only | 0.5435 | 0.8477 | `nl_gain`, `dfa_alpha`, `lyapunov`, `zero_one_K` |

This supports the anti-circularity framing: predictability proxies help, but the
property-to-advantage signal persists without `r2_linear` and without the direct
predictability columns.

## Quantum-Attribution Status

The current ad-hoc coupling ablation is a red flag, not a positive quantum-attribution
result.

From `results_meta/quantum_ablation.csv`:

- Rows: `140` chaotic-flow / chaotic-map variants.
- Mean `J=1` entangling QRC NRMSE: `0.3649`
- Mean `J=0` no-coupling QRC NRMSE: `0.3234`
- Mean matched ESN NRMSE: `0.3261`
- Mean advantage for `J=1`: `-0.0389`
- Mean advantage for `J=0`: `0.0027`
- Wins with advantage `> 0.05`: `J=1` has `39 / 140`; `J=0` has `40 / 140`.
- Mean `qrc_J0 - qrc_J1`: `-0.0415`, so removing coupling improves NRMSE on average in this run.

Therefore the repository should not claim that the current advantage is caused by
coupling or entanglement. Increment 4 must replace this ad-hoc script with a corrected,
reproducible attribution analysis before any quantum-mechanism claim is made.
