# Current Status

Last updated: 2026-06-29.

## Current Scope

The project now implements a dataset-categorization study, not a broad quantum-advantage
claim. The fixed Standard-Spin v1 QRC, preprocessing, input encoding, and readout protocol
are held constant while dataset families and parameter regimes vary.

Protocol audit update: `COMPARISON_PROTOCOL.md` now freezes the publication-facing
standard comparison v3. Layer 1 calibrates QRC and ESN once on held-out calibration
datasets, then freezes both for the atlas. The completed v2 artifacts are a conservative
stress test because they compare a fixed QRC against a validation-tuned ESN; they are not
the final symmetric fairness comparison.

Frontier audit update: `FRONTIER_PROTOCOL.md` now freezes the next paper-facing target,
**A Regime Map of Conditional Quantum Reservoir Advantage**. The implemented frontier
workflow creates a 20,000-row synthetic property atlas from the 50-template sweep, computes
30 fixed Tier-A features, and freezes a target-free 5,000-row discovery plus 5,000-row
prospective-validation selection before expensive QRC/ESN labels are computed.

Frontier execution update: the full 10,000-row selected frontier atlas has now been
evaluated under the frozen `standard_v3` comparison using `results_calibration_v3/`.
Validation was run with 100-row checkpoints under `caffeinate`, leaving all chunk outputs
recoverable in `results_frontier_validation/checkpoints/`.

Dataset audit update: the configured primary catalog and default sweep are now synthetic-only.
Santa Fe laser is retained only as an external real-data validation bridge. Existing generated
legacy artifacts still include the earlier 40 Santa Fe windows until the next rerun.

## Artifact Inventory

- `results_full/full_catalog.csv`: legacy 50-row schema-v1 artifact.
- `results_sweep/sweep_catalog.csv`: legacy 1000-row schema-v1 artifact.
- `results_features/extended_features_sweep.csv`: 1000 rows, 24 deterministic Tier-B features.
- `results_meta/importances.csv`: 16 ranked core-axis meta-model feature rows.
- `results_analysis/`: deterministic analysis tables and figures.
  - `sweep_summary.csv`: 11 rows including overall.
  - `family_advantage_bootstrap.csv`: 11 rows, 1000 bootstrap replicates.
  - `robustness_summary.csv`: 4 formal anti-circularity feature sets.
  - `importance_bootstrap.csv`: 16 all-feature importance rows, 30 bootstrap replicates.
- `results_atlas/`: 1000-row QRC usefulness atlas and figures.
- `results_publication/`: publication-facing figure/report package.
- `results_quantum_attribution/`: corrected paired J=1 vs J=0 attribution control.
- `results_visuals/`: ten state-of-the-art PNG/PDF figures, `index.html`, and report.
- `results_scientific_plots/`: five dense publication-style PNG/PDF multi-panel figures.
- `COMPARISON_PROTOCOL.md`: frozen comparison protocol v3, based on held-out global
  calibration followed by fixed-vs-fixed Spin-QRC versus dimension-matched sparse ESN.
- `NPJ_FRONTIER_PROTOCOL.md`: next NPJ-facing support-aware frontier protocol with 16 broad
  process families plus perturbation axes, 50,000 synthetic property candidates, 20,000
  recommended evaluated labels, optional 50,000-label stress run, OOD scoring, real-world
  probes, and rule/feature stability tests.
- `results_calibration_v3_explore/`: small held-out exploratory calibration artifact for
  layer 1.
- `results_sweep_v3_explore/`: small 50-row calibrated fixed-vs-fixed sanity sweep.
- Primary feature contract: schema v2 with 20 predefined measured dataset properties.
- Frontier feature contract: 30 Tier-A ground-rule properties, implemented in
  `FRONTIER_TIER_A_FIELDS`.
- `results_frontier_features_v3/`: 1000-row v3 anchor joined into the 30-feature table.
- `results_frontier_regime_v3/`: first 30-feature anchor regime-map analysis on the existing
  v3 evaluated sweep.
- `results_frontier_smoke/`: small CLI smoke outputs for property-atlas and target-free
  selection.
- `results_frontier_property/`: completed 20,000-row synthetic property atlas.
- `results_frontier_selection/`: frozen target-free 5,000 discovery + 5,000 validation
  selection.
- `results_frontier_discovery/`: completed 5,000-row discovery labels.
- `results_frontier_validation/`: completed 5,000-row prospective-validation labels plus
  50 resumable checkpoint chunks.
- `results_frontier_regime_discovery/` and `results_frontier_regime_validation/`: split-level
  30-feature regime analyses.
- `results_frontier_publication/`: frontier publication plot package with PNG/PDF figures,
  prospective predictions, metrics, report, and `index.html`.
- `results_qrc_power_audit/`: diagnostic QRC-capacity/encoding audit checking whether the
  primary QRC is obviously underpowered relative to the matched ESN.

## Dataset Counts

- Configured first catalog: 50 named synthetic benchmark datasets.
- Configured default sweep: 1000 synthetic rows.
- Frontier property atlas: 20,000 synthetic rows (`n_per_template=400`).
- Frontier evaluated atlas: 10,000 QRC/ESN-labeled rows, split into 5,000 discovery rows
  and 5,000 prospective validation rows.
- External validation bridge: 40 optional Santa Fe laser real-data windows.
- Current legacy artifact composition: 960 synthetic parameterized rows plus 40 Santa Fe
  laser real-bridge windows.
- Sweep QA: 0 all-zero metric rows after fixing high-order NARMA scaling and degenerate
  Ikeda/circle-map sweep ranges.
- Corrected attribution control: 360 chaotic-flow / chaotic-map rows.

Family counts in current legacy `results_sweep/sweep_catalog.csv`:

| family | rows |
|---|---:|
| chaotic_map | 200 |
| chaotic_flow | 160 |
| nonstationary | 140 |
| nonlinear_stochastic | 120 |
| oscillatory | 120 |
| input_driven | 100 |
| linear_stochastic | 60 |
| long_range | 40 |
| real_bridge | 40 |
| colored_noise | 20 |

## 1000-Row Sweep Result

Comparison protocol below is legacy/v2 context. Do not report these values as the final
symmetric standard v3 comparison until `results_sweep_v3/` and downstream analyses have
been regenerated from a held-out calibration config.

Target: `qrc_advantage = nrmse_esn_matched - nrmse_qrc_spin`.

- Overall mean advantage: `-0.1875`.
- Overall median advantage: `-0.0521`.
- QRC wins with advantage `> 0`: `341 / 1000`.
- QRC-useful rows with advantage `>= 0.05`: `204 / 1000`.
- Near ties: `291 / 1000`.
- Baseline preferred: `505 / 1000`.

Bootstrap intervals from `results_analysis/family_advantage_bootstrap.csv`:

| family | rows | mean advantage | 95% CI | qrc-useful rate |
|---|---:|---:|---:|---:|
| overall | 1000 | `-0.1875` | `[-0.2132, -0.1620]` | `0.204` |
| chaotic_map | 200 | `0.0411` | `[0.0067, 0.0778]` | `0.440` |
| long_range | 40 | `0.0230` | `[0.0005, 0.0471]` | `0.250` |
| nonlinear_stochastic | 120 | `0.0226` | `[0.0144, 0.0316]` | `0.183` |
| chaotic_flow | 160 | `-0.1362` | `[-0.1916, -0.0750]` | `0.256` |
| input_driven | 100 | `-0.4356` | `[-0.4906, -0.3786]` | `0.040` |
| real_bridge | 40 | `-0.5331` | `[-0.6311, -0.4315]` | `0.025` |
| oscillatory | 120 | `-0.7456` | `[-0.8258, -0.6750]` | `0.000` |

The strongest useful regime remains `chaotic_map`, but the expanded result is still
selective rather than broadly positive.

## Meta-Model Robustness

Gradient-boosting CV metrics from the 1000-row sweep:

| feature set | CV R2 | ROC-AUC | top drivers |
|---|---:|---:|---|
| all features | `0.6651` | `0.8564` | `r2_linear`, `ac_timescale`, `spectral_entropy`, `snr_db` |
| without `r2_linear` | `0.6718` | `0.8555` | `spectral_entropy`, `nl_gain`, `pred_nrmse_gbm`, `ac_timescale` |
| without predictability proxies | `0.6599` | `0.8558` | `spectral_entropy`, `nl_gain`, `ac_timescale`, `mem_capacity` |
| chaos/nonlinearity/complexity only | `0.5271` | `0.8365` | `nl_gain`, `spectral_entropy`, `dfa_alpha`, `spectral_flatness` |

This preserves the anti-circularity result: the property-to-usefulness signal remains
strong after removing `r2_linear` and after removing direct predictability proxies.

## Frontier 30-Feature Anchor Result

The existing v3 sweep was joined with deterministic extended features into
`results_frontier_features_v3/frontier_30_features.csv`:

- Rows: `1000`.
- Declared Tier-A features: `30`.
- Missing Tier-A features: `0`.

The first 30-feature anchor analysis in `results_frontier_regime_v3/` is not the final
frontier claim, but it validates the pipeline before the 20k/10k run:

- Gradient-boosted regression R2: `0.3717`.
- Gradient-boosted regression MAE: `0.1541`.
- Useful-class ROC-AUC: `0.7506`.
- Useful-class PR-AUC: `0.2631` against a useful base rate of `0.051`.
- Grouped by base generator: ROC-AUC `0.6757`, PR-AUC `0.1431`.
- Grouped by family: ROC-AUC `0.6337`, PR-AUC `0.1817`.

The strongest shallow-rule pocket in the 1000-row anchor is:

- `nl_gain <= -0.375`
- `n = 27`
- QRC-useful rate: `0.444`
- QRC win rate: `0.556`
- mean advantage: `+0.073`

This is a hypothesis for the frontier discovery/validation design, not yet a final paper
claim.

## Frontier Discovery/Validation Result

The completed frontier atlas uses the frozen `standard_v3` fixed-vs-fixed QRC/ESN protocol
with 30 Tier-A measured properties.

Discovery split (`results_frontier_discovery/`):

- Rows: `5000`.
- QRC wins with advantage `> 0`: `949 / 5000`.
- QRC-useful rows with advantage `>= 0.05`: `243 / 5000`.
- Mean advantage: `-0.1612`.
- Median advantage: `-0.0326`.
- Split-internal gradient-boosted regression R2: `0.4748`.
- Split-internal useful-class ROC-AUC: `0.8386`.
- Split-internal useful-class PR-AUC: `0.2182`.

Validation split (`results_frontier_validation/`):

- Rows: `5000`.
- QRC wins with advantage `> 0`: `955 / 5000`.
- QRC-useful rows with advantage `>= 0.05`: `258 / 5000`.
- Mean advantage: `-0.1596`.
- Median advantage: `-0.0297`.
- Split-internal gradient-boosted regression R2: `0.4780`.
- Split-internal useful-class ROC-AUC: `0.8304`.
- Split-internal useful-class PR-AUC: `0.2057`.

Prospective discovery-trained validation from `results_frontier_publication/`:

- Regression R2: `0.4930`.
- Regression MAE: `0.1254`.
- Useful-class ROC-AUC: `0.8563`.
- Useful-class PR-AUC: `0.2302`.
- Brier score: `0.0451`.

Generalization stress tests on the validation split:

- Grouped by base generator: R2 `0.0383`, ROC-AUC `0.7630`, PR-AUC `0.1488`.
- Grouped by family: R2 `-3.3062`, ROC-AUC `0.5114`, PR-AUC `0.0866`.

Interpretation: the row-level property-to-usefulness map is stable and prospectively
predictive inside the frozen synthetic atlas. Base-generator holdout remains partly
predictive, but family-held-out generalization is weak. The paper claim should therefore be
phrased as a conditional regime-map/dataset-categorization result, not as a broad
new-family predictor or a broad average QRC advantage.

Top discovery-trained frontier drivers:

1. `ext_psd_slope`
2. `nl_gain`
3. `ext_volatility_ac1`
4. `ext_trend_strength`
5. `lyapunov`
6. `ext_spectral_centroid`
7. `snr_db`
8. `pred_nrmse_gbm`

Strongest validation rule pocket:

- `adf_p > 0.009`
- `pred_nrmse_gbm > 0.293`
- `ext_lz_complexity <= 0.995`
- `n = 399`
- QRC-useful rate: `0.283`
- QRC win rate: `0.361`
- mean advantage: `-0.076`

This is a useful-enrichment pocket, not a positive-mean family-level advantage.

## Extended Features

`results_features/extended_features_sweep.csv` adds 24 deterministic Tier-B descriptors:
sample/approximate entropy, Lempel-Ziv complexity, Hurst R/S, spectral slope/moments,
zero-crossing and turning-point rates, outlier/spike rates, ARCH/volatility measures,
trend and seasonality strength, changepoint count, recurrence statistics, false-nearest
neighbors, approximate correlation dimension, BDS-like nonlinearity, zero fraction, and
positive-value CV2.

These are exploratory robustness features. They do not replace the frozen schema-v1 core
axes used in the main explanatory meta-model.

## Visual Suite

`results_visuals/` now contains a reproducible ten-figure visual package:

- visual abstract,
- dataset-property landscape,
- 1000-row sweep barcode,
- family outcome summary,
- advantage distributions,
- meta-model evidence and anti-circularity diagnostics,
- extended feature map,
- quantum-attribution guardrail,
- 50-generator benchmark inventory,
- all-points feature regressions with fitted lines and confidence bands.

The HTML entry point is `results_visuals/index.html`. The figures are designed to support
the dataset-categorization claim while keeping the null/mixed attribution result visible.

## Frontier Publication Figures

`results_frontier_publication/` contains the current paper-facing frontier visual package:

- Figure 1: prospective validation regime map with all 5,000 validation points.
- Figure 2: discovery-trained meta-model validation, calibration, importances, and
  grouped-generalization diagnostics.
- Figure 3: all-points feature regressions for the top frontier drivers.
- Figure 4: rule pockets, family distributions, split agreement, and claim boundary.

The HTML entry point is `results_frontier_publication/index.html`; the manifest is
`results_frontier_publication/frontier_publication_plots_manifest.json`.

## QRC Power Audit

`results_qrc_power_audit/` checks whether the current `standard_v3` QRC is obviously
underpowered relative to the matched ESN. This is a diagnostic only, not a replacement for
the frozen atlas protocol.

Held-out calibration-set audit (`26` datasets, seed `0`):

| variant | features | mean QRC NRMSE | matched ESN NRMSE | mean advantage |
|---|---:|---:|---:|---:|
| `standard_v3_current_6q60` | 60 | `0.8721` | `0.8082` | `-0.0639` |
| `clean_capacity_8q80` | 80 | `0.8929` | `0.8095` | `-0.0834` |
| `clean_capacity_10q100` | 100 | `0.8779` | `0.7974` | `-0.0805` |
| `clean_deeper_6q84` | 84 | `0.9235` | `0.7905` | `-0.1330` |
| `clean_deeper_8q112` | 112 | `0.9050` | `0.7914` | `-0.1136` |
| `weak_amp_damping_6q60` | 60 | `1.0828` | `0.8082` | `-0.2746` |
| `weak_dephasing_6q60` | 60 | `1.0724` | `0.8082` | `-0.2642` |
| `encoding_reupload_6q60` | 60 | `1.0575` | `0.8082` | `-0.2493` |
| `encoding_reupload_8q80` | 80 | `1.0664` | `0.8095` | `-0.2569` |

Balanced validation-subset audit (`108` rows, `12` per family, seed `0`):

| variant | features | mean QRC NRMSE | matched ESN NRMSE | mean advantage | QRC win rate |
|---|---:|---:|---:|---:|---:|
| `standard_v3_current_6q60` | 60 | `0.9736` | `0.9197` | `-0.0539` | `0.306` |
| `clean_capacity_10q100` | 100 | `0.9736` | `0.8953` | `-0.0784` | `0.287` |
| `clean_deeper_8q112` | 112 | `1.0070` | `0.9320` | `-0.0749` | `0.259` |
| `encoding_reupload_6q60` | 60 | `1.0855` | `0.9197` | `-0.1658` | `0.278` |

Interpretation: the primary QRC does not appear to be losing merely because it is too
small, too shallow, non-dissipative, or missing the older reupload encoding. Larger/deeper
QRCs did not materially improve QRC performance, and increasing feature dimension also
helps the matched ESN. This supports the current conclusion that ESN dominance is a real
protocol outcome, not an obvious underpowered-QRC artifact. A full paper robustness layer
could still add a larger formal QRC calibration grid before final submission.

## NPJ Frontier v4 Plan

`NPJ_FRONTIER_PROTOCOL.md` is now the proposed next paper-facing protocol. It addresses the
main limitations of the completed v3.1 frontier result:

- expands the broad synthetic taxonomy from 9 families to 16 canonical process families plus
  perturbation axes;
- keeps the 30 Tier-A features as the headline explanatory coordinate system;
- increases the property-only candidate atlas to 50,000 synthetic configurations;
- recommends 20,000 evaluated labels, split into 10,000 discovery and 10,000 prospective
  validation rows;
- defines an optional all-in 50,000-label stress run;
- adds a discovery-fitted atlas-support / OOD score for every synthetic and real row;
- adds real-world probes only after the synthetic model is frozen;
- adds bootstrap stability tests for top features and rule pockets;
- adds NVAR/NG-RC and validation-tuned ESN as secondary classical robustness baselines.

The intended v4 claim is:

> QRC usefulness over a matched ESN is rare but structured and support-aware: it can be
> prospectively predicted in canonical time-series property space, and real-world probes
> are interpretable only when they fall inside atlas support.

The v4 protocol still does not permit a broad quantum-advantage claim.

## Scientific Publication Figures

`results_scientific_plots/` contains compact, information-dense figures intended for a
paper draft or appendix:

- Figure 1: dense atlas summary.
- Figure 2: all-point feature regressions.
- Figure 3: family-property matrix with useful rates and effect intervals.
- Figure 4: meta-model validation, calibration, robustness, and importance intervals.
- Figure 5: quantum-attribution guardrail.

The CLI is `qrc_dataset_profiler.run_scientific_plots`.

## Quantum-Attribution Control

The corrected paired attribution control was rerun on the expanded chaotic-flow/map subset:

| family | rows | mean `J0-J1` delta | 95% CI | mechanism signal |
|---|---:|---:|---:|---|
| overall | 360 | `0.0127` | `[-0.0128, 0.0389]` | null/mixed |
| chaotic_flow | 160 | `0.0787` | `[0.0316, 0.1254]` | positive for J=1 |
| chaotic_map | 200 | `-0.0402` | `[-0.0654, -0.0131]` | negative for J=1 |

This remains a control result. It does not support a broad coupling, entanglement, or
fundamental quantum-advantage mechanism claim.
