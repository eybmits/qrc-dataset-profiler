# Roadmap

## Increment 4: Paper-Grade Analysis Suite

Goal: turn the current sweep and ad-hoc analyses into reproducible, reviewer-facing
evidence for dataset categorization: learn which time-series properties indicate that the
fixed Spin-QRC is especially useful relative to matched classical baselines. Quantum
attribution is a control for claim hygiene, not the primary objective.

## Completed in Increment 4A

- Added a formal analysis module and CLI, `qrc_dataset_profiler.run_analysis`,
  that reads `results_sweep/sweep_catalog.csv` and writes a deterministic analysis folder.
- Added a QRC usefulness atlas CLI, `qrc_dataset_profiler.run_usefulness_map`, that writes
  row-level usefulness labels, meta-model predictions, property-map coordinates, and
  family category figures.
- Added a publication package CLI, `qrc_dataset_profiler.run_publication_package`, that
  assembles multi-panel PNG/PDF figures and a concise atlas report from committed results.
- Produced bootstrap confidence intervals for family-level QRC advantage and meta-model
  feature importances.
- Formalized the anti-circularity suite:
  - all core features,
  - without `r2_linear`,
  - without direct predictability proxies,
  - chaos/nonlinearity/complexity-only feature set.
- Generated reviewer-grade tables and figures for the sweep summary, meta-model robustness,
  and family-level advantage distribution.

## Completed in Increment 4B

- Replaced the ad-hoc quantum ablation with a corrected attribution protocol:
  - compare matched `J=1` and `J=0` reservoirs under identical feature dimensions,
  - include matched seeds and family stratification,
  - report confidence intervals and paired differences,
  - treat negative or null results as first-class outcomes.

The corrected result is null/mixed overall: `chaotic_flow` favors `J=1`, but
`chaotic_map` favors `J=0`, so the project still must not claim a broad coupling,
entanglement, or quantum-mechanism explanation.

## Completed in Increment 5

- Expanded the first catalog from 20 to 50 named forecasting benchmarks.
- Added standard additional regimes: NARMA2/5/30, nonlinear channel equalization,
  Ikeda/tent/sine/circle/Lozi/standard/quadratic maps, Duffing, Van der Pol,
  Lorenz96, Chua, ARMA/ARIMA/seasonal AR, SETAR, EGARCH, stochastic volatility,
  bilinear/ARCH, brown/blue noise, amplitude modulation, damping, level shifts,
  intermittent demand, and trend-seasonality.
- Changed the default parameterized sweep to 1000 rows at `n_per_family=20`.
  The initial generated artifacts used 960 synthetic rows plus 40 Santa Fe laser
  real-bridge windows.
- Added `qrc_dataset_profiler.run_extended_features`, which writes a deterministic
  Tier-B feature table with 24 extra descriptors for entropy, recurrence, volatility,
  trend/seasonality, changepoints, embedding geometry, intermittency, and spectral shape.
- Regenerated the full catalog, sweep catalog, meta-model, analysis suite, usefulness
  atlas, attribution control, and publication figures from the expanded artifacts.
- Added `qrc_dataset_profiler.run_visual_suite`, which writes a ten-figure PNG/PDF
  visual package plus `results_visuals/index.html` and a concise visual report.
- Added `qrc_dataset_profiler.run_scientific_plots`, which writes five dense
  publication-style multi-panel figures to `results_scientific_plots/`.

## Completed in Increment 6A

- Researched and froze the publication-facing three-layer model comparison in
  `COMPARISON_PROTOCOL.md`.
- Set the Layer-1 primary fairness comparison to:
  - weakly dissipative input-injection Spin-QRC with train-split-only input scaling,
    fixed `p1=0.02`, fixed `p_phi=0.01`, and no RZ reuploading,
  - dimension-matched canonical sparse random leaky ESN,
  - both reservoirs calibrated once on held-out synthetic calibration datasets and then
    frozen globally for atlas evaluation.
- Kept the validation-tuned sparse ESN as a Layer-2 strong-classical stress test, and kept
  the legacy simple-cycle ESN, linear ridge, GBM, QRC reuploading, and paired `J=1` vs
  `J=0` checks as robustness/ablation controls.
- Added code support for `--comparison-protocol standard_v3` while retaining
  `--comparison-protocol standard_v2` and `--comparison-protocol legacy_v1` for
  reproducing current artifacts.
- Moved Santa Fe laser out of the primary catalog/sweep and into external validation;
  the primary 50-entry catalog and default 1000-row sweep are now synthetic-only, with
  Hénon-Heiles added as the synthetic replacement.

## Next: Increment 6B

- Calibrate QRC and ESN once on held-out synthetic calibration datasets.
- Regenerate the full 50-row synthetic catalog and 1000-row synthetic sweep under `standard_v3`
  with the frozen calibration config.
- Recompute extended features, meta-model, analysis suite, usefulness atlas, attribution
  controls, and publication figures from the v3 catalog.
- Add NVAR/next-generation reservoir computing as an additional robustness baseline.
- Build confidence-aware meta-model outputs only after the v3 rerun, so uncertainty
  calibration reflects the frozen standard comparison.

## Increment 7: Frontier Regime Map

Paper target: **A Regime Map of Conditional Quantum Reservoir Advantage**.

Completed now:

- Added `FRONTIER_PROTOCOL.md` as the paper-facing frontier atlas protocol.
- Added `qrc_dataset_profiler.frontier` and `qrc_dataset_profiler.run_frontier`.
- Added the 30-feature Tier-A ground-rule contract:
  - 20 schema-v2 core properties,
  - 10 frontier upgrades for predictability gap, volatility, recurrence, spectral shape,
    trend/changepoints, and Lempel-Ziv complexity.
- Materialized `predictability_gap_linear_gbm` as a deterministic feature.
- Added a one-pass property-atlas builder for the 20,000-row synthetic candidate pool.
- Added target-free discovery/validation selection:
  - 5,000 discovery labels,
  - 5,000 prospective validation labels,
  - selection based only on generator identity and measured properties.
- Completed the 20,000-row synthetic property atlas.
- Completed the 10,000-row selected evaluated atlas under frozen `standard_v3`:
  - 5,000 discovery rows in `results_frontier_discovery/`,
  - 5,000 validation rows in `results_frontier_validation/`,
  - validation run protected with `caffeinate` and 100-row checkpoints.
- Added the 30-feature frontier regime analysis:
  - gradient-boosted regression/classification,
  - PR-AUC for the rare useful class,
  - grouped CV by base generator and family,
  - shallow raw-threshold rule extraction,
  - regime-map and importance figures.
- Added `qrc_dataset_profiler.run_frontier plot`, which writes the frontier publication
  plot package in `results_frontier_publication/`.

Reproduction sequence:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier property-atlas --out results_frontier_property --n-per-template 400 --fast
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier select --property-atlas results_frontier_property/frontier_property_atlas.csv --out results_frontier_selection --n-discovery 5000 --n-validation 5000
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier evaluate-selection --selection results_frontier_selection/frontier_evaluation_selection.csv --out results_frontier_discovery --split discovery --comparison-protocol standard_v3 --calibration-config results_calibration_v3/frozen_config.json --fast --seeds 3 --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier evaluate-selection --selection results_frontier_selection/frontier_evaluation_selection.csv --out results_frontier_validation --split validation --comparison-protocol standard_v3 --calibration-config results_calibration_v3/frozen_config.json --fast --seeds 3 --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier analyze --evaluated-table results_frontier_discovery/frontier_discovery_evaluated_30_features.csv --out results_frontier_regime_discovery
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier analyze --evaluated-table results_frontier_validation/frontier_validation_evaluated_30_features.csv --out results_frontier_regime_validation
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier plot --out results_frontier_publication
```

Key frontier result:

- Validation QRC wins: `955 / 5000`.
- Validation QRC-useful rows at `qrc_advantage >= 0.05`: `258 / 5000`.
- Discovery-trained prospective validation: R2 `0.4930`, ROC-AUC `0.8563`, PR-AUC `0.2302`.
- Base-generator holdout remains informative: ROC-AUC `0.7630`.
- Family holdout is weak: ROC-AUC `0.5114`, R2 `-3.3062`.

Next scientific work:

- Use `NPJ_FRONTIER_PROTOCOL.md` as the next paper-facing design:
  - expand from 9 to 16 broad process families plus perturbation axes;
  - build a 50,000-row synthetic property candidate atlas;
  - evaluate 20,000 selected synthetic labels as the recommended NPJ run, with an optional
    50,000-label camera-ready stress run;
  - add atlas-support / OOD confidence scores;
  - add real-world probes as external interpolation checks only;
  - bootstrap/stability-test feature rankings and rule pockets;
  - add NVAR/next-generation reservoir computing as an additional robustness baseline.
- Convert the frontier result into a paper draft with claim wording centered on conditional
  dataset-property regimes and explicit support boundaries.
- Keep the quantum-attribution section as a guardrail/control, not as the central claim.

## Claim Policy

- Allowed now: dataset properties predict when the legacy Spin-QRC implementation beats the
  matched legacy ESN baseline on the current generated legacy sweep, and can be used to
  categorize datasets by expected QRC usefulness.
- Allowed after Increment 6B rerun: the same categorization claim under the frozen
  standard v3 comparison, if the v3 evidence supports it.
- Not allowed now: broad average QRC superiority, fundamental quantum advantage, or a
  strong coupling/entanglement mechanism claim.
- Mechanism attribution is optional follow-up science, not required for the core
  dataset-profiler/meta-model contribution.

## Acceptance Criteria

- `PYTHONPATH=src python -m pytest -q` passes.
- The analysis CLI runs from committed inputs and writes all tables/figures deterministically.
- The generated summary reports the exact feature sets, seeds, thresholds, and row counts.
- Negative quantum-attribution findings remain visible rather than being filtered out.
