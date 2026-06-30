# qrc_dataset_profiler

Unified protocol to characterize univariate time-series datasets with the same schema,
evaluate a fixed Standard-Spin v1 quantum reservoir against classical baselines, and
learn a meta-model that categorizes when QRC is especially useful.

The current result is useful but intentionally conservative: the 1000-dataset sweep and
meta-model show a robust property-to-advantage signal, including anti-circularity checks.
The goal is not to prove a fundamental quantum advantage. Quantum-attribution checks are
included as controls so the dataset-categorization claim stays bounded and defensible.

## Repository Map

- `PROTOCOL.md` - frozen protocol contract and schema semantics.
- `COMPARISON_PROTOCOL.md` - frozen publication-facing model comparison protocol v3.
- `FRONTIER_PROTOCOL.md` - 20k-property / 5k+5k evaluated frontier atlas plan.
- `NPJ_FRONTIER_PROTOCOL.md` - next 50k-candidate support-aware NPJ frontier protocol.
- `src/qrc_dataset_profiler/spec.py` - machine-readable schema definitions.
- `src/qrc_dataset_profiler/run_profile.py` - profile the initial catalog.
- `src/qrc_dataset_profiler/run_study.py` - build Block E baseline/QRC targets.
- `src/qrc_dataset_profiler/run_calibration.py` - held-out global QRC/ESN calibration for fixed-vs-fixed fairness.
- `src/qrc_dataset_profiler/run_meta.py` - fit the explanatory meta-model.
- `src/qrc_dataset_profiler/run_analysis.py` - reproducible reviewer-facing analysis suite.
- `src/qrc_dataset_profiler/run_usefulness_map.py` - row-level QRC usefulness atlas and plots.
- `src/qrc_dataset_profiler/run_extended_features.py` - deterministic Tier-B feature table.
- `src/qrc_dataset_profiler/run_publication_package.py` - publication figure/report package.
- `src/qrc_dataset_profiler/run_quantum_attribution.py` - corrected paired J=1 vs J=0 attribution runner.
- `src/qrc_dataset_profiler/run_visual_suite.py` - state-of-the-art visual suite and HTML index.
- `src/qrc_dataset_profiler/run_scientific_plots.py` - dense publication-style multi-panel figures.
- `src/qrc_dataset_profiler/run_frontier.py` - 30-feature frontier atlas, selection, and regime-map analysis.
- `src/qrc_dataset_profiler/frontier_plots.py` - prospective frontier publication figures.
- `STATUS.md` - current evidence, artifact inventory, and claim boundaries.
- `ROADMAP.md` - next planned analysis increment.

## Current Status

- **Increment 1 complete:** package scaffold, initial dataset generators, schema property
  estimators, smoke profiler outputs, and ground-truth validation.
- **Increment 2 complete:** Standard-Spin v1 reservoir, linear/GBM/ESN baselines,
  fair shared forecasting task alignment, and Block E targets.
- **Increment 3 complete:** parameterized sweep, scalable study mode, and
  explanatory meta-model for QRC advantage.
- **Increment 4 complete:** paper-grade analysis suite, formal anti-circularity reports,
  reviewer figures, and corrected paired J=1 vs J=0 attribution are implemented. The
  corrected attribution result is mixed/null overall, so no coupling mechanism claim is made.
- **Increment 5 complete:** 50 named benchmark datasets, 1000-row sweep/atlas, and a
  deterministic 24-feature Tier-B descriptor table.
- **Visual package complete:** ten state-of-the-art PNG/PDF figures, an HTML index,
  and a concise visual report in `results_visuals/`.
- **Scientific figure package complete:** five dense multi-panel publication figures in
  `results_scientific_plots/`.
- **Synthetic-primary catalog fixed:** Santa Fe laser is now external validation only;
  the primary 50-entry catalog and default 1000-row sweep are synthetic-only.
- **Comparison protocol v3 frozen:** the paper-facing standard comparison is now symmetric:
  QRC and ESN are each calibrated once on held-out calibration datasets, then frozen for
  the atlas. The previous v2 comparison remains a conservative stress test with a
  validation-tuned ESN.
- **Primary feature contract v2 frozen:** the explanatory meta-model uses 20 predefined
  measured dataset properties, adding sample entropy and Hurst R/S to the previous core axes.
- **Frontier protocol v3.1 evaluated:** the paper-title target is now “A Regime Map of
  Conditional Quantum Reservoir Advantage.” The frontier workflow has a completed 20,000-row
  synthetic property-only atlas, 30 fixed Tier-A features, and a completed target-free
  5,000-row discovery plus 5,000-row prospective-validation evaluated atlas under frozen
  `standard_v3`.
- **Frontier publication figures complete:** four PNG/PDF frontier figures, prospective
  predictions, metrics, report, and `index.html` are in `results_frontier_publication/`.
- **NPJ frontier protocol drafted:** `NPJ_FRONTIER_PROTOCOL.md` defines the next
  support-aware v4 atlas: 16 broad process families plus perturbation axes, 50,000
  synthetic property candidates, 20,000 evaluated labels as the recommended paper run,
  optional 50,000-label stress run, real-world external probes, OOD support scoring, and
  rule/feature stability testing.

## Dataset Counts

- Configured primary catalog: **50 synthetic rows** from the 50 nominal protocol entries.
- Configured default sweep: **1000 synthetic parameterized rows** across 9 families.
- External validation bridge: **40 Santa Fe laser real-data windows** are available via
  `make_real_bridge_specs`, but are not part of the primary synthetic atlas.
- Extended features: **1000 rows x 24 deterministic Tier-B features** in `results_features/`.
- Corrected attribution control: **360 rows** from the expanded chaotic-flow and chaotic-map subset.
- Frontier property atlas: **20,000 synthetic rows** from `n_per_template=400`.
- Frontier evaluated atlas: **10,000 selected rows** split into 5,000 discovery and
  5,000 prospective validation labels.
- Frontier validation result: **955 / 5000** any QRC wins and **258 / 5000** useful rows
  at `qrc_advantage >= 0.05`.
- Discovery-trained prospective meta-model on validation: **R2 0.4930**, **ROC-AUC 0.8563**,
  **PR-AUC 0.2302**.

## Key Local Artifacts

- `results_full/full_catalog.csv` - 50-row first full catalog with Block E targets.
- `results_sweep/sweep_catalog.csv` - 1000-row parameterized sweep catalog.
- `results_features/extended_features_sweep.csv` - deterministic Tier-B feature bank.
- `results_meta/importances.csv` - meta-model feature importances.
- `results_meta/importance_bar.png` - importance visualization.
- `results_meta/partial_dependence.png` - partial-dependence visualization.
- `results_meta/quantum_ablation.csv` - ad-hoc J=1 vs J=0 ablation; currently a red flag.
- `results_analysis/` - deterministic Increment-4 analysis tables and figures.
- `results_atlas/` - QRC usefulness map, family category summaries, and atlas figures.
- `results_publication/` - multi-panel PNG/PDF figures and concise atlas report.
- `results_quantum_attribution/` - corrected paired J=1 vs J=0 attribution artifacts.
- `results_visuals/` - ten-figure visual suite, `index.html`, and visual report.
- `results_scientific_plots/` - five dense publication-style PNG/PDF multi-panel figures.
- `results_frontier_property/` - 20,000-row synthetic property atlas.
- `results_frontier_selection/` - frozen target-free discovery/validation selection.
- `results_frontier_discovery/` - 5,000-row discovery labels.
- `results_frontier_validation/` - 5,000-row validation labels and 50 checkpoint chunks.
- `results_frontier_regime_discovery/` and `results_frontier_regime_validation/` - split
  regime-map analyses.
- `results_frontier_publication/` - frontier PNG/PDF figure package, metrics, report, and
  HTML index.
- `results_sweep_tiny/sweep_catalog.csv` - tiny sanity sweep output.

## Quickstart

```bash
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m qrc_dataset_profiler.run_profile --smoke --out results
PYTHONPATH=src python -m qrc_dataset_profiler.run_calibration --fast --out results_calibration_v3
PYTHONPATH=src python -m qrc_dataset_profiler.run_study --smoke --out results_full
PYTHONPATH=src python -m qrc_dataset_profiler.run_study --sweep --fast --out results_sweep --calibration-config results_calibration_v3/frozen_config.json
PYTHONPATH=src python -m qrc_dataset_profiler.run_extended_features --sweep --fast --out results_features
PYTHONPATH=src python -m qrc_dataset_profiler.run_meta --catalog results_sweep/sweep_catalog.csv --out results_meta
PYTHONPATH=src python -m qrc_dataset_profiler.run_analysis --catalog results_sweep/sweep_catalog.csv --out results_analysis
PYTHONPATH=src python -m qrc_dataset_profiler.run_usefulness_map --catalog results_sweep/sweep_catalog.csv --out results_atlas
PYTHONPATH=src python -m qrc_dataset_profiler.run_quantum_attribution --catalog results_sweep/sweep_catalog.csv --out results_quantum_attribution
PYTHONPATH=src python -m qrc_dataset_profiler.run_publication_package --out results_publication
PYTHONPATH=src python -m qrc_dataset_profiler.run_visual_suite --out results_visuals
PYTHONPATH=src python -m qrc_dataset_profiler.run_scientific_plots --out results_scientific_plots
```

`run_study` now defaults to `--comparison-protocol standard_v3`. Use
`--comparison-protocol standard_v2` for the conservative validation-tuned ESN stress test,
and `--comparison-protocol legacy_v1` only to reproduce earlier simple-cycle artifacts.

Frontier regime-map workflow:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier property-atlas --out results_frontier_property --n-per-template 400 --fast
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier select --property-atlas results_frontier_property/frontier_property_atlas.csv --out results_frontier_selection --n-discovery 5000 --n-validation 5000
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier evaluate-selection --selection results_frontier_selection/frontier_evaluation_selection.csv --out results_frontier_discovery --split discovery --comparison-protocol standard_v3 --calibration-config results_calibration_v3/frozen_config.json --fast --seeds 3 --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier evaluate-selection --selection results_frontier_selection/frontier_evaluation_selection.csv --out results_frontier_validation --split validation --comparison-protocol standard_v3 --calibration-config results_calibration_v3/frozen_config.json --fast --seeds 3 --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier analyze --evaluated-table results_frontier_discovery/frontier_discovery_evaluated_30_features.csv --out results_frontier_regime_discovery
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier analyze --evaluated-table results_frontier_validation/frontier_validation_evaluated_30_features.csv --out results_frontier_regime_validation
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier plot --out results_frontier_publication
```

## Claim Boundary

The current evidence supports the claim that dataset properties can predict when a fixed
Spin-QRC implementation is relatively useful under the tested frozen `standard_v3`
protocol. The frontier evidence supports a conditional regime-map claim inside the synthetic
atlas, with moderate base-generator generalization and weak family-held-out generalization.
No version claims a broad average QRC win or a coupling/entanglement mechanism.
