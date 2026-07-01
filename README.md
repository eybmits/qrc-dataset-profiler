# qrc_dataset_profiler

Unified protocol to characterize univariate time-series datasets with the same schema,
compare canonical frozen spin-QRC variants against a feature-matched frozen ESN, and
learn a regime map that categorizes when QRC is especially worth testing.

The current paper-facing result is a v5 regime atlas: 50,000 synthetic property
candidates, a frozen target-free 20,000-row labeled discovery/validation subset,
three globally calibrated QRC variants, and a feature-matched frozen sparse ESN. The
goal is not to prove fundamental quantum advantage. The supported claim is conditional:
dataset properties identify regimes where QRC is worth testing against a strong ESN
baseline.

Project website: `https://eybmits.github.io/qrc-dataset-profiler/`.

## Repository Map

- `PROTOCOL.md` - frozen protocol contract and schema semantics.
- `TRIAGE.md` - CSV input contract and user-facing QRC-usefulness triage guide.
- `COMPARISON_PROTOCOL.md` - historical frozen model comparison protocol v3.
- `FRONTIER_PROTOCOL.md` - earlier frontier atlas plan and selection machinery.
- `NPJ_FRONTIER_PROTOCOL.md` - 50k-candidate support-aware frontier protocol.
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
- `src/qrc_dataset_profiler/run_triage.py` - user-facing QRC-usefulness triage for a new CSV time series.
- `src/qrc_dataset_profiler/run_triage_web.py` - local browser interface for the triage tool.
- `scripts/export_static_triage_model.py` - export the lightweight browser-side triage model for GitHub Pages.
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
- **v5 calibration complete:** QRC-M, QRC-E, QRC-D, and the sparse leaky ESN are calibrated
  once on 320 held-out synthetic calibration rows and then frozen globally.
- **v5 atlas complete:** 50,000 synthetic property candidates, 20,000 target-free selected
  labels, split into 10,000 discovery and 10,000 prospective-validation rows.
- **v5 publication package complete:** seven PNG/PDF figure sets, tables, report, and HTML
  index in `results_v5_publication/`.
- **Paper draft complete:** LaTeX source is under `paper/`; the compiled paper PDF is
  `paper/mapping_quantum_reservoir_advantage.pdf`.
- **Triage tools complete:** the public website provides a lightweight browser-side
  CSV analyzer; `run_triage` and `run_triage_web` remain the full Python reference
  screeners using the frozen discovery atlas and atlas-support/OOD scoring.

## Dataset Counts

- Primary v5 property atlas: **50,000 synthetic rows** from 16 broad process families.
- v5 labeled atlas: **20,000 rows**, split into **10,000 discovery** and **10,000
  prospective validation** rows.
- v5 feature contract: **30 Tier-A descriptors**.
- v5 validation result: **3,239 / 10,000** any best-QRC wins and **1,253 / 10,000**
  useful rows at `best_qrc_advantage_vs_esn >= 0.05`.
- v5 validation meta-model: **R2 0.4072**, **ROC-AUC 0.8357**, **PR-AUC 0.4305**.
- Full 50,000-row atlas shape: **45,500 forecasting rows** and **4,500 input-driven
  memory rows**, all scalar (`n_channels = 1`).

## Key Local Artifacts

- `results_calibration_v5/frozen_v5_config.json` - globally frozen QRC/ESN configuration.
- `results_frontier_v4_property/frontier_property_atlas.csv` - 50,000-row property atlas.
- `results_frontier_v4_selection/frontier_evaluation_selection.csv` - target-free 20,000-row
  discovery/validation selection.
- `results_frontier_v5_discovery/frontier_discovery_evaluated_v5_multi_qrc.csv` - 10,000
  discovery labels.
- `results_frontier_v5_validation/frontier_validation_evaluated_v5_multi_qrc.csv` - 10,000
  prospective-validation labels.
- `results_v5_publication/` - v5 paper-facing tables, seven figure sets, report, and HTML.
- `paper/` - executable IEEE-style LaTeX paper source and compiled PDF.
- `docs/` - static researcher-facing GitHub Pages website with browser-side CSV triage.
- `docs/assets/triage_model.json` - compact exported atlas model for browser triage.
- `TRIAGE.md` - public CSV contract and interpretation boundary for the triage tool.

## Quickstart

Install and run the fast repository checks:

```bash
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest -q
```

Rebuild the paper-facing v5 analysis package from committed v5 labels:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_publication --out results_v5_publication
```

Regenerate the full v5 atlas from scratch:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier property-atlas --taxonomy v4 --out results_frontier_v4_property --n-per-template 500 --fast --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier select --property-atlas results_frontier_v4_property/frontier_property_atlas.csv --out results_frontier_v4_selection --n-discovery 10000 --n-validation 10000 --selection-protocol v4
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_protocol calibrate --out results_calibration_v5 --fast
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_protocol evaluate-selection --selection results_frontier_v4_selection/frontier_evaluation_selection.csv --calibration-config results_calibration_v5/frozen_v5_config.json --out results_frontier_v5_discovery --split discovery --fast --seeds 1 --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_protocol evaluate-selection --selection results_frontier_v4_selection/frontier_evaluation_selection.csv --calibration-config results_calibration_v5/frozen_v5_config.json --out results_frontier_v5_validation --split validation --fast --seeds 1 --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_publication --out results_v5_publication
```

Earlier increment workflows remain reproducible. `run_study` defaults to
`--comparison-protocol standard_v3`; use `standard_v2` for the conservative
validation-tuned ESN stress test, and `legacy_v1` only to reproduce early
simple-cycle artifacts.

Rebuild the LaTeX paper:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf mapping_quantum_reservoir_advantage.pdf
```

QRC-usefulness triage for a new univariate time series:

Public browser analyzer:

```text
https://eybmits.github.io/qrc-dataset-profiler/#try
```

This static web analyzer runs in the visitor's browser and does not upload the
CSV. It uses a compact JavaScript-computable feature subset for fast screening.
The Python command below is the reference triage path for paper-grade reporting.

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_triage \
  --series path/to/my_series.csv \
  --column value \
  --name my_dataset
```

The triage command does not run QRC or ESN on the submitted dataset. It computes the
30 atlas descriptors, fits the lightweight discovery-only meta-model from
`results_frontier_v5_discovery/frontier_discovery_evaluated_v5_multi_qrc.csv`, adds an
atlas-support/OOD score, and reports whether QRC is worth testing against the frozen ESN.
Use `--format json --out triage_report.json` for a machine-readable report. See
`TRIAGE.md` for the CSV contract, atlas-shape notes, and interpretation boundary.

Local browser interface for the same triage path:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_triage_web --port 8765
```

Then open `http://127.0.0.1:8765`, upload or paste a univariate CSV, and run the triage.

## Claim Boundary

The current evidence supports a protocol-local regime-map claim: under globally frozen
canonical QRC variants and a globally frozen feature-matched sparse ESN, dataset properties
predict when QRC is worth testing. The strongest empirical QRC-usefulness regime is slow,
stateful forecasting: drifting trends, regime changes, persistent low-frequency structure,
long memory, and temporally correlated fluctuations. The project does not claim broad
average QRC superiority, computational quantum advantage, or a proven coupling/entanglement
mechanism.
