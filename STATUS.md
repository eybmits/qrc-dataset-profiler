# Current Status

Last updated: 2026-07-01.

## Paper-Facing State

The current paper-facing project is the v5 multi-QRC regime atlas. It is a
dataset-categorization benchmark, not a broad quantum-advantage claim.

Main evidence:

- 50,000 synthetic property candidates from the v4 taxonomy.
- 20,000 target-free selected labeled rows:
  - 10,000 discovery rows.
  - 10,000 prospective-validation rows.
- 30 fixed Tier-A dataset descriptors.
- Three canonical QRC variants, calibrated once globally and then frozen:
  - QRC-M: mechanistic input-injection spin-QRC, no RZ reuploading, no dissipation.
  - QRC-E: encoding-enhanced spin-QRC with fixed RZ reuploading.
  - QRC-D: dissipative spin-QRC with fixed mild local dissipation.
- Comparator: globally calibrated, then frozen, feature-matched sparse random leaky ESN.
- Readout: same ridge-readout protocol for QRC and ESN.

The strongest supported scientific claim is conditional: QRC usefulness is selective,
measurable, and predictable from dataset properties. The strongest empirical QRC-usefulness
region is slow, stateful forecasting with drifting trends, regime changes, persistent
low-frequency structure, long memory, and temporally correlated fluctuations.

## Current Results

Validation split (`results_v5_publication/tables/v5_split_summary.csv`):

- Rows: 10,000.
- Families: 16.
- Base generators: 52.
- Best-QRC win rate: 32.39%.
- Best-QRC useful rate at `best_qrc_advantage_vs_esn >= 0.05`: 12.53%.
- Mean best-QRC advantage: -0.0584 NRMSE.
- Mean frozen ESN NRMSE: 0.8049.
- Mean best-QRC NRMSE: 0.8633.

Discovery-trained validation meta-model (`results_v5_publication/v5_publication_manifest.json`):

- Regression R2: 0.4072.
- Regression MAE: 0.1006.
- Classification ROC-AUC: 0.8357.
- Classification PR-AUC: 0.4305.
- Classification Brier score: 0.0880.

Strongest validation regime:

- Drifting trend / `unit_root_trend`: 353 rows, 47.9% useful rate, 51.8% win rate,
  mean advantage +0.240.
- Strongest extracted rule pocket: high trend strength plus low spectral centroid,
  420 validation rows, 47.4% useful rate, mean advantage +0.209.

## Key Artifacts

- `results_calibration_v5/frozen_v5_config.json`: frozen QRC-M/QRC-E/QRC-D and ESN config.
- `results_frontier_v4_property/frontier_property_atlas.csv`: 50,000-row property atlas.
- `results_frontier_v4_selection/frontier_evaluation_selection.csv`: 20,000-row target-free
  discovery/validation selection.
- `results_frontier_v5_discovery/frontier_discovery_evaluated_v5_multi_qrc.csv`: 10,000
  discovery labels.
- `results_frontier_v5_validation/frontier_validation_evaluated_v5_multi_qrc.csv`: 10,000
  prospective-validation labels.
- `results_v5_publication/`: v5 publication tables, figures, HTML index, and report.
- `paper/`: LaTeX paper source and compiled PDF.
- `TRIAGE.md`: public CSV contract and interpretation boundary for the triage tool.

## Triage Tools

The repository now includes `qrc_dataset_profiler.run_triage` and
`qrc_dataset_profiler.run_triage_web`, user-facing screening tools for new univariate CSV
time series. They compute the 30 atlas descriptors, fit the discovery-only meta-model from
the frozen v5 discovery table, add atlas-support/OOD scoring, and report whether QRC is
worth testing against the frozen ESN.

These tools are pre-benchmark screening signals only. They do not run QRC or ESN on the
submitted dataset and do not prove a QRC win.

## Claim Boundary

Allowed:

- A protocol-local regime map for QRC usefulness against a frozen feature-matched ESN.
- Dataset-level triage: measured time-series properties can identify where QRC deserves
  testing.
- Selective advantage: QRC is useful in structured pockets, not on average.

Not allowed:

- Broad average QRC superiority.
- Computational quantum advantage.
- A proven entanglement, coupling, dissipation, or interference mechanism.
- A universal predictor for completely unseen dataset families without support/OOD checks.

## Verification

Current expected repository checks:

```bash
PYTHONPATH=src python -m pytest
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```
