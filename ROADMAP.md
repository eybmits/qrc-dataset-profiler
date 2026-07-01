# Roadmap

## Completed Paper-Facing Milestone: v5 Regime Atlas

The project has reached a paper-facing reproducible benchmark state:

- v4 taxonomy property atlas: 50,000 synthetic candidates across 16 broad process families.
- Frozen target-free evaluation selection: 20,000 rows.
- Frozen v5 comparison protocol:
  - QRC-M, QRC-E, QRC-D calibrated once globally.
  - Sparse random leaky ESN calibrated once globally.
  - No per-dataset reservoir tuning.
  - Same feature dimension and ridge readout protocol.
- v5 publication package:
  - seven figure sets,
  - summary/robustness/rule/feature tables,
  - HTML report,
  - LaTeX paper source and compiled PDF.
- Public triage tool for new univariate CSV time series.

## Reproducibility Target

The repository should stay reproducible from committed artifacts and deterministic commands:

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_publication --out results_v5_publication
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Long-running full regeneration remains checkpointed:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier property-atlas --taxonomy v4 --out results_frontier_v4_property --n-per-template 500 --fast --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier select --property-atlas results_frontier_v4_property/frontier_property_atlas.csv --out results_frontier_v4_selection --n-discovery 10000 --n-validation 10000 --selection-protocol v4
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_protocol calibrate --out results_calibration_v5 --fast
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_protocol evaluate-selection --selection results_frontier_v4_selection/frontier_evaluation_selection.csv --calibration-config results_calibration_v5/frozen_v5_config.json --out results_frontier_v5_discovery --split discovery --fast --seeds 1 --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_protocol evaluate-selection --selection results_frontier_v4_selection/frontier_evaluation_selection.csv --calibration-config results_calibration_v5/frozen_v5_config.json --out results_frontier_v5_validation --split validation --fast --seeds 1 --checkpoint-every 100
```

## Near-Term Cleanup

- Keep README, STATUS, PROTOCOL, TRIAGE, and the paper text synchronized with the v5 claim
  boundary.
- Keep LaTeX build products out of git while tracking the stable compiled paper PDF.
- Preserve generated checkpoints and manifests needed for long-run recovery.
- Keep the triage tool limited to screening language; it must not imply proof of QRC
  advantage on a submitted dataset.

## Optional Scientific Extensions

These are not required for the current paper claim:

- Label the remaining 30,000 candidate rows as a post-freeze 50,000-label stress replication.
- Add input-driven user triage for two-column `input,target` CSVs.
- Add broader real-world probes as external interpolation checks, not main evidence.
- Add matched long-memory ESN controls in the slow-stateful high-usefulness pocket.
- Add mechanistic follow-up experiments on reservoir memory kernels, latent-state
  recoverability, coupling, dissipation, and encoding controls.

## Claim Policy

Allowed now:

- Conditional, protocol-local QRC usefulness is learnable as a dataset-regime map.
- The strongest current QRC-usefulness regime is slow, stateful forecasting.
- Practitioners can use the triage tool to decide whether QRC is worth testing.

Not allowed:

- Broad average QRC superiority.
- Computational quantum advantage.
- A proven quantum-mechanism explanation.
