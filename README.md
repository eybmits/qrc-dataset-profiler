# qrc_dataset_profiler

Unified protocol to characterize univariate time-series datasets with the same schema,
evaluate a fixed Standard-Spin v1 quantum reservoir against classical baselines, and
learn which dataset properties explain when Spin-QRC wins.

The current result is useful but intentionally conservative: the 360-dataset sweep and
meta-model show a robust property-to-advantage signal, including anti-circularity checks.
The first quantum-attribution ablation does **not** support a coupling/entanglement claim
yet, so that claim must remain unresolved until a corrected analysis is complete.

## Repository Map

- `PROTOCOL.md` - frozen protocol contract and schema semantics.
- `src/qrc_dataset_profiler/spec.py` - machine-readable schema definitions.
- `src/qrc_dataset_profiler/run_profile.py` - profile the initial catalog.
- `src/qrc_dataset_profiler/run_study.py` - build Block E baseline/QRC targets.
- `src/qrc_dataset_profiler/run_meta.py` - fit the explanatory meta-model.
- `STATUS.md` - current evidence, artifact inventory, and claim boundaries.
- `ROADMAP.md` - next planned analysis increment.

## Current Status

- **Increment 1 complete:** package scaffold, 20 dataset generators, schema-v1 property
  estimators, smoke profiler outputs, and ground-truth validation.
- **Increment 2 complete:** Standard-Spin v1 reservoir, linear/GBM/ESN baselines,
  fair shared forecasting task alignment, and Block E targets.
- **Increment 3 complete:** parameterized 360-dataset sweep, scalable study mode, and
  explanatory meta-model for QRC advantage.
- **Increment 4 planned:** paper-grade analysis suite with bootstrap intervals,
  formal anti-circularity reports, reviewer figures, and corrected quantum attribution.

## Key Local Artifacts

- `results_full/full_catalog.csv` - 19-row first full catalog with Block E targets.
- `results_sweep/sweep_catalog.csv` - 360-row parameterized sweep catalog.
- `results_meta/importances.csv` - meta-model feature importances.
- `results_meta/importance_bar.png` - importance visualization.
- `results_meta/partial_dependence.png` - partial-dependence visualization.
- `results_meta/quantum_ablation.csv` - ad-hoc J=1 vs J=0 ablation; currently a red flag.
- `results_sweep_tiny/sweep_catalog.csv` - tiny sanity sweep output.

## Quickstart

```bash
pip install -e ".[dev]"
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m qrc_dataset_profiler.run_profile --smoke --out results
PYTHONPATH=src python -m qrc_dataset_profiler.run_study --smoke --out results_full
PYTHONPATH=src python -m qrc_dataset_profiler.run_study --sweep --fast --out results_sweep
PYTHONPATH=src python -m qrc_dataset_profiler.run_meta --catalog results_sweep/sweep_catalog.csv --out results_meta
```

## Claim Boundary

The current evidence supports the narrower claim that dataset properties can predict when
the present Spin-QRC implementation beats the matched ESN baseline. It does not yet support
a strong claim that the observed advantage is caused by quantum coupling or entanglement.
