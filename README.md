# qrc_dataset_profiler

Unified protocol to characterize time-series datasets (synthetic + real) with the **same
fields**, then test a fixed **Standard-Spin v1** quantum reservoir against classical baselines
to learn *which dataset properties* explain when Spin-QRC wins.

- **`PROTOCOL.md`** — the frozen contract (datasets, schema v1, Standard-Spin v1). Read this first.
- **`src/qrc_dataset_profiler/spec.py`** — machine-readable schema (`DatasetSpec`, `Dataset`, `DatasetRecord`).

## Status
- **Increment 1** (in progress): generators (20 datasets), schema-v1 property estimators,
  smoke `run_profile` → `catalog.parquet` + correlation/coverage report + ground-truth validation.
- **Increment 2** (planned): Standard-Spin v1 reservoir + baselines + targets + meta-model.

Increment 2 study runs use a fair shared evaluation protocol for linear, GBM, ESN, and
QRC: the same `build_task` alignment, the same reservoir/readout washout, and the same
train/validation/test split. Forecast datasets default to `horizon = round(ac_timescale)`
to avoid trivial one-step oversampled-flow forecasts; pass `--horizon N` to override this.
Input-driven datasets always use horizon 1.

## Quickstart (after Increment 1)
```bash
pip install -e ".[dev]"
python -m qrc_dataset_profiler.run_profile --smoke
pytest -q
python -m qrc_dataset_profiler.run_study --smoke
```
