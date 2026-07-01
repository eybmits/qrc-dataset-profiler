# QRC Triage Tool

This document defines the user-facing input contract for testing whether a new
time series is likely to be worth evaluating with QRC.

The triage tool is a screening algorithm. It does not run QRC or ESN on the
submitted dataset. It computes dataset descriptors, compares them with the
atlas, and reports whether QRC is worth testing against the frozen ESN protocol.

## Public Browser Analyzer

The project website includes a static, globally accessible CSV analyzer:

```text
https://eybmits.github.io/qrc-dataset-profiler/#try
```

This version runs directly in the visitor's browser. The CSV is not uploaded to
a server. It computes a compact set of JavaScript-friendly markers, loads
`docs/assets/triage_model.json`, and reports a plain-language recommendation:

- `QRC is worth testing`;
- `QRC may be worth testing`;
- `ESN is probably enough first`;
- `Run a direct benchmark` when the series is too different from atlas support.

The browser analyzer is intentionally lighter than the full Python triage stack.
Its exported model is trained on the v5 discovery atlas using browser-computable
features only. The asset records held-out validation quality; at export time the
browser subset reached ROC-AUC 0.806 on the v5 validation split. Treat it as a
fast screening tool for researchers, not as the paper-grade reference analysis.

Regenerate the public browser model with:

```bash
python scripts/export_static_triage_model.py
```

## Accepted CSV Shape

The current triage CLI supports univariate forecasting datasets:

```csv
value
1.02
1.08
1.05
1.11
```

or a timestamped table with one chosen numeric target column:

```csv
timestamp,value
2024-01-01,1.02
2024-01-02,1.08
2024-01-03,1.05
```

Run:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_triage \
  --series path/to/my_series.csv \
  --column value \
  --name my_dataset
```

For a headerless one-column CSV:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_triage \
  --series path/to/my_series.csv \
  --no-header \
  --column 0
```

Machine-readable output:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_triage \
  --series path/to/my_series.csv \
  --column value \
  --format json \
  --out triage_report.json
```

## Full Python Web Interface

The full 30-feature triage path is also available as a Python-backed browser
interface:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_triage_web --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

This webpage accepts the same univariate CSV shape as the CLI. It sends the CSV
text to a local `/api/triage` endpoint, computes the 30 descriptors on the
Python backend, and shows a plain-language QRC priority:

- whether QRC should be prioritized, tested if available, deprioritized, or benchmarked
  directly because the dataset is outside familiar atlas support;
- the expected QRC error change versus ESN;
- a QRC-priority score;
- similarity to known atlas examples;
- simple markers such as drifting trend, slow low-frequency structure, persistence, long
  autocorrelation, and volatility memory.

## Data Assumptions

- Rows must be ordered from oldest to newest.
- The selected column must be numeric or coercible to numeric.
- The tool treats the series as evenly sampled. Timestamp gaps are not modeled.
- Missing values are allowed and internally filled, but many missing values make
  the recommendation less reliable.
- Multivariate forecasting is not supported yet. Choose one target channel.
- The default analysis window is the last 800 samples, matching the fast atlas
  scale. Use `--window head`, `--window full`, or `--max-length` to override.
- A few hundred samples are recommended. Very short series can make fragile
  descriptors invalid.

## Relation To The 50,000-Row Atlas

The synthetic candidate atlas is scalar/univariate:

- 50,000 / 50,000 rows have `n_channels = 1`.
- 45,500 / 50,000 rows are standard forecasting tasks: one ordered scalar series.
- 4,500 / 50,000 rows are input-driven memory tasks with scalar input `u[t]` and
  scalar target `y[t]`.

The current public triage CLI implements the forecasting case, which covers most
of the atlas and most common real-world univariate forecasting datasets. The
input-driven atlas rows naturally have a two-column shape:

```csv
input,target
0.21,0.07
0.93,0.24
0.44,0.19
```

Input-driven user triage is not exposed yet. Until that extension exists, use
the one-column forecasting path for ordinary time-series forecasting datasets.

## Algorithm

For one submitted series, the triage command:

1. Reads the selected numeric column.
2. Computes the same 30 Tier-A descriptors used by the v5 atlas.
3. Fits the lightweight discovery-only meta-model from
   `results_frontier_v5_discovery/frontier_discovery_evaluated_v5_multi_qrc.csv`.
4. Predicts best-QRC advantage versus the frozen ESN.
5. Predicts the probability that QRC is useful at the atlas threshold
   `best_qrc_advantage_vs_esn >= 0.05`.
6. Computes atlas-support/OOD scores from the discovery table.
7. Prints a conservative recommendation.

Recommendations are:

- `high_priority_qrc_test`: the dataset lies in a region where QRC is strongly
  worth testing.
- `worth_testing_qrc_if_available`: QRC is plausible, but not a high-confidence
  priority.
- `esn_first_qrc_low_priority`: the atlas suggests the ESN should remain the
  default.
- `outside_atlas_support__run_direct_benchmark`: the dataset is too far from the
  synthetic atlas support; do not trust the triage score alone.

## Claim Boundary

Triage output is not a proof of quantum advantage on a new dataset. It is a
pre-benchmark screening signal. A positive recommendation means: this dataset is
worth testing with the frozen QRC/ESN benchmark. A negative recommendation means:
the ESN is the better default unless there is an independent reason to test QRC.
