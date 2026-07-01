# Reproducibility Guide

This repository is organized so the paper-facing v5 result can be checked without rerunning
the full expensive atlas, while the full long-run commands remain documented and
checkpointable.

## Environment

```bash
pip install -e ".[dev]"
```

Optional packages improve some extended descriptors but are not required for the committed
v5 paper package:

```bash
pip install -e ".[dev,stats,backstop]"
```

## Fast Verification

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_publication --out results_v5_publication
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf mapping_quantum_reservoir_advantage.pdf
```

Expected test status at the current checkpoint: all tests pass, with numerical conditioning
warnings from small ridge/ESN test cases.

## Paper Rendering

The LaTeX entry point is:

```text
paper/main.tex
```

The tracked compiled paper is:

```text
paper/mapping_quantum_reservoir_advantage.pdf
```

The build directory is intentionally ignored by git:

```text
paper/build/
```

## Full v5 Atlas Regeneration

The full regeneration is expensive. Use checkpointing and `caffeinate` on macOS if the
machine may sleep.

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier property-atlas --taxonomy v4 --out results_frontier_v4_property --n-per-template 500 --fast --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_frontier select --property-atlas results_frontier_v4_property/frontier_property_atlas.csv --out results_frontier_v4_selection --n-discovery 10000 --n-validation 10000 --selection-protocol v4
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_protocol calibrate --out results_calibration_v5 --fast
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_protocol evaluate-selection --selection results_frontier_v4_selection/frontier_evaluation_selection.csv --calibration-config results_calibration_v5/frozen_v5_config.json --out results_frontier_v5_discovery --split discovery --fast --seeds 1 --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_protocol evaluate-selection --selection results_frontier_v4_selection/frontier_evaluation_selection.csv --calibration-config results_calibration_v5/frozen_v5_config.json --out results_frontier_v5_validation --split validation --fast --seeds 1 --checkpoint-every 100
PYTHONPATH=src python -m qrc_dataset_profiler.run_v5_publication --out results_v5_publication
```

## User-Facing Triage

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_triage \
  --series path/to/my_series.csv \
  --column value \
  --name my_dataset
```

See `TRIAGE.md` for CSV requirements and claim boundaries.

## Claim Boundary

The reproducible v5 package supports a conditional regime-atlas claim under the declared
frozen QRC/ESN protocols. It does not establish broad average QRC superiority,
computational quantum advantage, or a proven quantum mechanism.
