# Roadmap

## Increment 4: Paper-Grade Analysis Suite

Goal: turn the current sweep and ad-hoc analyses into reproducible, reviewer-facing
evidence without overstating the quantum mechanism.

## Planned Work

- Add a formal analysis module and CLI, for example `qrc_dataset_profiler.run_analysis`,
  that reads `results_sweep/sweep_catalog.csv` and writes a deterministic analysis folder.
- Produce bootstrap confidence intervals for family-level QRC advantage and meta-model
  feature importances.
- Formalize the anti-circularity suite:
  - all core features,
  - without `r2_linear`,
  - without direct predictability proxies,
  - chaos/nonlinearity/complexity-only feature set.
- Generate reviewer-grade tables and figures for the sweep summary, meta-model robustness,
  and family-level advantage distribution.
- Replace the ad-hoc quantum ablation with a corrected attribution protocol:
  - compare matched `J=1` and `J=0` reservoirs under identical feature dimensions,
  - include matched seeds and family stratification,
  - report confidence intervals and paired differences,
  - treat negative or null results as first-class outcomes.

## Claim Policy

- Allowed now: dataset properties predict when this Spin-QRC implementation beats the
  matched ESN baseline on the current synthetic sweep.
- Not allowed now: a strong coupling, entanglement, or quantum-mechanism claim.
- Required before a paper claim: reproducible Increment-4 attribution results showing a
  robust positive paired effect for the quantum reservoir variant.

## Acceptance Criteria

- `PYTHONPATH=src python -m pytest -q` passes.
- The analysis CLI runs from committed inputs and writes all tables/figures deterministically.
- The generated summary reports the exact feature sets, seeds, thresholds, and row counts.
- Negative quantum-attribution findings remain visible rather than being filtered out.
