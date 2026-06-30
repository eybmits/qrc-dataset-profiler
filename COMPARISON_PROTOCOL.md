# Frozen Comparison Protocol v3

Date frozen: 2026-06-29.

This file freezes the publication-facing model comparison. `PROTOCOL.md` remains the
schema and dataset contract. This file defines the primary QRC-vs-classical comparison
that should be used for the next full rerun before paper claims are made.
`FRONTIER_PROTOCOL.md` defines the next scaled atlas: 30 Tier-A explanatory features,
candidate-pool expansion, frontier sampling, validation gates, and runtime staging.

Protocol v3 supersedes v2 as the paper-facing fairness protocol. The previous v2 comparison
is retained as a conservative stress test: fixed QRC versus validation-tuned ESN.

## Decision

The primary scientific question is dataset categorization:

> Which measured time-series properties predict when a fixed spin-QRC protocol is useful
> relative to a strong, matched classical reservoir baseline?

The protocol does not seek, and must not claim, fundamental quantum advantage, broad
average QRC superiority, or a coupling/entanglement mechanism.

The primary atlas is synthetic-only: 50 named synthetic benchmark generators expanded to
1000 synthetic parameterized rows and characterized by 20 predefined measured properties.
Santa Fe laser is retained only as an external real-data validation bridge and must be
reported separately from the primary synthetic atlas.

## Three-Layer Evaluation Protocol

Layer 1 is the paper-facing fairness test:

- Calibrate QRC once on held-out synthetic calibration datasets.
- Calibrate ESN once on the same held-out calibration datasets.
- Freeze both reservoirs globally.
- Run the full atlas with no per-dataset reservoir hyperparameter tuning for either model.

Layer 2 is the strong classical baseline test:

- Use the frozen QRC from Layer 1.
- Compare it against a validation-tuned sparse ESN.
- Report this as conservative against QRC, not as the symmetric fairness comparison.

Layer 3 is matched tuning-budget robustness:

- Either both reservoirs receive no per-dataset reservoir tuning, or both receive an equal
  small validation-grid budget.
- Report this as a robustness check for sensitivity to tuning budget.

## Literature Rationale

- Fujii and Nakajima establish QRC as input-driven quantum dynamics with linear readout,
  input injection into a qubit, virtual nodes, and ESN comparisons. This supports a
  persistent spin-reservoir protocol with trained readout only.
- Dissipative-QRC results motivate weak open-system relaxation/dephasing as a standard
  way to enforce fading memory while remaining hardware realistic. This protocol therefore
  makes fixed weak dissipation part of the primary QRC rather than tuning it per dataset.
- Nakajima et al. support temporal/spatial multiplexing as a standard QRC capacity
  enhancement, but this protocol keeps only temporal multiplexing in the primary model
  to avoid extra architecture search.
- Martinez-Pena et al. support transverse-field spin-network QRC and motivate operating
  near thermal/ergodic regimes, but the present project's paired `J=1` vs `J=0` result is
  null/mixed overall, so no mechanism claim is allowed.
- Govia et al. show that QRC input encodings can themselves supply nonlinear transformations.
  Therefore, the primary QRC uses the simplest defensible input-qubit injection. Circuit-style
  reuploading is retained only as an encoding ablation.
- Lukosevicius' ESN guide identifies the canonical ESN ingredients: random reservoir
  weights, sparsity, spectral radius, dense input weights, input scaling, leak rate, and
  linear readout. Therefore, the primary classical reservoir must be a sparse random
  leaky ESN, not the earlier simple-cycle ESN. For the primary fairness claim, ESN
  reservoir hyperparameters are frozen globally rather than validation-selected per dataset.
- Gauthier et al. motivate NVAR/next-generation reservoir computing as an important
  additional robustness baseline, not as the primary ESN comparator.

## Primary QRC

Name: `spin_qrc_dissipative_v2`.

Fixed settings:

- Model: transverse-field Ising spin reservoir with fixed weak local dissipation.
- Hamiltonian: `H = J * sum_<i,j> Z_i Z_j + h * sum_i X_i`.
- Topology: ring nearest-neighbor ZZ couplings.
- Qubits: `N = 8` for publication rerun; `N = 6` only for smoke/fast development.
- Couplings: `J = 1.0`, `h = 1.0`.
- Trotter step: `dt = 0.25`.
- Depth: `D = 5`.
- Virtual nodes: `V = 5`.
- Observables: all single-qubit `Z_i` plus nearest-neighbor ring `Z_i Z_j`.
- Feature dimension: `V * (N + N)` for ring topology, so `80` for the publication rerun.
- State: persistent reservoir state across time.
- Input injection: reset/reinject the first qubit only.
- Input preprocessing: fill finite values, compute mean/std on the train split only,
  transform with `0.5 * (tanh(z_train_scaled) + 1)`, then clip to `[0, 1]`.
- Primary encoding: no per-layer `RZ(pi*u)` reuploading.
- Dissipation: local amplitude damping with per-layer probability `p1 = 0.02` and local
  dephasing with per-layer probability `p_phi = 0.01`, both fixed globally.
- Dissipation implementation: quantum-trajectory unraveling for scalable atlas runs;
  exact density-matrix evolution is retained only for small-system verification.
- Readout: ridge regression only; alpha selected on validation split; final fit on
  train+validation; report test NRMSE.
- Seeds: trajectory seeds are averaged over the frozen seed set. Finite-shot variants use
  matched seeds and are sensitivity analyses only.

## Primary ESN

Name: `frozen_sparse_random_leaky_esn_matched_v3`.

Fixed settings:

- Reservoir: tanh leaky ESN with sparse random recurrent weights.
- Reservoir size: exactly matched to the QRC feature dimension.
- Recurrent density: `0.1`.
- Input weights: dense uniform `[-1, 1]`, scaled by the fixed global input scale.
- Bias: dense uniform `[-0.2, 0.2]`, fixed.
- Input preprocessing: train-split z-score of the same scalar drive used by QRC.
- Spectral radius: `rho = 0.9`, fixed globally.
- Leak rate: `leak = 0.3`, fixed globally.
- Input scale: `input_scale = 1.0`, fixed globally.
- Reservoir hyperparameter selection: none per dataset.
- Seeds: average over the frozen seed set `0, 1, 2` by default; use `0..4` for the
  final publication rerun if runtime permits.
- Readout: same ridge protocol and alpha grid as QRC.

This intentionally makes the primary comparison symmetric: QRC reservoir hyperparameters
and ESN reservoir hyperparameters are both calibrated once on held-out datasets and then
frozen globally. Ridge readout alpha is still selected on the validation split for both
models because the readout is the trained model component in both reservoir-computing systems.

The previous validation-tuned ESN grid is retained only as a strong-classical robustness
baseline:

- Spectral radius grid: `(0.7, 0.9, 1.0, 1.1, 1.3)`.
- Leak grid: `(0.1, 0.3, 0.6, 1.0)`.
- Input-scale grid: `(0.3, 1.0, 2.0)`.
- Selection: choose ESN hyperparameters by validation NRMSE only.

## Robustness Baselines

The following are not the primary comparator but must remain visible:

- Simple-cycle leaky ESN: reproduces legacy artifacts and tests sensitivity to deterministic
  low-complexity reservoirs.
- Validation-tuned sparse ESN: the v2 conservative stress test; useful for showing what
  remains when the classical reservoir receives per-dataset reservoir tuning.
- Linear ridge autoregression: tests whether the task is already linearly solvable.
- GBM on lagged windows: nonlinear tabular forecaster and predictability guardrail.
- NVAR/NG-RC: recommended next robustness baseline before a paper submission.
- Coherent QRC ablation: same fixed spin reservoir with `p1 = p_phi = 0`.
- QRC encoding ablation: original input injection plus `RZ(pi*u)` reuploading, reported only
  as an ablation because encoding nonlinearity can itself create apparent QRC gains.
- QRC coupling attribution: paired `J=1` vs `J=0`, identical feature dimensions and matched
  seeds. Null/negative outcomes remain first-class results.

## Primary Target

For the v3 rerun:

`qrc_advantage = nrmse_frozen_sparse_random_leaky_esn_matched_v3 - nrmse_spin_qrc_dissipative_v2`

Positive values mean the fixed QRC has lower test NRMSE than the matched sparse ESN.
Use the same usefulness thresholds as the current atlas unless explicitly bumped:

- `qrc_useful`: `qrc_advantage >= 0.05`
- `near_tie`: `abs(qrc_advantage) < 0.05`
- `baseline_preferred`: `qrc_advantage <= -0.05`

## Artifact Status

The current legacy/v2 result folders are valid for method development and robustness
comparison, but a paper claim about the symmetric standard comparison requires regenerating
the full synthetic atlas with:

```bash
PYTHONPATH=src python -m qrc_dataset_profiler.run_study --sweep --out results_sweep_v3 --comparison-protocol standard_v3 --seeds 5
```

Then rerun extended features, meta-model, analysis, usefulness map, attribution controls,
and publication figures against the v3 catalog.

## Frozen References

- Fujii, K. and Nakajima, K. Harnessing Disordered-Ensemble Quantum Dynamics for Machine
  Learning. Physical Review Applied 8, 024030 (2017). https://doi.org/10.1103/PhysRevApplied.8.024030
- Nakajima, K. et al. Boosting Computational Power through Spatial Multiplexing in Quantum
  Reservoir Computing. Physical Review Applied 11, 034021 (2019). https://doi.org/10.1103/PhysRevApplied.11.034021
- Martinez-Pena, R. et al. Dynamical Phase Transitions in Quantum Reservoir Computing.
  Physical Review Letters 127, 100502 (2021). https://doi.org/10.1103/PhysRevLett.127.100502
- Sannia, A. et al. Dissipation as a resource for quantum reservoir computing.
  Quantum 8, 1291 (2024). https://quantum-journal.org/papers/q-2024-03-20-1291/
- Govia, L. C. G. et al. Nonlinear input transformations are ubiquitous in quantum reservoir
  computing. arXiv:2107.00147 (2021). https://arxiv.org/abs/2107.00147
- Lukosevicius, M. A Practical Guide to Applying Echo State Networks. Neural Networks:
  Tricks of the Trade, Reloaded (2012). https://www.ai.rug.nl/minds/uploads/PracticalESN.pdf
- Gauthier, D. J. et al. Next generation reservoir computing. Nature Communications 12,
  5564 (2021). https://doi.org/10.1038/s41467-021-25801-2
