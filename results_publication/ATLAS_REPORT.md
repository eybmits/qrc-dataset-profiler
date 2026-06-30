# QRC Usefulness Atlas Report

## Headline

The 1000-row atlas identifies 204 QRC-useful datasets, with chaotic_map as the clearest useful family.

## Dataset And Labels

- Atlas size: `1000` datasets from the parameterized sweep (`960` synthetic rows plus `40` Santa Fe real-bridge windows).
- First full catalog: `50` rows, including the public `santa_fe_laser` real bridge.
- `qrc_useful`: `204` rows.
- `near_tie`: `291` rows.
- `baseline_preferred`: `505` rows.
- Label target: `qrc_advantage = nrmse_esn_matched - nrmse_qrc_spin`.

## Main Map Result

- Strongest QRC-useful family: `chaotic_map` with qrc-useful rate `0.440`.
- Overall mean QRC advantage CI: `[-0.2132, -0.1620]`.
- Atlas in-sample row-level category agreement: `0.798`.

## Meta-Model

- Cross-validated regression R2: `0.6651`.
- Cross-validated ROC-AUC for qrc-useful classification: `0.8564`.
- Top features: `r2_linear,ac_timescale,spectral_entropy,snr_db,dfa_alpha,zero_one_K,kpss_p,pred_nrmse_gbm`.
- Anti-circularity without direct predictability proxies: R2 `0.6599`, ROC-AUC `0.8558`.

## Attribution Control

- Corrected paired J=1 vs J=0 overall effect: `0.0127`.
- 95% CI: `[-0.0128, 0.0389]`.
- Interpretation: `null_or_mixed`.

## Claim Boundary

This supports a dataset-categorization claim: the fixed Spin-QRC is selectively useful in identifiable synthetic and real-bridge time-series regimes. It does not establish broad average QRC superiority, fundamental quantum advantage, or a coupling/entanglement mechanism.
