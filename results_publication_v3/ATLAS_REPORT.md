# QRC Usefulness Atlas Report

## Headline

The 1000-row atlas identifies 51 QRC-useful datasets, with chaotic_flow as the clearest useful family.

## Dataset And Labels

- Atlas size: `1000` datasets from the parameterized sweep (synthetic=1000).
- `qrc_useful`: `51` rows.
- `near_tie`: `426` rows.
- `baseline_preferred`: `523` rows.
- Label target: `qrc_advantage = nrmse_esn_matched - nrmse_qrc_spin`.

## Main Map Result

- Strongest QRC-useful family: `chaotic_flow` with qrc-useful rate `0.115`.
- Overall mean QRC advantage CI: `[-0.2265, -0.1817]`.
- Atlas in-sample row-level category agreement: `0.778`.

## Meta-Model

- Cross-validated regression R2: `0.3220`.
- Cross-validated ROC-AUC for qrc-useful classification: `0.7691`.
- Top features: `perm_entropy,lyapunov,snr_db,pred_nrmse_gbm,spectral_entropy,adf_p,ac_timescale,nl_gain`.
- Anti-circularity without direct predictability proxies: R2 `0.3251`, ROC-AUC `0.7492`.

## Attribution Control

- Corrected paired coupled-vs-J=0 overall effect: `0.1856`.
- 95% CI: `[0.1614, 0.2106]`.
- Interpretation: `positive`.

## Claim Boundary

This supports a dataset-categorization claim: the fixed Spin-QRC is selectively useful in identifiable synthetic time-series regimes. It does not establish broad average QRC superiority, fundamental quantum advantage, or a coupling/entanglement mechanism.
