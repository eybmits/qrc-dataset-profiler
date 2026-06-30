# QRC Usefulness Atlas Report

## Headline

The 1000-row atlas identifies 36 QRC-useful datasets, with nonstationary as the clearest useful family.

## Dataset And Labels

- Atlas size: `1000` datasets from the parameterized sweep (synthetic=1000).
- `qrc_useful`: `36` rows.
- `near_tie`: `246` rows.
- `baseline_preferred`: `718` rows.
- Label target: `qrc_advantage = nrmse_esn_matched - nrmse_qrc_spin`.

## Main Map Result

- Strongest QRC-useful family: `nonstationary` with qrc-useful rate `0.107`.
- Overall mean QRC advantage CI: `[-0.4338, -0.3855]`.
- Atlas in-sample row-level category agreement: `0.845`.

## Meta-Model

- Cross-validated regression R2: `0.7566`.
- Cross-validated ROC-AUC for qrc-useful classification: `0.8145`.
- Top features: `spectral_entropy,perm_entropy,pred_nrmse_gbm,snr_db,dom_freq,ac_timescale,sample_entropy,dfa_alpha`.
- Anti-circularity without direct predictability proxies: R2 `0.7550`, ROC-AUC `0.8081`.

## Attribution Control

- Corrected paired J=1 vs J=0 overall effect: `0.0211`.
- 95% CI: `[0.0157, 0.0262]`.
- Interpretation: `positive`.

## Claim Boundary

This supports a dataset-categorization claim: the fixed Spin-QRC is selectively useful in identifiable synthetic time-series regimes. It does not establish broad average QRC superiority, fundamental quantum advantage, or a coupling/entanglement mechanism.
