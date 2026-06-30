# Real-World External Validation

Main evidence remains the synthetic atlas. This report uses real-world benchmark probes only as external validation of the synthetic-trained regime map.

## Probe Atlas

- Real property windows: `140`.
- Real domains: `15`.
- Sources: `20`.
- OOD windows by synthetic support score: `30` / `140`.
- Mean predicted QRC advantage: `-0.130`.

## Highest-Scored Real Domains

- `domain:cloud_metrics`: n `4`, predicted advantage `-0.029`, support `0.497`, OOD rate `0.000`.
- `domain:transport`: n `1`, predicted advantage `-0.046`, support `0.029`, OOD rate `1.000`.
- `domain:traffic`: n `4`, predicted advantage `-0.060`, support `0.332`, OOD rate `0.000`.
- `domain:weather`: n `2`, predicted advantage `-0.066`, support `0.197`, OOD rate `0.000`.
- `domain:m4_monthly`: n `16`, predicted advantage `-0.068`, support `0.066`, OOD rate `0.625`.
- `domain:m4_weekly`: n `27`, predicted advantage `-0.081`, support `0.084`, OOD rate `0.259`.
- `domain:m4_quarterly`: n `4`, predicted advantage `-0.090`, support `0.028`, OOD rate `1.000`.
- `domain:m4_daily`: n `32`, predicted advantage `-0.111`, support `0.109`, OOD rate `0.250`.

## Frozen-Protocol Real Labels

- Labeled windows: `48`.
- Mean observed QRC advantage: `-0.606`.
- Median observed QRC advantage: `-0.365`.
- QRC win rate: `0.042`.
- QRC useful rate at advantage >= 0.05: `0.021`.
- Mean NMAE advantage: `-0.584`.
- Mean QRC-vs-NVAR NRMSE advantage: `-0.468`.
- Prediction R2 on labeled real probes: `-0.884`.
- Prediction ROC-AUC for real QRC-useful labels: `0.149`.

## Best Observed Real Windows

- `m4_monthly_sample_M14770_w0` (m4_monthly, predicted_unfavorable): pred `-0.249`, observed `0.094`, support `0.194`, OOD `False`.
- `m4_hourly_sample_H371_w0` (m4_hourly, domain_balanced): pred `-0.120`, observed `0.017`, support `0.332`, OOD `False`.
- `m4_weekly_sample_W15_w0` (m4_weekly, predicted_promising): pred `0.003`, observed `0.000`, support `0.035`, OOD `True`.
- `nab_cloud_cpu_value_w0` (cloud_metrics, domain_balanced): pred `-0.029`, observed `-0.001`, support `0.398`, OOD `False`.
- `m4_hourly_sample_H224_w1` (m4_hourly, predicted_unfavorable): pred `-0.239`, observed `-0.035`, support `0.134`, OOD `False`.
- `m4_hourly_sample_H224_w0` (m4_hourly, predicted_unfavorable): pred `-0.239`, observed `-0.055`, support `0.135`, OOD `False`.
- `m4_hourly_sample_H205_w1` (m4_hourly, predicted_unfavorable): pred `-0.282`, observed `-0.070`, support `0.139`, OOD `False`.
- `m4_hourly_sample_H232_w1` (m4_hourly, predicted_unfavorable): pred `-0.282`, observed `-0.084`, support `0.138`, OOD `False`.

## Claim Boundary

The real probes do not train the regime map and do not replace the synthetic atlas as the main evidence. In this pass they mainly serve as a conservative transfer check: the fixed QRC does not show broad real-world advantage against the frozen feature-matched ESN, and the synthetic ranking does not yet transfer reliably to these real probes.

Figures: `real_window_prediction_map.png`, `real_labeled_prediction_vs_observed.png`.
