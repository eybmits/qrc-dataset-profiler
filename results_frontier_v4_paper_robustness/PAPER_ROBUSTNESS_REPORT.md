# Paper Robustness Report

## Core Robustness

- Best prospective feature set: `without_direct_predictability` ROC-AUC `0.790`, PR-AUC `0.224`.
- Chaos/nonlinearity/complexity-only: ROC-AUC `0.758`, PR-AUC `0.201`.
- Without chaos/nonlinearity/complexity: ROC-AUC `0.777`, PR-AUC `0.214`.
- Without direct predictability proxies: ROC-AUC `0.790`, PR-AUC `0.224`.

## Regime Enrichment

- Overall QRC-useful rate: `0.060`.
- Persistence + drift + low-frequency + moderate-complexity pocket useful rate: `0.225`.

## Real-World Probes

- `airline_passengers_monthly`: predicted advantage `-0.029`, P(useful) `0.312`, support `0.029`, OOD `True`.
- `melbourne_daily_min_temperature`: predicted advantage `-0.066`, P(useful) `0.061`, support `0.212`, OOD `False`.
- `silso_monthly_sunspots`: predicted advantage `-0.125`, P(useful) `0.096`, support `0.156`, OOD `False`.
- `noaa_mauna_loa_co2`: predicted advantage `-0.175`, P(useful) `0.300`, support `0.074`, OOD `False`.
- `fred_usd_eur_exchange`: predicted advantage `-0.240`, P(useful) `0.068`, support `0.005`, OOD `True`.
- `ett_hourly_transformer_temperature`: predicted advantage `-0.250`, P(useful) `0.065`, support `0.023`, OOD `True`.
- `pjme_hourly_load`: predicted advantage `-0.310`, P(useful) `0.079`, support `0.099`, OOD `False`.

## Metric Robustness

- Subset NRMSE mean advantage `-0.279`; NMAE mean advantage `-0.237`; QRC beats NVAR by NRMSE rate `0.225`.

## Mechanism Guardrail

- Mean paired delta J0-J*: `0.189`; fraction J* better than J0 `0.875`.

Claim boundary: these artifacts support a protocol-local regime-map and robustness claim. They do not establish broad QRC superiority, hardware quantum advantage, or an entanglement mechanism.
