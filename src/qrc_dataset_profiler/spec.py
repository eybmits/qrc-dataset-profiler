"""Frozen schema v1 for the QRC dataset profiler.

This is the machine-readable form of PROTOCOL.md. Treat it as a contract:
add fields and bump SCHEMA_VERSION, never silently rename/repurpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import math

SCHEMA_VERSION = "v2"

# Minimum series length for fragile estimators to be considered valid.
MIN_LENGTH_LYAPUNOV = 500
MIN_LENGTH_DFA = 500
MIN_LENGTH_ZERO_ONE = 300

NAN = float("nan")


@dataclass
class DatasetSpec:
    """Declarative description of one dataset; consumed by generators.generate()."""

    name: str
    family: str
    source: str  # "synthetic" | "semi" | "real"
    task_type: str  # "forecast" | "input_driven"
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    length: int = 4000
    n_channels: int = 1
    horizon: int = 1


@dataclass
class Dataset:
    """Output of a generator.

    For task_type="forecast", `series` is the observed series and the target is the
    1-step-ahead value of `series`. For task_type="input_driven", `inputs` is the
    driving signal u and `series` is the target y. `ground_truth` holds the known
    property values (Block D); keys map to DatasetRecord ground-truth fields.
    """

    spec: DatasetSpec
    series: Any  # np.ndarray, shape (length,)
    inputs: Any | None = None  # np.ndarray for input_driven tasks, else None
    ground_truth: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetRecord:
    """One row of the catalog. All property fields default to NaN; fragile
    estimators carry a *_valid flag (default False). See PROTOCOL.md §4."""

    # --- Block A: identity / provenance ---
    dataset_id: str = ""
    name: str = ""
    family: str = ""
    source: str = ""
    task_type: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    schema_version: str = SCHEMA_VERSION
    n_channels: int = 1
    length: int = 0
    horizon: int = 1
    missing_frac: float = 0.0
    irregular_sampling: bool = False

    # --- Block B: basic statistics ---
    mean: float = NAN
    std: float = NAN
    skew: float = NAN
    kurtosis: float = NAN

    # --- Block C: the 10 axes ---
    # 1 Memory
    ac_timescale: float = NAN
    ami_first_min: float = NAN
    mem_capacity: float = NAN
    mem_capacity_valid: bool = False
    # 2 Linearity
    r2_linear: float = NAN
    # 3 Nonlinearity
    nl_gain: float = NAN
    # 4 Noise
    snr_db: float = NAN
    snr_valid: bool = False
    # 5 Chaos
    lyapunov: float = NAN
    lyapunov_valid: bool = False
    zero_one_K: float = NAN
    zero_one_valid: bool = False
    # 6 Frequency
    spectral_entropy: float = NAN
    dom_freq: float = NAN
    spectral_flatness: float = NAN
    # 7 Stationarity (computed on raw series)
    adf_p: float = NAN
    kpss_p: float = NAN
    n_diffs: int = 0
    # 8 Long-range dependence
    dfa_alpha: float = NAN
    dfa_valid: bool = False
    # 9 Complexity
    perm_entropy: float = NAN
    sample_entropy: float = NAN
    # 10 Rescaled-range long memory
    hurst_rs: float = NAN
    # 10 Predictability
    forecastability: float = NAN
    pred_nrmse_linear: float = NAN
    pred_nrmse_gbm: float = NAN
    predictability_gap_linear_gbm: float = NAN

    # --- Block D: ground truth (synthetic only; NaN/None for real) ---
    true_lyapunov: float = NAN
    true_memory_order: float = NAN
    true_n_frequencies: float = NAN
    true_hurst: float = NAN
    is_chaotic: float = NAN  # 1.0 / 0.0 / NaN(unknown)

    # --- Block E: targets (filled in Increment 2) ---
    nrmse_linear: float = NAN
    nrmse_esn_matched: float = NAN
    nrmse_qrc_spin: float = NAN
    nrmse_gbm: float = NAN
    qrc_advantage: float = NAN

    def to_row(self) -> dict[str, Any]:
        """Flatten to a parquet-friendly dict (params serialized as repr)."""
        row = asdict(self)
        row["params"] = repr(row["params"])
        return row


# Column groups (names only) — useful for the meta-model to select the core axes.
CORE_AXIS_FIELDS: tuple[str, ...] = (
    "ac_timescale", "ami_first_min", "mem_capacity",
    "r2_linear", "nl_gain", "snr_db",
    "lyapunov", "zero_one_K",
    "spectral_entropy", "dom_freq", "spectral_flatness",
    "adf_p", "kpss_p", "n_diffs",
    "dfa_alpha", "perm_entropy", "sample_entropy", "hurst_rs",
    "forecastability", "pred_nrmse_gbm",
)
TIER_A_UPGRADE_FIELDS: tuple[str, ...] = (
    "predictability_gap_linear_gbm",
    "ext_volatility_ac1",
    "ext_arch_lm5",
    "ext_recurrence_rate",
    "ext_recurrence_determinism",
    "ext_psd_slope",
    "ext_spectral_centroid",
    "ext_trend_strength",
    "ext_changepoint_count",
    "ext_lz_complexity",
)
FRONTIER_TIER_A_FIELDS: tuple[str, ...] = CORE_AXIS_FIELDS + TIER_A_UPGRADE_FIELDS
TIER_B_EXTENDED_FIELDS: tuple[str, ...] = (
    "ext_approx_entropy_m2",
    "ext_spectral_bandwidth",
    "ext_spectral_rolloff85",
    "ext_zero_crossing_rate",
    "ext_turning_point_rate",
    "ext_outlier_rate_3sigma",
    "ext_spike_rate_mad6",
    "ext_seasonality_strength",
    "ext_fnn_fraction",
    "ext_corr_dim_approx",
    "ext_bds_like",
    "ext_zero_fraction",
    "ext_cv2_positive",
)
GROUND_TRUTH_FIELDS: tuple[str, ...] = (
    "true_lyapunov", "true_memory_order", "true_n_frequencies", "true_hurst", "is_chaotic",
)
TARGET_FIELDS: tuple[str, ...] = (
    "nrmse_linear", "nrmse_esn_matched", "nrmse_qrc_spin", "nrmse_gbm", "qrc_advantage",
)


def is_valid(value: float) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))
