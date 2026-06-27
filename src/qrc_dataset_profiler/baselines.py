"""Increment-2 baseline placeholders."""

from __future__ import annotations

import numpy as np


def linear_baseline(series: np.ndarray, inputs: np.ndarray | None = None, **kwargs):
    """Fit/evaluate the Ridge linear baseline planned in PROTOCOL.md section 5."""

    raise NotImplementedError("linear_baseline is scheduled for Increment 2.")


def esn_matched_baseline(series: np.ndarray, inputs: np.ndarray | None = None, **kwargs):
    """Fit/evaluate the dimension-matched leaky ESN baseline planned in PROTOCOL.md section 5."""

    raise NotImplementedError("esn_matched_baseline is scheduled for Increment 2.")


def gbm_baseline(series: np.ndarray, inputs: np.ndarray | None = None, **kwargs):
    """Fit/evaluate the GBM baseline planned in PROTOCOL.md section 5."""

    raise NotImplementedError("gbm_baseline is scheduled for Increment 2.")
