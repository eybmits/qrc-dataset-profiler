"""Increment-2 reservoir placeholder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardSpinV1:
    """Fixed transverse-field Ising spin reservoir from PROTOCOL.md section 5.

    Increment 2 must implement a persistent exact-statevector reservoir with
    Ising ZZ couplings, transverse X field, RZ input encoding, virtual nodes,
    and optional finite-shot Pauli noise. Increment 1 intentionally exposes the
    public constructor only.
    """

    n_qubits: int = 8
    J: float = 1.0
    h: float = 1.0
    dt: float = 0.25
    depth: int = 5
    topology: str = "ring"
    virtual_nodes: int = 5
    reupload: bool = True
    shots: int | None = None
    seed: int = 0

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("StandardSpinV1 is scheduled for Increment 2; see PROTOCOL.md section 5.")
