"""Standard-Spin v1 exact-statevector reservoir.

The implementation vendors the minimal statevector machinery needed by the
frozen protocol: single-qubit gates, diagonal Ising-ZZ phases, projective input
reset, virtual-node measurements, and finite-shot Pauli expectation noise.
Couplings are the uniform protocol values J and h; the seed affects only
finite-shot noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _rx(theta: float) -> ComplexArray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> ComplexArray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> ComplexArray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def _apply_single_qubit_gate(state: ComplexArray, gate: ComplexArray, qubit: int, n_qubits: int) -> ComplexArray:
    tensor = state.reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    restored = np.moveaxis(updated, 0, qubit)
    return np.ascontiguousarray(restored.reshape(-1))


def _bit_mask(qubit: int, n_qubits: int) -> int:
    return 1 << (n_qubits - qubit - 1)


def _z_signs(n_qubits: int) -> FloatArray:
    indices = np.arange(2**n_qubits)
    signs = np.empty((n_qubits, indices.size), dtype=np.float64)
    for qubit in range(n_qubits):
        mask = _bit_mask(qubit, n_qubits)
        signs[qubit] = np.where((indices & mask) == 0, 1.0, -1.0)
    return signs


def _add_shot_noise(features: FloatArray, shots: int, rng: np.random.Generator) -> FloatArray:
    shots = max(int(shots), 1)
    clipped = np.clip(features, -1.0, 1.0)
    sigma = np.sqrt(np.maximum(1.0 - clipped**2, 0.0) / shots)
    noisy = clipped + rng.normal(0.0, sigma, clipped.shape)
    return np.clip(noisy, -1.0, 1.0).astype(np.float64)


@dataclass
class StandardSpinV1:
    """Fixed transverse-field Ising spin reservoir from PROTOCOL.md section 5."""

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

    def __post_init__(self) -> None:
        if not (1 <= int(self.n_qubits) <= 10):
            raise ValueError("StandardSpinV1 supports 1..10 exact-statevector qubits")
        if int(self.depth) < 1:
            raise ValueError("depth must be >= 1")
        if int(self.virtual_nodes) < 1:
            raise ValueError("virtual_nodes must be >= 1")
        if self.topology not in {"ring", "chain"}:
            raise ValueError("topology must be 'ring' or 'chain'")
        if self.shots is not None and int(self.shots) < 1:
            raise ValueError("shots must be a positive integer or None")

    @property
    def pairs(self) -> list[tuple[int, int]]:
        pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
        if self.topology == "ring" and self.n_qubits > 1:
            pairs.append((self.n_qubits - 1, 0))
        return pairs

    @property
    def feature_dim(self) -> int:
        return min(self.virtual_nodes, self.depth) * (self.n_qubits + len(self.pairs))

    def transform(self, inputs: np.ndarray) -> FloatArray:
        """Advance one persistent reservoir through a scalar driving sequence."""

        u01 = self._scale_inputs(inputs)
        state = self._zero_state()
        rng = np.random.default_rng(self.seed)
        signs_z = _z_signs(self.n_qubits)
        signs_zz = np.vstack([signs_z[i] * signs_z[j] for i, j in self.pairs]) if self.pairs else np.empty((0, state.size))
        zz_phase = self._zz_phase(signs_zz)
        record_from = self.depth - min(self.virtual_nodes, self.depth)

        out = np.empty((u01.size, self.feature_dim), dtype=np.float64)
        for t, u_t in enumerate(u01):
            state = self._inject_input(state, float(u_t))
            row_parts: list[FloatArray] = []
            rz_gate = _rz(float(np.pi * u_t))
            for layer in range(self.depth):
                if self.reupload:
                    for qubit in range(self.n_qubits):
                        state = _apply_single_qubit_gate(state, rz_gate, qubit, self.n_qubits)
                state *= zz_phase
                rx_gate = _rx(float(2.0 * self.h * self.dt))
                for qubit in range(self.n_qubits):
                    state = _apply_single_qubit_gate(state, rx_gate, qubit, self.n_qubits)
                if layer >= record_from:
                    row_parts.append(self._measure_z_zz(state, signs_z, signs_zz))
            features = np.concatenate(row_parts)
            if self.shots is not None:
                features = _add_shot_noise(features, self.shots, rng)
            out[t] = np.clip(features, -1.0, 1.0)
        return out

    def _zero_state(self) -> ComplexArray:
        state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        return state

    def _inject_input(self, state: ComplexArray, u_t: float) -> ComplexArray:
        tensor = state.reshape((2,) * self.n_qubits).copy()
        slc = [slice(None)] * self.n_qubits
        slc[0] = 1
        tensor[tuple(slc)] = 0.0
        state = tensor.reshape(-1)
        norm = float(np.linalg.norm(state))
        if norm < 1e-12:
            state = self._zero_state()
        else:
            state = state / norm
        theta = 2.0 * np.arcsin(np.sqrt(float(np.clip(u_t, 0.0, 1.0))))
        return _apply_single_qubit_gate(state, _ry(theta), 0, self.n_qubits)

    def _zz_phase(self, signs_zz: FloatArray) -> ComplexArray:
        if signs_zz.size == 0:
            return np.ones(2**self.n_qubits, dtype=np.complex128)
        exponent = np.sum(signs_zz, axis=0)
        return np.exp(-0.5j * float(2.0 * self.J * self.dt) * exponent).astype(np.complex128)

    @staticmethod
    def _measure_z_zz(state: ComplexArray, signs_z: FloatArray, signs_zz: FloatArray) -> FloatArray:
        probs = np.abs(state) ** 2
        z = signs_z @ probs
        zz = signs_zz @ probs if signs_zz.size else np.empty(0, dtype=np.float64)
        return np.concatenate([z, zz]).astype(np.float64)

    @staticmethod
    def _scale_inputs(inputs: np.ndarray) -> FloatArray:
        x = np.asarray(inputs, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return np.asarray([], dtype=np.float64)
        finite = np.isfinite(x)
        if not finite.any():
            z = np.zeros_like(x, dtype=np.float64)
        else:
            idx = np.arange(x.size)
            filled = np.interp(idx, idx[finite], x[finite]) if not finite.all() else x
            mu = float(np.mean(filled))
            sd = float(np.std(filled))
            z = np.zeros_like(filled, dtype=np.float64) if sd < 1e-12 else (filled - mu) / sd
        return (0.5 * (np.tanh(z) + 1.0)).astype(np.float64)
