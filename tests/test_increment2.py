from dataclasses import replace

import numpy as np

import qrc_dataset_profiler.baselines as baselines_module
from qrc_dataset_profiler.baselines import (
    esn_matched_baseline,
    gbm_baseline,
    linear_baseline,
    qrc_nrmse,
)
from qrc_dataset_profiler.generators import ALL_SPECS, generate
from qrc_dataset_profiler.reservoir import StandardSpinV1
from qrc_dataset_profiler.spec import Dataset, DatasetSpec


def test_standard_spin_transform_shape_range_and_feature_dim():
    cfg = StandardSpinV1(n_qubits=4, depth=5, virtual_nodes=3)
    x = np.sin(np.linspace(0.0, 12.0, 300))
    features = cfg.transform(x)

    assert features.shape == (300, cfg.feature_dim)
    assert cfg.feature_dim == 3 * (4 + 4)
    assert np.isfinite(features).all()
    assert np.max(np.abs(features)) <= 1.0 + 1e-12


def test_standard_spin_fading_memory_converges_after_common_drive():
    rng = np.random.default_rng(123)
    common = rng.normal(size=100)
    cfg = StandardSpinV1(n_qubits=4, depth=5, virtual_nodes=5)

    a = cfg.transform(np.concatenate([rng.normal(size=50), common]))
    b = cfg.transform(np.concatenate([rng.normal(size=50), common]))

    assert np.mean(np.abs(a[-1] - b[-1])) < 0.25


def test_esn_dimension_matches_standard_spin_feature_dim():
    cfg = StandardSpinV1(n_qubits=4)
    spec = DatasetSpec("ar", "test", "synthetic", "forecast", seed=0, length=300)
    rng = np.random.default_rng(0)
    x = np.zeros(spec.length)
    eps = rng.normal(scale=0.2, size=spec.length)
    for t in range(1, spec.length):
        x[t] = 0.7 * x[t - 1] + eps[t]
    details = esn_matched_baseline(Dataset(spec, x), qrc_cfg=cfg, seed=0, return_details=True)

    assert details["reservoir_size"] == cfg.feature_dim
    assert np.isfinite(details["nrmse"])


def test_qrc_has_no_input_memory_lift():
    assert not hasattr(baselines_module, "_input_memory_lift")


def test_esn_matched_baseline_solves_linear_memory():
    spec = next(s for s in ALL_SPECS if s.name == "linear_memory")
    ds = generate(replace(spec, length=700))
    cfg = StandardSpinV1(n_qubits=4, depth=5, virtual_nodes=5)

    esn = esn_matched_baseline(ds, qrc_cfg=cfg, seed=0)

    assert np.isfinite(esn)
    assert esn < 0.15


def test_qrc_readout_uses_reservoir_feature_width_only(monkeypatch):
    spec = next(s for s in ALL_SPECS if s.name == "narma10")
    ds = generate(replace(spec, length=120))
    cfg = StandardSpinV1(n_qubits=4, depth=5, virtual_nodes=5)
    captured: dict[str, tuple[int, int]] = {}

    def fake_evaluate_readout(features, y, splits=None):
        captured["shape"] = features.shape
        return 0.0

    monkeypatch.setattr(baselines_module, "evaluate_readout", fake_evaluate_readout)

    qrc = baselines_module.qrc_nrmse(ds, cfg, seed=0)

    assert qrc == 0.0
    assert captured["shape"][1] == cfg.feature_dim


def test_all_baselines_return_finite_nrmse_on_simple_ar_series():
    rng = np.random.default_rng(321)
    spec = DatasetSpec("ar", "test", "synthetic", "forecast", seed=321, length=500)
    x = np.zeros(spec.length)
    eps = rng.normal(scale=0.2, size=spec.length)
    for t in range(2, spec.length):
        x[t] = 0.65 * x[t - 1] - 0.2 * x[t - 2] + eps[t]
    ds = Dataset(spec, x)
    cfg = StandardSpinV1(n_qubits=4, depth=3, virtual_nodes=3)

    vals = [
        linear_baseline(ds),
        gbm_baseline(ds, seed=0),
        esn_matched_baseline(ds, qrc_cfg=cfg, seed=0),
        qrc_nrmse(ds, cfg, seed=0),
    ]
    assert np.isfinite(vals).all()


def test_nrmse_is_bounded():
    # A divergent prediction must not produce an unbounded NRMSE that would create a
    # giant spurious qrc_advantage outlier (chirp ESN previously hit 25.8).
    from qrc_dataset_profiler.baselines import _nrmse, NRMSE_CAP

    y_true = np.sin(np.linspace(0, 20, 200))
    y_pred = 1e6 * np.ones_like(y_true)
    assert _nrmse(y_true, y_pred) == NRMSE_CAP
    assert _nrmse(y_true, y_true) < 0.01
