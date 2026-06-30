from dataclasses import replace
import json

import numpy as np
import pandas as pd

import qrc_dataset_profiler.run_study as run_study
from qrc_dataset_profiler.generators import ALL_SPECS, generate, make_real_bridge_specs, make_sweep_specs
from qrc_dataset_profiler.spec import DatasetSpec
from qrc_dataset_profiler.spec import GROUND_TRUTH_FIELDS


def test_all_specs_generate_requested_shape_or_unavailable():
    assert len(ALL_SPECS) == 50
    for spec in ALL_SPECS:
        ds = generate(spec)
        assert spec.source == "synthetic"
        assert ds.series.shape == (spec.length,)
        assert np.isfinite(ds.series).all(), spec.name
        if spec.task_type == "input_driven":
            assert ds.inputs is not None
            assert ds.inputs.shape == (spec.length,)
            assert np.isfinite(ds.inputs).all()
        else:
            assert ds.inputs is None


def test_ground_truth_keys_from_protocol_are_present():
    by_name = {spec.name: generate(spec) for spec in ALL_SPECS}
    expected = {
        "logistic_r4": {"true_lyapunov", "is_chaotic"},
        "henon": {"true_lyapunov", "is_chaotic"},
        "lorenz63": {"true_lyapunov", "is_chaotic"},
        "mackey_glass_t17": {"is_chaotic"},
        "mackey_glass_t30": {"is_chaotic"},
        "mackey_glass_t30": {"is_chaotic"},
        "henon_heiles": {"is_chaotic"},
        "narma10": {"true_memory_order"},
        "narma20": {"true_memory_order"},
        "linear_memory": {"true_memory_order"},
        "nonlinear_ipc": {"true_memory_order"},
        "ar2": {"true_memory_order"},
        "mso8": {"true_n_frequencies", "true_frequencies"},
        "quasi_periodic": {"true_n_frequencies", "true_frequencies"},
        "fbm_h08": {"true_hurst"},
    }
    for name, keys in expected.items():
        assert keys.issubset(by_name[name].ground_truth.keys())


def test_high_order_narma_is_non_degenerate():
    for name in ("narma20", "narma30"):
        spec = next(s for s in ALL_SPECS if s.name == name)
        ds = generate(replace(spec, length=800))
        assert np.isfinite(ds.series).all()
        assert float(np.std(ds.series)) > 1e-3


def test_make_sweep_specs_is_large_deterministic_and_generates_finite_series():
    specs = make_sweep_specs(n_per_family=10, seed=7)

    assert len(specs) >= 200
    assert [s.name for s in specs] == [s.name for s in make_sweep_specs(n_per_family=10, seed=7)]
    assert len({s.name for s in specs}) == len(specs)

    for spec in specs:
        assert spec.params
        assert "generator" in spec.params
        ds = generate(replace(spec, length=180))
        assert ds.series.shape == (180,), spec.name
        assert np.isfinite(ds.series).all(), spec.name
        if spec.task_type == "input_driven":
            assert ds.inputs is not None
            assert ds.inputs.shape == (180,)
            assert np.isfinite(ds.inputs).all()
        known_keys = set(ds.ground_truth).intersection(GROUND_TRUTH_FIELDS)
        assert spec.source == "synthetic"
        assert known_keys, spec.name


def test_default_sweep_specs_have_1000_synthetic_rows():
    specs = make_sweep_specs(n_per_family=20, seed=0)

    assert len(specs) == 1000
    assert {s.source for s in specs} == {"synthetic"}
    assert sum(s.params.get("generator") == "henon_heiles" for s in specs) == 40
    assert sum(s.family == "real_bridge" for s in specs) == 0


def test_santa_fe_real_bridge_specs_are_external_validation_only():
    specs = make_real_bridge_specs(n_windows=40, seed=0)

    assert len(specs) == 40
    assert {s.source for s in specs} == {"real"}
    assert {s.family for s in specs} == {"real_bridge"}
    assert {s.params.get("generator") for s in specs} == {"santa_fe_laser"}


def test_swept_map_specs_are_non_degenerate():
    specs = [
        s
        for s in make_sweep_specs(n_per_family=20, seed=0)
        if s.params.get("generator") in {"ikeda_map", "circle_map"}
    ]

    assert specs
    for spec in specs:
        ds = generate(replace(spec, length=800))
        assert float(np.std(ds.series)) > 1e-8, spec.name


def test_tiny_sweep_catalog_write_path_has_qrc_advantage(tmp_path, monkeypatch):
    specs = make_sweep_specs(n_per_family=2, seed=0)[:4]
    monkeypatch.setattr(run_study, "linear_baseline", lambda ds: 0.8)
    monkeypatch.setattr(run_study, "gbm_baseline", lambda ds, seed=0: 0.7)
    monkeypatch.setattr(run_study, "esn_matched_baseline", lambda ds, qrc_cfg=None, seed=0, esn_grid=None: 0.6)
    monkeypatch.setattr(run_study, "qrc_nrmse", lambda ds, cfg, seed=0: 0.5)
    monkeypatch.setattr(run_study, "_has_pyarrow", lambda: False)

    df, path, _cfg, seed_count = run_study.build_catalog(
        specs,
        out_dir=tmp_path,
        fast=True,
        seeds=1,
        output_stem="sweep_catalog",
        comparison_protocol="legacy_v1",
    )

    assert seed_count == 1
    assert path.name == "sweep_catalog.csv"
    assert path.exists()
    assert "qrc_advantage" in df.columns
    assert np.allclose(df["qrc_advantage"], 0.1)
    assert "qrc_advantage" in pd.read_csv(path).columns


def test_standard_v2_catalog_uses_fixed_dissipative_qrc(tmp_path, monkeypatch):
    specs = make_sweep_specs(n_per_family=1, seed=0)[:2]
    monkeypatch.setattr(run_study, "linear_baseline", lambda ds: 0.8)
    monkeypatch.setattr(run_study, "gbm_baseline", lambda ds, seed=0: 0.7)
    monkeypatch.setattr(run_study, "esn_sparse_baseline", lambda ds, qrc_cfg=None, seed=0, esn_grid=None: 0.6)
    monkeypatch.setattr(run_study, "qrc_nrmse_standard", lambda ds, cfg, seed=0: 0.5)
    monkeypatch.setattr(run_study, "_has_pyarrow", lambda: False)

    _df, _path, cfg, _seed_count = run_study.build_catalog(
        specs,
        out_dir=tmp_path,
        fast=True,
        seeds=1,
        output_stem="sweep_catalog",
        comparison_protocol="standard_v2",
    )

    assert cfg.reupload is False
    assert cfg.amplitude_damping == 0.02
    assert cfg.dephasing == 0.01
    assert cfg.dissipation_method == "trajectory"


def test_standard_v3_catalog_freezes_qrc_and_esn_reservoir_hyperparameters(tmp_path, monkeypatch):
    specs = make_sweep_specs(n_per_family=1, seed=0)[:2]
    seen_grids = []
    monkeypatch.setattr(run_study, "linear_baseline", lambda ds: 0.8)
    monkeypatch.setattr(run_study, "gbm_baseline", lambda ds, seed=0: 0.7)

    def fake_esn(ds, qrc_cfg=None, seed=0, esn_grid=None):
        seen_grids.append(esn_grid)
        return 0.6

    monkeypatch.setattr(run_study, "esn_sparse_baseline", fake_esn)
    monkeypatch.setattr(run_study, "qrc_nrmse_standard", lambda ds, cfg, seed=0: 0.5)
    monkeypatch.setattr(run_study, "_has_pyarrow", lambda: False)

    _df, _path, cfg, _seed_count = run_study.build_catalog(
        specs,
        out_dir=tmp_path,
        fast=True,
        seeds=1,
        output_stem="sweep_catalog",
        comparison_protocol="standard_v3",
    )

    assert cfg.reupload is False
    assert cfg.amplitude_damping == 0.02
    assert cfg.dephasing == 0.01
    assert seen_grids
    assert all(grid == {"rho": (0.9,), "leak": (0.3,), "input_scale": (1.0,)} for grid in seen_grids)

    manifest = json.loads((tmp_path / "sweep_catalog_manifest.json").read_text())
    assert manifest["comparison_protocol"] == "standard_v3"
    assert manifest["primary_esn"] == "frozen_sparse_random_leaky_esn"
    assert manifest["esn"]["hyperparameter_selection"] == "none_reservoir_hyperparameters_frozen_globally"
    assert manifest["esn"]["grid"] == {"rho": [0.9], "leak": [0.3], "input_scale": [1.0]}


def test_santa_fe_loader_accepts_canonical_local_parts(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savetxt(data_dir / "SantaFeA.dat", np.arange(1000, dtype=float))
    np.savetxt(data_dir / "SantaFeA2.dat", np.arange(1000, 5000, dtype=float))
    monkeypatch.chdir(tmp_path)

    spec = DatasetSpec("santa_fe_laser", "real_bridge", "real", "forecast", {"generator": "santa_fe_laser", "dataset": "A"}, seed=120)
    ds = generate(replace(spec, length=4000))

    assert not ds.ground_truth.get("_unavailable")
    assert ds.series.shape == (4000,)
    assert ds.series[0] == 0.0
    assert ds.series[-1] == 3999.0


def test_santa_fe_loader_uses_window_start(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savetxt(data_dir / "SantaFeA.dat", np.arange(1000, dtype=float))
    np.savetxt(data_dir / "SantaFeA2.dat", np.arange(1000, 5000, dtype=float))
    monkeypatch.chdir(tmp_path)

    spec = DatasetSpec("santa_fe_laser", "real_bridge", "real", "forecast", {"generator": "santa_fe_laser", "dataset": "A"}, seed=120)
    ds = generate(replace(spec, length=100, params={**spec.params, "window_start": 2500}))

    assert ds.series[0] == 2500.0
    assert ds.series[-1] == 2599.0
