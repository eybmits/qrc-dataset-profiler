from dataclasses import replace

import numpy as np
import pandas as pd

import qrc_dataset_profiler.run_study as run_study
from qrc_dataset_profiler.generators import ALL_SPECS, generate, make_sweep_specs
from qrc_dataset_profiler.spec import GROUND_TRUTH_FIELDS


def test_all_specs_generate_requested_shape_or_unavailable():
    assert len(ALL_SPECS) == 20
    for spec in ALL_SPECS:
        ds = generate(spec)
        if ds.ground_truth.get("_unavailable"):
            assert spec.name == "santa_fe_laser"
            assert ds.series.size == 0
            continue
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


def test_make_sweep_specs_is_large_deterministic_and_generates_finite_series():
    specs = make_sweep_specs(n_per_family=10, seed=7)

    assert len(specs) >= 120
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
        assert known_keys, spec.name


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
    )

    assert seed_count == 1
    assert path.name == "sweep_catalog.csv"
    assert path.exists()
    assert "qrc_advantage" in df.columns
    assert np.allclose(df["qrc_advantage"], 0.1)
    assert "qrc_advantage" in pd.read_csv(path).columns
