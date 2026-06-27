import numpy as np

from qrc_dataset_profiler.generators import ALL_SPECS, generate


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
