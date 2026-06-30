import json

import qrc_dataset_profiler.calibration as calibration
import qrc_dataset_profiler.run_study as run_study
from qrc_dataset_profiler.calibration import run_global_calibration
from qrc_dataset_profiler.generators import make_sweep_specs


def test_select_calibration_specs_is_balanced_and_held_out_seeded():
    specs = make_sweep_specs(n_per_family=2, seed=917)
    selected = calibration.select_calibration_specs(specs, rows_per_family=2, seed=917)
    counts = {}
    for spec in selected:
        counts[spec.family] = counts.get(spec.family, 0) + 1

    assert selected
    assert max(counts.values()) <= 2
    assert {spec.source for spec in selected} == {"synthetic"}
    assert all(spec.seed >= 917 * 100_000 for spec in selected)


def test_run_global_calibration_writes_frozen_config(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "_materialize_calibration_datasets", lambda specs, fast: ([], []))

    def fake_materialize(specs, fast):
        from qrc_dataset_profiler.generators import generate

        rows = []
        datasets = []
        for spec in specs[:3]:
            ds = generate(spec)
            datasets.append(ds)
            rows.append(
                {
                    "name": spec.name,
                    "family": spec.family,
                    "source": spec.source,
                    "task_type": spec.task_type,
                    "seed": spec.seed,
                    "length": spec.length,
                    "horizon": spec.horizon,
                    "params": repr(spec.params),
                }
            )
        return datasets, rows

    monkeypatch.setattr(calibration, "_materialize_calibration_datasets", fake_materialize)
    monkeypatch.setattr(calibration, "_score_qrc_candidate", lambda datasets, cfg, seeds: {"mean_val_nrmse": cfg.J, "median_val_nrmse": cfg.J, "mean_test_nrmse": cfg.J, "median_test_nrmse": cfg.J})
    monkeypatch.setattr(
        calibration,
        "_score_esn_candidate",
        lambda datasets, qrc_cfg, cand, seeds: {
            "mean_val_nrmse": cand["rho"],
            "median_val_nrmse": cand["rho"],
            "mean_test_nrmse": cand["rho"],
            "median_test_nrmse": cand["rho"],
        },
    )

    manifest = run_global_calibration(
        out_dir=tmp_path,
        sweep_seed=917,
        sweep_n_per_family=1,
        calibration_rows_per_family=1,
        fast=True,
        seeds=1,
        qrc_grid={"J": (1.2, 0.8), "h": (1.0,), "dt": (0.25,), "amplitude_damping": (0.02,), "dephasing": (0.01,)},
        esn_grid={"rho": (1.3, 0.7), "leak": (0.3,), "input_scale": (1.0,)},
    )

    frozen = json.loads((tmp_path / "frozen_config.json").read_text())
    assert manifest["comparison_protocol"] == "standard_v3"
    assert frozen["qrc"]["J"] == 0.8
    assert frozen["esn"]["rho"] == 0.7
    assert (tmp_path / "qrc_calibration_scores.csv").exists()
    assert (tmp_path / "esn_calibration_scores.csv").exists()


def test_build_catalog_loads_standard_v3_calibration_config(tmp_path, monkeypatch):
    frozen = {
        "comparison_protocol": "standard_v3",
        "qrc": {
            "class": "StandardSpinV1",
            "n_qubits": 4,
            "J": 1.2,
            "h": 1.0,
            "dt": 0.2,
            "depth": 3,
            "topology": "ring",
            "virtual_nodes": 3,
            "reupload": False,
            "amplitude_damping": 0.02,
            "dephasing": 0.01,
            "dissipation_method": "trajectory",
        },
        "esn": {"rho": 0.7, "leak": 0.6, "input_scale": 2.0},
    }
    cfg_path = tmp_path / "frozen_config.json"
    cfg_path.write_text(json.dumps(frozen))
    seen_grids = []
    specs = make_sweep_specs(n_per_family=1, seed=0)[:1]
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
        out_dir=tmp_path / "study",
        fast=True,
        seeds=1,
        comparison_protocol="standard_v3",
        calibration_config=cfg_path,
    )

    assert cfg.n_qubits == 4
    assert cfg.J == 1.2
    assert seen_grids == [{"rho": (0.7,), "leak": (0.6,), "input_scale": (2.0,)}]
    manifest = json.loads((tmp_path / "study" / "full_catalog_manifest.json").read_text())
    assert manifest["calibration_config"] == str(cfg_path)
    assert manifest["esn"]["hyperparameter_selection"] == "selected_once_on_held_out_calibration_set_then_frozen"
