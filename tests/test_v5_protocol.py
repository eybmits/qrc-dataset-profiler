import json

import numpy as np
import pandas as pd

import qrc_dataset_profiler.v5_protocol as v5
from qrc_dataset_profiler.reservoir import StandardSpinV1
from qrc_dataset_profiler.spec import Dataset, DatasetSpec


def test_v5_calibration_writes_three_qrc_variants_and_esn(tmp_path, monkeypatch):
    specs = [
        DatasetSpec("a", "linear_stochastic", "synthetic", "forecast", seed=1, length=80),
        DatasetSpec("b", "long_range", "synthetic", "forecast", seed=2, length=80),
    ]
    datasets = [Dataset(spec, np.sin(np.linspace(0, 4, spec.length))) for spec in specs]
    rows = [
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
        for spec in specs
    ]

    monkeypatch.setattr(v5, "make_sweep_specs_v4", lambda n_per_template, seed: specs)
    monkeypatch.setattr(v5, "select_calibration_specs", lambda specs, rows_per_family, seed: specs)
    monkeypatch.setattr(v5, "_materialize_calibration_datasets", lambda specs, fast: (datasets, rows))
    monkeypatch.setattr(
        v5,
        "_score_qrc_candidate",
        lambda datasets, cfg, seeds: {
            "mean_val_nrmse": float(cfg.J + 0.01 * cfg.reupload + cfg.amplitude_damping),
            "median_val_nrmse": float(cfg.J),
            "mean_test_nrmse": float(cfg.J),
            "median_test_nrmse": float(cfg.J),
        },
    )
    monkeypatch.setattr(
        v5,
        "_score_esn_candidate",
        lambda datasets, qrc_cfg, cand, seeds: {
            "mean_val_nrmse": float(cand["rho"]),
            "median_val_nrmse": float(cand["rho"]),
            "mean_test_nrmse": float(cand["rho"]),
            "median_test_nrmse": float(cand["rho"]),
        },
    )

    manifest = v5.run_v5_calibration(out_dir=tmp_path, small_grid=True)

    assert manifest["comparison_protocol"] == "standard_v5_multi_qrc"
    assert set(manifest["qrc_variants"]) == {"qrc_m", "qrc_e", "qrc_d"}
    assert manifest["esn"]["reservoir_size"] == manifest["qrc_variants"]["qrc_m"]["feature_dim"]
    assert (tmp_path / "frozen_v5_config.json").exists()
    assert (tmp_path / "qrc_m_calibration_scores.csv").exists()
    frozen = json.loads((tmp_path / "frozen_v5_config.json").read_text())
    assert frozen["qrc_variants"]["qrc_m"]["coupling_mode"] == "disordered"


def test_write_v5_evaluated_selection_merges_variant_targets(tmp_path, monkeypatch):
    selection = pd.DataFrame(
        {
            "dataset_id": ["a:1:80", "b:2:80"],
            "name": ["a", "b"],
            "family": ["linear_stochastic", "long_range"],
            "source": ["synthetic", "synthetic"],
            "task_type": ["forecast", "forecast"],
            "params": ["{}", "{}"],
            "seed": [1, 2],
            "length": [80, 80],
            "horizon": [1, 1],
            "evaluation_split": ["discovery", "discovery"],
        }
    )
    selection_path = tmp_path / "selection.csv"
    selection.to_csv(selection_path, index=False)
    qrc = StandardSpinV1(n_qubits=2, topology="complete", virtual_nodes=1, depth=1, reupload=False, coupling_mode="disordered")
    frozen = {
        "comparison_protocol": "standard_v5_multi_qrc",
        "esn": {"reservoir_size": qrc.feature_dim, "rho": 0.9, "leak": 0.6, "input_scale": 1.0},
        "qrc_variants": {
            variant: {
                "n_qubits": qrc.n_qubits,
                "J": qrc.J,
                "h": qrc.h,
                "dt": qrc.dt,
                "depth": qrc.depth,
                "topology": qrc.topology,
                "virtual_nodes": qrc.virtual_nodes,
                "reupload": variant == "qrc_e",
                "amplitude_damping": 0.02 if variant == "qrc_d" else 0.0,
                "dephasing": 0.01 if variant == "qrc_d" else 0.0,
                "dissipation_method": qrc.dissipation_method,
                "coupling_mode": qrc.coupling_mode,
                "coupling_seed": qrc.coupling_seed,
                "feature_dim": qrc.feature_dim,
            }
            for variant in v5.V5_VARIANTS
        },
    }
    cfg_path = tmp_path / "frozen_v5_config.json"
    cfg_path.write_text(json.dumps(frozen))

    def fake_eval(specs, **_kwargs):
        return pd.DataFrame(
            {
                "dataset_id": [f"{spec.name}:{spec.seed}:{spec.length}" for spec in specs],
                "nrmse_esn_v5": [0.8, 0.9],
                "nmae_esn_v5": [0.7, 0.8],
                "nrmse_qrc_m": [0.75, 1.0],
                "advantage_qrc_m_vs_esn": [0.05, -0.1],
                "nrmse_qrc_e": [0.7, 0.85],
                "advantage_qrc_e_vs_esn": [0.1, 0.05],
                "nrmse_qrc_d": [0.9, 0.7],
                "advantage_qrc_d_vs_esn": [-0.1, 0.2],
                "best_qrc_variant": ["qrc_e", "qrc_d"],
                "best_qrc_advantage_vs_esn": [0.1, 0.2],
                "qrc_any_useful": [True, True],
            }
        )

    monkeypatch.setattr(v5, "_evaluate_v5_specs", fake_eval)

    df, path = v5.write_v5_evaluated_selection(
        selection_path=selection_path,
        calibration_config=cfg_path,
        out_dir=tmp_path / "evaluated",
        split="discovery",
    )

    assert path.name == "frontier_discovery_evaluated_v5_multi_qrc.csv"
    assert len(df) == 2
    assert {"advantage_qrc_m_vs_esn", "advantage_qrc_e_vs_esn", "advantage_qrc_d_vs_esn", "best_qrc_variant"}.issubset(df.columns)
    manifest = json.loads((tmp_path / "evaluated" / "frontier_discovery_v5_manifest.json").read_text())
    assert manifest["complete"] is True
