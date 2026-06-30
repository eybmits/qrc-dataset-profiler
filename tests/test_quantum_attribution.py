from dataclasses import replace

import numpy as np
import pandas as pd

import qrc_dataset_profiler.quantum_attribution as qa
from qrc_dataset_profiler.quantum_attribution import (
    run_quantum_attribution,
    spec_from_catalog_row,
    summarize_attribution,
)
from qrc_dataset_profiler.spec import DatasetSpec


def _catalog():
    return pd.DataFrame(
        {
            "name": ["logistic_a", "logistic_b", "ar"],
            "family": ["chaotic_map", "chaotic_map", "linear_stochastic"],
            "source": ["synthetic", "synthetic", "synthetic"],
            "task_type": ["forecast", "forecast", "forecast"],
            "params": [
                "{'generator': 'logistic', 'r': 3.9}",
                "{'generator': 'logistic', 'r': 3.8}",
                "{'generator': 'ar', 'phi': [0.6]}",
            ],
            "seed": [10, 11, 12],
            "length": [120, 120, 120],
            "n_channels": [1, 1, 1],
            "horizon": [1, 1, 1],
            "nrmse_esn_matched": [0.5, 0.7, 0.9],
        }
    )


def test_spec_from_catalog_row_preserves_protocol_fields():
    spec = spec_from_catalog_row(_catalog().iloc[0])

    assert isinstance(spec, DatasetSpec)
    assert spec.name == "logistic_a"
    assert spec.params["generator"] == "logistic"
    assert spec.seed == 10
    assert spec.length == 120
    assert spec.horizon == 1


def test_summarize_attribution_marks_negative_signal():
    paired = pd.DataFrame(
        {
            "family": ["a", "a", "b", "b"],
            "paired_delta_J0_minus_J1": [-0.2, -0.1, 0.1, 0.2],
            "advantage_J1_vs_esn": [0.0, 0.1, 0.2, 0.3],
            "advantage_J0_vs_esn": [0.2, 0.2, 0.1, 0.1],
        }
    )

    summary = summarize_attribution(paired, bootstrap_replicates=20, seed=0)

    by_family = {row.family: row.mechanism_signal for row in summary.itertuples(index=False)}
    assert by_family["a"] == "negative"
    assert by_family["b"] == "positive"


def test_run_quantum_attribution_writes_paired_outputs(tmp_path, monkeypatch):
    def fake_generate(spec):
        from qrc_dataset_profiler.generators import generate

        return generate(replace(spec, length=80))

    def fake_qrc_nrmse(ds, cfg, seed=0):
        base = 0.4 if cfg.J == 1.0 else 0.5
        return base + 0.001 * seed

    monkeypatch.setattr(qa, "generate", fake_generate)
    monkeypatch.setattr(qa, "qrc_nrmse", fake_qrc_nrmse)

    manifest = run_quantum_attribution(
        _catalog(),
        out_dir=tmp_path,
        families=("chaotic_map",),
        seeds=2,
        n_qubits=4,
        depth=2,
        virtual_nodes=2,
        bootstrap_replicates=10,
        seed=3,
    )

    paired = pd.read_csv(tmp_path / "paired_attribution.csv")
    summary = pd.read_csv(tmp_path / "family_attribution_bootstrap.csv")
    assert manifest["rows_written"] == 2
    assert manifest["reservoir"]["feature_dim_J1"] == manifest["reservoir"]["feature_dim_J0"]
    assert np.allclose(paired["paired_delta_J0_minus_J1"], 0.1)
    assert set(summary["family"]) == {"overall", "chaotic_map"}
    assert (tmp_path / "attribution_manifest.json").exists()
