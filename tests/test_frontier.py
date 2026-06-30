import numpy as np
import pandas as pd

import qrc_dataset_profiler.frontier as frontier
from qrc_dataset_profiler.frontier import (
    build_frontier_30_table,
    build_property_atlas,
    compute_support_scores,
    frontier_enrichment_score,
    select_evaluation_atlas,
    specs_from_selection,
    write_evaluated_selection,
)
from qrc_dataset_profiler.reservoir import StandardSpinV1
from qrc_dataset_profiler.generators import V4_PROCESS_FAMILIES, make_sweep_specs, make_sweep_specs_v4
from qrc_dataset_profiler.spec import FRONTIER_TIER_A_FIELDS


def _fake_property_atlas(n=120):
    rng = np.random.default_rng(0)
    rows = []
    families = ["chaotic_flow", "nonstationary", "linear_stochastic"]
    generators = ["lorenz63", "chirp", "ar"]
    for i in range(n):
        row = {
            "dataset_id": f"d{i}:{i}:800",
            "name": f"d{i}",
            "family": families[i % len(families)],
            "source": "synthetic",
            "task_type": "forecast",
            "params": {"generator": generators[i % len(generators)]},
            "seed": i,
            "length": 800,
            "base_generator": generators[i % len(generators)],
            "qrc_advantage": rng.normal(),
        }
        for j, feature in enumerate(FRONTIER_TIER_A_FIELDS):
            row[feature] = float(rng.normal(loc=(i % 7) / 10.0, scale=0.5 + j * 0.001))
        rows.append(row)
    return pd.DataFrame(rows)


def test_property_atlas_builds_30_features_without_targets():
    specs = make_sweep_specs(n_per_family=1, seed=3)[:3]
    df = build_property_atlas(specs, fast=True, max_rows=3)

    assert len(df) == 3
    assert set(df["source"]) == {"synthetic"}
    assert "base_generator" in df.columns
    for feature in FRONTIER_TIER_A_FIELDS:
        assert feature in df.columns
    assert df["predictability_gap_linear_gbm"].notna().any()


def test_v4_taxonomy_covers_16_families_and_100_template_units():
    specs = make_sweep_specs_v4(n_per_template=1, seed=5)

    assert len(specs) == 100
    assert set(V4_PROCESS_FAMILIES).issubset({spec.family for spec in specs})
    assert any(spec.params.get("perturbation_axes") for spec in specs)
    assert all(spec.source == "synthetic" for spec in specs)


def test_frontier_30_join_materializes_predictability_gap():
    catalog = pd.DataFrame(
        {
            "dataset_id": ["a", "b"],
            "name": ["a", "b"],
            "family": ["f", "f"],
            "source": ["synthetic", "synthetic"],
            "task_type": ["forecast", "forecast"],
            "params": ["{'generator': 'ar'}", "{'generator': 'ar'}"],
            "seed": [1, 2],
            "length": [800, 800],
            "pred_nrmse_linear": [1.2, 0.9],
            "pred_nrmse_gbm": [0.8, 1.0],
            "qrc_advantage": [0.1, -0.1],
        }
    )
    extended = pd.DataFrame(
        {
            "dataset_id": ["a", "b"],
            "ext_volatility_ac1": [0.2, 0.3],
            "ext_arch_lm5": [0.4, 0.5],
        }
    )

    df = build_frontier_30_table(catalog, extended)

    assert len(df) == 2
    assert np.allclose(df["predictability_gap_linear_gbm"], [0.4, -0.1])
    for feature in FRONTIER_TIER_A_FIELDS:
        assert feature in df.columns


def test_selection_is_deterministic_and_target_free():
    atlas = _fake_property_atlas(160)
    selected_a = select_evaluation_atlas(atlas, n_discovery=30, n_validation=30, seed=9)
    perturbed = atlas.copy()
    perturbed["qrc_advantage"] = np.arange(len(perturbed)) * 1000.0
    selected_b = select_evaluation_atlas(perturbed, n_discovery=30, n_validation=30, seed=9)

    ids_a = selected_a.loc[selected_a["evaluation_split"] != "unselected", "dataset_id"].tolist()
    ids_b = selected_b.loc[selected_b["evaluation_split"] != "unselected", "dataset_id"].tolist()
    assert ids_a == ids_b
    assert (selected_a["evaluation_split"] == "discovery").sum() == 30
    assert (selected_a["evaluation_split"] == "validation").sum() == 30
    assert set(selected_a["selection_role"].dropna()) >= {"", "broad_balanced", "frontier_enriched"}


def test_frontier_enrichment_score_ignores_target_columns():
    atlas = _fake_property_atlas(80)
    score_a = frontier_enrichment_score(atlas)
    atlas["qrc_advantage"] = np.linspace(-100, 100, len(atlas))
    score_b = frontier_enrichment_score(atlas)

    assert np.allclose(score_a, score_b)
    assert ((0.0 <= score_a) & (score_a <= 1.0)).all()


def test_support_scores_are_discovery_fitted_and_bounded():
    discovery = _fake_property_atlas(80).drop(columns=["qrc_advantage"])
    target = _fake_property_atlas(20).drop(columns=["qrc_advantage"])

    scores = compute_support_scores(discovery, target, k_values=(5, 10))

    assert len(scores) == len(target)
    assert {"support_score", "ood_flag", "family_entropy", "nearest_family_mixture"}.issubset(scores.columns)
    assert ((0.0 <= scores["support_score"]) & (scores["support_score"] <= 1.0)).all()


def test_specs_from_selection_reconstructs_selected_dataset_specs():
    atlas = _fake_property_atlas(40)
    selected = select_evaluation_atlas(atlas, n_discovery=5, n_validation=5, seed=2)

    specs = specs_from_selection(selected, split="discovery")

    assert len(specs) == 5
    assert {s.source for s in specs} == {"synthetic"}
    assert all("generator" in s.params for s in specs)
    assert all(s.length == 800 for s in specs)


def test_checkpointed_evaluation_writes_resumable_final_table(tmp_path, monkeypatch):
    atlas = _fake_property_atlas(24)
    selected = select_evaluation_atlas(atlas, n_discovery=6, n_validation=6, seed=4)
    selection_path = tmp_path / "selection.csv"
    selected.to_csv(selection_path, index=False)

    def fake_build_catalog(specs, *, out_dir, output_stem, **_kwargs):
        rows = []
        for spec in specs:
            rows.append(
                {
                    "dataset_id": f"{spec.name}:{spec.seed}:{spec.length}",
                    "nrmse_linear": 1.0,
                    "nrmse_esn_matched": 0.8,
                    "nrmse_qrc_spin": 0.7,
                    "nrmse_gbm": 0.6,
                    "qrc_advantage": 0.1,
                }
            )
        df = pd.DataFrame(rows)
        path = out_dir / f"{output_stem}.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df, path, StandardSpinV1(n_qubits=6), 1

    monkeypatch.setattr(frontier, "build_catalog", fake_build_catalog)

    df, path = write_evaluated_selection(
        selection_path=selection_path,
        out_dir=tmp_path / "evaluated",
        split="discovery",
        fast=True,
        seeds=1,
        checkpoint_every=2,
    )

    assert path.name == "frontier_discovery_evaluated_30_features.csv"
    assert len(df) == 6
    assert df["qrc_advantage"].notna().all()
    assert (tmp_path / "evaluated" / "frontier_discovery_evaluated_30_features_partial.csv").exists()
    assert len(list((tmp_path / "evaluated" / "checkpoints").glob("frontier_discovery_chunk_*.csv"))) == 3
