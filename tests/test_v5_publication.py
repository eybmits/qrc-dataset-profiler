import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from qrc_dataset_profiler.spec import FRONTIER_TIER_A_FIELDS


def _fake_v5_table(n=120, seed=0, split="discovery"):
    rng = np.random.default_rng(seed)
    families = ["colored_noise", "chaotic_flow", "unit_root_trend", "input_driven_memory"]
    variants = ["qrc_m", "qrc_e", "qrc_d"]
    rows = []
    for i in range(n):
        signal = rng.normal()
        family = families[i % len(families)]
        base = 0.08 * signal + (0.08 if family == "colored_noise" else -0.03 if family == "input_driven_memory" else 0.0)
        adv_m = base + rng.normal(scale=0.035)
        adv_e = base * 0.75 + rng.normal(scale=0.04)
        adv_d = base * 0.55 + (0.04 if family == "colored_noise" else 0.0) + rng.normal(scale=0.04)
        advs = {"qrc_m": adv_m, "qrc_e": adv_e, "qrc_d": adv_d}
        best_variant = max(advs, key=advs.get)
        esn = 0.85 + 0.04 * rng.normal()
        row = {
            "dataset_id": f"{split}_{i}:s{i}:{800+i}",
            "name": f"{split}_{i}",
            "family": family,
            "source": "synthetic",
            "task_type": "forecast",
            "params": "{}",
            "seed": i,
            "length": 800 + i,
            "horizon": 1,
            "base_generator": f"gen_{i % 12}",
            "nrmse_linear": esn + 0.14,
            "nrmse_gbm": esn + 0.05,
            "nrmse_nvar": esn + 0.03,
            "nmae_nvar": esn + 0.02,
            "nrmse_esn_v5": esn,
            "nmae_esn_v5": esn * 0.8,
            "best_qrc_variant": best_variant,
            "best_qrc_advantage_vs_esn": advs[best_variant],
            "qrc_any_win": advs[best_variant] > 0,
            "qrc_any_useful": advs[best_variant] >= 0.05,
            "label_seed_count": 1,
        }
        for variant in variants:
            adv = advs[variant]
            row[f"nrmse_{variant}"] = esn - adv
            row[f"nmae_{variant}"] = esn * 0.8 - adv * 0.8
            row[f"advantage_{variant}_vs_esn"] = adv
            row[f"nmae_advantage_{variant}_vs_esn"] = adv * 0.8
            row[f"{variant}_useful"] = adv >= 0.05
        for j, feature in enumerate(FRONTIER_TIER_A_FIELDS):
            row[feature] = float(signal + 0.04 * j + rng.normal(scale=0.45))
        rows.append(row)
    return pd.DataFrame(rows)


def test_v5_publication_cli_writes_tables_figures_and_html(tmp_path):
    discovery = _fake_v5_table(n=120, seed=1, split="discovery")
    validation = _fake_v5_table(n=120, seed=2, split="validation")
    discovery_path = tmp_path / "discovery.csv"
    validation_path = tmp_path / "validation.csv"
    discovery.to_csv(discovery_path, index=False)
    validation.to_csv(validation_path, index=False)
    out_dir = tmp_path / "v5_publication"

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qrc_dataset_profiler.run_v5_publication",
            "--discovery-table",
            str(discovery_path),
            "--validation-table",
            str(validation_path),
            "--out",
            str(out_dir),
            "--family-bootstraps",
            "20",
            "--importance-bootstraps",
            "3",
            "--formats",
            "png",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    expected_tables = {
        "v5_split_summary.csv",
        "v5_family_summary.csv",
        "v5_variant_summary.csv",
        "v5_feature_set_robustness.csv",
        "v5_validation_predictions.csv",
        "v5_claims_and_guardrails.csv",
    }
    assert expected_tables.issubset({p.name for p in (out_dir / "tables").iterdir()})
    expected_figures = {
        "v5_01_outcome_overview.png",
        "v5_02_regime_map.png",
        "v5_03_family_effects.png",
        "v5_04_feature_regressions.png",
        "v5_05_meta_model_validation.png",
        "v5_06_robustness.png",
        "v5_07_family_feature_matrix.png",
    }
    assert expected_figures.issubset({p.name for p in (out_dir / "figures").iterdir()})
    assert (out_dir / "index.html").exists()
    manifest = json.loads((out_dir / "v5_publication_manifest.json").read_text())
    assert manifest["n_discovery"] == 120
    assert manifest["n_validation"] == 120
    assert len(manifest["figures"]) == 7
