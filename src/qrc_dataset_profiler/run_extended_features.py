"""Build deterministic Tier-B feature tables for protocol datasets."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from qrc_dataset_profiler.generators import ALL_SPECS, generate, make_sweep_specs
from qrc_dataset_profiler.properties import compute_backstop
from qrc_dataset_profiler.run_study import _study_spec
from qrc_dataset_profiler.spec import DatasetSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write deterministic Tier-B feature tables.")
    parser.add_argument("--sweep", action="store_true", help="Use the parameterized sweep instead of ALL_SPECS.")
    parser.add_argument("--fast", action="store_true", help="Match run_study --fast lengths for sweep artifacts.")
    parser.add_argument("--smoke", action="store_true", help="Match run_study --smoke lengths for first-catalog artifacts.")
    parser.add_argument("--sweep-n-per-family", type=int, default=20, help="Variants per swept generator family.")
    parser.add_argument("--sweep-seed", type=int, default=0, help="Deterministic sweep seed.")
    parser.add_argument("--out", default="results_features", help="Output directory.")
    args = parser.parse_args(argv)

    specs = make_sweep_specs(args.sweep_n_per_family, seed=args.sweep_seed) if args.sweep else ALL_SPECS
    out_dir = Path(args.out)
    stem = "extended_features_sweep" if args.sweep else "extended_features_full"
    df = build_extended_feature_table(specs, smoke=args.smoke, fast=args.fast)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    manifest = {
        "feature_table_version": "tier-b-v1",
        "n_rows": int(len(df)),
        "n_features": int(len([c for c in df.columns if c.startswith("ext_") or c.startswith("catch22_") or c.startswith("ts_")])),
        "sweep": bool(args.sweep),
        "fast": bool(args.fast),
        "smoke": bool(args.smoke),
        "sweep_n_per_family": int(args.sweep_n_per_family),
        "sweep_seed": int(args.sweep_seed),
        "output": path.name,
        "claim_boundary": "Tier-B features are exploratory robustness descriptors, not replacements for the frozen schema-v1 core axes.",
    }
    (out_dir / "extended_features_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={path}")
    print(f"n_rows={len(df)}")
    print(f"n_features={manifest['n_features']}")
    return 0


def build_extended_feature_table(
    specs: list[DatasetSpec],
    *,
    smoke: bool = False,
    fast: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for base_spec in specs:
        spec = _study_spec(base_spec, smoke=smoke, fast=fast)
        ds = generate(spec)
        if ds.ground_truth.get("_unavailable"):
            warnings.warn(f"skipping unavailable dataset {spec.name}", RuntimeWarning)
            continue
        source = ds.inputs if spec.task_type == "input_driven" and ds.inputs is not None else ds.series
        features = compute_backstop(source)
        rows.append(
            {
                "dataset_id": f"{spec.name}:{spec.seed}:{spec.length}",
                "name": spec.name,
                "family": spec.family,
                "source": spec.source,
                "task_type": spec.task_type,
                "seed": int(spec.seed),
                "length": int(spec.length),
                **features,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
