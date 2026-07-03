# Paper Build

The LaTeX entry point is `main.tex`.

Build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf mapping_quantum_reservoir_advantage.pdf
```

The generated `build/` directory is ignored by git. The stable compiled PDF
`mapping_quantum_reservoir_advantage.pdf` is tracked for review convenience.

## Figure Inventory

Paper-facing figures are stored in `gfx/`.

- `regime-map.pdf` - main v5 regime map.
- `family-effects.pdf` - family-level QRC-favorable rates and effects.
- `meta-model-validation.pdf` - meta-model validation summary.
- `dataset-property-atlas-benchmarks.pdf` - 50k synthetic property atlas with Lorenz, Mackey-Glass, and NARMA callouts.
- `family-qrc-favorable-regimes.pdf` - clean family ranking of QRC-favorable validation configurations.
- `drifting-trend-example.pdf` - example time series from a QRC-favorable drifting-trend family.
