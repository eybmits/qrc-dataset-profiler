# Paper Build

The LaTeX entry point is `main.tex`.

Build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf mapping_quantum_reservoir_advantage.pdf
```

The generated `build/` directory is ignored by git. The stable compiled PDF
`mapping_quantum_reservoir_advantage.pdf` is tracked for review convenience.
