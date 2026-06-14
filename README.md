# RNAmelt — RNA Melting-Curve Analysis

Two-state thermodynamic analysis (ΔH, ΔS, ΔG, Tm) of UV-absorbance or
fluorescence melting curves. The same Python pipeline runs in three
places:

- **Browser** — `index.html` loads Python via Pyodide (WebAssembly). Data
  never leaves the user's machine; no server, no install.
- **CLI** — `rnamelt FILE.csv [...]` for batch / scripted use.
- **Python API** — `from rnamelt import MeltAnalysis` for notebook /
  pipeline use. Returns typed result objects with optional `matplotlib`
  plotting.

## Project structure

```
RNAmelt/
├── pyproject.toml          # Packaging metadata (pip-installable)
├── index.html              # Single-page UI: controls, charts (Chart.js), Pyodide bridge
├── rnamelt/
│   ├── __init__.py         # Public API re-exports
│   ├── __main__.py         # CLI entry point
│   ├── api.py              # MeltAnalysis class + analyze_* functional wrappers
│   ├── results.py          # Typed result dataclasses (SingleResult, MultiResult, …)
│   ├── plots.py            # matplotlib helpers — optional, lazy-imported
│   ├── constants.py        # R, T0
│   ├── cleaning.py         # CSV parsing, column normalisation
│   ├── functions.py        # Pure helpers (sigmoidal model, fraction unfolded, …)
│   ├── methods.py          # Tm extraction, van't Hoff, full curve fit, multi fit, conc. series
│   ├── analysis_melting.py # Pipeline orchestrator — single entry point `run(df, params)`
│   └── utils.py            # safe_json, unit helpers
└── tests/
    ├── test_methods.py     # Method-level unit tests
    ├── test_analysis.py    # End-to-end concentration-series tests
    ├── test_api.py         # Functional façade round-trips
    └── test_meltanalysis.py # MeltAnalysis class + result dataclasses
```

Module layering is strict: `functions` → `methods` → `analysis_melting` →
`api` → `__main__` / browser. `functions` stays stateless. `plots` is
lazy-imported so the browser bundle never pulls matplotlib.

## Installation

```bash
# from a clone of this repo
pip install .

# with the optional matplotlib plotting helpers
pip install '.[plots]'

# editable install for development
pip install -e '.[plots]'
```

This exposes the `rnamelt` console script and lets you `import rnamelt`
from any working directory.

## CSV format

```
temperature, sample_1, sample_2, ...
20.0,        0.412,    0.388, ...
20.5,        0.415,    0.391, ...
```

- Column 0: temperature in °C (any header name).
- Columns 1+: absorbance or fluorescence signal (any header name).
- Rows that are entirely NaN, or where temperature is missing, are dropped.

## Running the browser app

Must be served over HTTP — `file://` won't load Pyodide.

```bash
python -m http.server 8000
# open http://localhost:8000
```

## CLI

Same orchestrator, no browser needed. Result is JSON on stdout; pass
`--csv-out` to also write a results CSV in the format the browser
download produces.

```bash
# default: van't Hoff + full fit on every signal column in the CSV
rnamelt melt.csv --csv-out batch.csv

# single column
rnamelt melt.csv --column sample_1 --struct-type heterodimer \
    --oligo 5.0 --T-low 20 --T-high 90 --csv-out single.csv

# shared-ΔH fit across all signal columns (defaults to all columns at --oligo
# if no --oligo-multi is given)
rnamelt melt.csv --column __multi__ --struct-type heterodimer \
    --oligo-multi sample_1=0.5 --oligo-multi sample_2=5.0 --oligo-multi sample_3=50

# concentration-series van't Hoff (1/Tm vs ln(C_T/f)) — requires --oligo-multi
rnamelt melt.csv --column __concentration__ --struct-type heterodimer \
    --oligo-multi sample_1=0.5 --oligo-multi sample_2=5.0 --oligo-multi sample_3=50 \
    --csv-out conc.csv
```

`rnamelt --help` lists every flag (equivalent: `python -m rnamelt --help`),
including the three solver-tuning groups (`scipy.optimize.least_squares`
overrides, van't Hoff linearisation, full-fit initial guesses).
Exit code is 0 on success, 1 if the analysis returns an error, 2 on bad input.

## Python API

The recommended surface is the `MeltAnalysis` class. It holds the
DataFrame plus every configuration knob (struct type, salt, transition
window, baseline offsets, solver / vH / fit-init overrides) and exposes
the three modes as methods that return typed result objects from
`rnamelt.results`.

```python
from rnamelt import MeltAnalysis, FitFailed

m = MeltAnalysis.from_csv(
    "melt.csv",
    struct_type="heterodimer",   # "heterodimer" | "homodimer" | "monomer"
    salt=150.0,                  # NaCl (mM) — metadata only
    T_low=None, T_high=None,     # transition window (°C); None = full range
    bl_lower_offset=10.0,        # folded-baseline span above T_low (°C)
    bl_upper_offset=10.0,        # unfolded-baseline span below T_high (°C)
)
m.signal_columns                 # ['sample_1', 'sample_2', ...]

# Single column — returns SingleResult; numpy arrays live on the object
r = m.single("sample_1", oligo=0.5)
r.Tm_raw, r.vh.dH, r.fit.Tm, r.fit.dG

try:
    r.fit.require()              # raises FitFailed if not r.fit.ok
except FitFailed as e:
    print(e)

# Shared-ΔH joint fit across columns
multi = m.multi({"sample_1": 0.5, "sample_2": 5.0, "sample_3": 50.0})
multi.dH, multi.dS
for c in multi.columns:
    print(c.name, c.Tm_fit, c.oligoC)

# Concentration-series van't Hoff — three regressions (raw / vH / full fit)
conc = m.concentration({"sample_1": 0.5, "sample_2": 5.0, "sample_3": 50.0})
for s in (conc.series_raw, conc.series_vh, conc.series_fit):
    if s.ok:
        print(s.dH, s.dG_37, s.r_squared)

# Every column at once
batch = m.single_all(oligo=0.5)  # dict[str, SingleResult]

# Export forms (same shape the browser download produces)
r.to_dict(); r.to_dataframe(); r.to_csv("out.csv")
```

### From raw arrays — no CSV

`MeltAnalysis.from_arrays` bypasses the CSV layer. `signals` accepts a
1D array (one column), a mapping `{name: array}`, or a 2D array (one
column per signal, with optional `names=`):

```python
import numpy as np
from rnamelt import MeltAnalysis

T = np.linspace(20, 90, 71)

m = MeltAnalysis.from_arrays(T, signal_F4)                        # 1D
m = MeltAnalysis.from_arrays(T, {"F4": sigF4, "F5": sigF5})       # dict
m = MeltAnalysis.from_arrays(T, arr2d, names=["A", "B", "C"])     # 2D
```

### Plotting (optional — needs matplotlib)

Install with `pip install '.[plots]'`. Each result class has a `.plot()`
method that lazy-imports `rnamelt.plots`, so matplotlib never enters the
Pyodide bundle.

```python
fig, axes = m.single("F4").plot(figsize=(15, 5))      # raw / vH / full fit
fig, ax   = m.multi(oligo_multi).plot()
fig, axes = m.concentration(oligo_multi).plot()

# convenience shortcut on the analyzer
fig, axes, result = m.plot("F4")
```

### Functional helpers (back-compat)

Thin wrappers kept for backwards compatibility — same orchestrator, but
they take a cleaned `pandas.DataFrame` and return the legacy dict
directly:

```python
import pandas as pd
from rnamelt import (
    analyze_single, analyze_multi, analyze_concentration, analyze_csv,
)
from rnamelt.cleaning import clean

df = clean(pd.read_csv("melt.csv"))
analyze_single(df, "sample_1", struct_type="heterodimer", oligo=0.5)
analyze_multi(df, {"sample_1": 0.5, "sample_2": 5.0})
analyze_concentration(df, {"sample_1": 0.5, "sample_2": 5.0})

# read_csv + clean + dispatch in one call
analyze_csv("melt.csv", mode="single", column="sample_1", oligo=0.5)
```

## Analysis modes

All three modes go through `rnamelt.analysis_melting.run(df, params)`;
the `column` param selects the mode.

| `column` value      | Mode                       | What it does |
|---------------------|----------------------------|--------------|
| any signal-column name | **Single column**       | Tm by baseline intersection, van't Hoff linearisation, full two-state nonlinear fit. Returns ΔH/ΔS/ΔG/Tm from each method. |
| `__multi__`         | **Shared-ΔH multi fit**    | Joint fit of every column with a common ΔH, ΔS but independent baselines. Each column carries its own concentration. |
| `__concentration__` | **Concentration-series van't Hoff** | Extracts three Tm values per column (raw / van't Hoff / full-fit) and runs `1/Tm = (R/ΔH)·ln(C_T/f) + ΔS/ΔH` linear regression for each. `f = 1` (homodimer) or `f = 4` (heterodimer). |

## Two-state model

Observed signal is a linear combination of folded and unfolded baselines,
weighted by the fraction unfolded θ(T):

```
A(T) = (m_F·T + b_F)·(1 − θ) + (m_U·T + b_U)·θ
K(T) = exp(−ΔG / RT),    ΔG = ΔH − T·ΔS
θ    = K / (1 + K)                       (monomer)
θ    = (√(1 + 8·c0·K) − 1) / (4·c0·K)    (dimer)
```

Units throughout: ΔH in kcal/mol, ΔS in kcal/(mol·K), T in K
(`T_K = T_C − T0`, with `T0 = −273.15`).

## Tests

```bash
pytest tests/                       # or
python -m unittest discover tests
```

Tests run outside the browser against the same `rnamelt/` package the
browser loads.

## Deployment

The browser app is a static folder — drop it on any HTTP host
(GitHub Pages, Netlify, Cloudflare Pages, plain nginx). No build step.

The Python package builds with `python -m build` (produces sdist + wheel
in `dist/`) and uploads to PyPI with `python -m twine upload dist/*`.

## License

MIT — see [`LICENSE`](LICENSE).
