# RNAmelt — RNA Melting-Curve Analysis

Two-state thermodynamic analysis (ΔH, ΔS, ΔG, Tm) of UV-absorbance or
fluorescence melting curves. The same Python pipeline runs in two places:

- **Browser** — `index.html` loads Python via Pyodide (WebAssembly). Data
  never leaves the user's machine; no server, no install.
- **CLI** — `rnamelt FILE.csv [...]` for batch / scripted use.
- **Python API** — `from rnamelt import analyze_single, analyze_multi,
  analyze_concentration, analyze_csv` for use in notebooks or larger
  pipelines.

## Project structure

```
RNAmelt/
├── pyproject.toml          # Packaging metadata (pip-installable)
├── index.html              # Single-page UI: controls, charts (Chart.js), Pyodide bridge
├── rnamelt/
│   ├── __init__.py         # Public API re-exports
│   ├── __main__.py         # CLI entry point
│   ├── api.py              # Python API: analyze_single / _multi / _concentration / _csv
│   ├── constants.py        # R, T0
│   ├── cleaning.py         # CSV parsing, column normalisation
│   ├── functions.py        # Pure helpers (sigmoidal model, fraction unfolded, …)
│   ├── methods.py          # Tm extraction, van't Hoff, full curve fit, multi fit, conc. series
│   ├── analysis_melting.py # Pipeline orchestrator — single entry point `run(df, params)`
│   └── utils.py            # safe_json, unit helpers
└── tests/
    ├── test_methods.py     # vant_hoff_concentration unit tests
    └── test_analysis.py    # End-to-end concentration-series tests
```

Module layering is strict: `functions` → `methods` → `analysis_melting` →
`__main__` / browser. `functions` stays stateless.

## Installation

```bash
# from a clone of this repo
pip install .

# or editable install for development
pip install -e .

# with test deps
pip install -e ".[test]"
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

`rnamelt --help` lists every flag (equivalent: `python -m rnamelt --help`).
Exit code is 0 on success, 1 if the analysis returns an error, 2 on bad input.

## Python API

For notebook / pipeline use, three mode-specific functions wrap the same
orchestrator the CLI uses. They take a cleaned `pandas.DataFrame` and
return a plain dict.

```python
import pandas as pd
from rnamelt import (
    analyze_single, analyze_multi, analyze_concentration, analyze_csv,
)
from rnamelt.cleaning import clean

df = clean(pd.read_csv("melt.csv"))

# single column
r = analyze_single(df, "sample_1", struct_type="heterodimer", oligo=0.5,
                   T_low=20, T_high=90)
print(r["TmRaw"], r["fit_result"]["dH"])

# shared-ΔH multi fit
r = analyze_multi(df, {"sample_1": 0.5, "sample_2": 5.0, "sample_3": 50.0},
                  struct_type="heterodimer")

# concentration-series van't Hoff (returns three regressions: raw / vH / fit)
r = analyze_concentration(df,
                          {"sample_1": 0.5, "sample_2": 5.0, "sample_3": 50.0},
                          struct_type="heterodimer")
for key in ("raw", "vh", "fit"):
    s = r["series"][key]
    print(key, s["dH"], s["r_squared"])
```

`analyze_csv(path, mode=...)` is a one-liner that wraps `pd.read_csv` +
`clean` + dispatch:

```python
analyze_csv("melt.csv", mode="single", column="sample_1", oligo=0.5)
analyze_csv("melt.csv", mode="concentration",
            oligo_multi={"sample_1": 0.5, "sample_2": 5.0})
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
python -m unittest discover tests
```

Tests run outside the browser against the same `rnamelt/` package the
browser loads.

## Deployment

The browser app is a static folder — drop it on any HTTP host
(GitHub Pages, Netlify, Cloudflare Pages, plain nginx). No build step.
