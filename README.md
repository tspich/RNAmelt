# rnamelt

[![PyPI](https://img.shields.io/pypi/v/rnamelt.svg)](https://pypi.org/project/rnamelt/)
[![Python versions](https://img.shields.io/pypi/pyversions/rnamelt.svg)](https://pypi.org/project/rnamelt/)
[![License: MIT](https://img.shields.io/pypi/l/rnamelt.svg)](https://github.com/tspich/RNAmelt/blob/main/LICENSE)

Two-state thermodynamic analysis (ΔH, ΔS, ΔG, Tm) of UV-absorbance and
fluorescence RNA / DNA melting curves. One small, dependency-light Python
package that works as a library, a CLI, and a browser app.

- **Library** — `from rnamelt import MeltAnalysis`, typed result objects,
  optional `matplotlib` plotting.
- **CLI** — `rnamelt FILE.csv [...]` for batch / scripted runs.
- **Browser** — the same Python pipeline runs client-side via Pyodide.
  Live demo: <https://tspich.github.io/RNAmelt/>.

## Install

```bash
pip install rnamelt                 # base — numpy, pandas, scipy
pip install 'rnamelt[plots]'        # adds matplotlib for .plot() helpers
```

Requires Python ≥ 3.10.

## Quickstart

```python
from rnamelt import MeltAnalysis

m = MeltAnalysis.from_csv("melt.csv", struct_type="heterodimer")
r = m.single("sample_1", oligo=0.5)     # oligo in µM
print(r.Tm_fit, r.fit.dH, r.fit.dG)
```

```bash
rnamelt melt.csv --column sample_1 --oligo 0.5 --struct-type heterodimer
```

## CSV input format

```
temperature, sample_1, sample_2, ...
20.0,        0.412,    0.388, ...
20.5,        0.415,    0.391, ...
```

- Column 0: temperature in °C (any header name).
- Columns 1+: absorbance or fluorescence (any header name).
- Rows entirely NaN, or with a missing temperature, are dropped.

## CLI

Same orchestrator as the library. JSON on stdout; pass `--csv-out` for a
results CSV in the same format the browser produces.

```bash
# van't Hoff + full fit on every signal column
rnamelt melt.csv --csv-out batch.csv

# single column
rnamelt melt.csv --column sample_1 --struct-type heterodimer \
    --oligo 5.0 --T-low 20 --T-high 90 --csv-out single.csv

# shared-ΔH joint fit across columns
rnamelt melt.csv --column __multi__ --struct-type heterodimer \
    --oligo-multi sample_1=0.5 --oligo-multi sample_2=5.0

# concentration-series van't Hoff (1/Tm vs ln(C_T/f))
rnamelt melt.csv --column __concentration__ --struct-type heterodimer \
    --oligo-multi sample_1=0.5 --oligo-multi sample_2=5.0 --oligo-multi sample_3=50 \
    --csv-out conc.csv
```

`rnamelt --help` lists every flag, including three solver-tuning groups
(`scipy.optimize.least_squares` overrides, van't Hoff linearisation,
full-fit initial guesses). Exit codes: `0` success, `1` analysis error,
`2` bad input.

## Python API

The recommended surface is `MeltAnalysis`. It holds the DataFrame and
every configuration knob (struct type, salt, transition window, baseline
offsets, solver / vH / fit-init overrides) and exposes the three modes as
methods returning typed result objects from `rnamelt.results`.

```python
from rnamelt import MeltAnalysis, FitFailed

m = MeltAnalysis.from_csv(
    "melt.csv",
    struct_type="heterodimer",   # "heterodimer" | "homodimer" | "monomer"
    salt=150.0,                  # NaCl (mM) — metadata only
    T_low=None, T_high=None,     # transition window (°C); None = full range
    bl_lower_offset=10.0,        # folded baseline span above T_low (°C)
    bl_upper_offset=10.0,        # unfolded baseline span below T_high (°C)
)
m.signal_columns                 # ['sample_1', 'sample_2', ...]
```

### Three modes

```python
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

`signals` accepts a 1D array, a `{name: array}` mapping, or a 2D array
(one column per signal, with optional `names=`):

```python
import numpy as np
from rnamelt import MeltAnalysis

T = np.linspace(20, 90, 71)

m = MeltAnalysis.from_arrays(T, signal_F4)                        # 1D
m = MeltAnalysis.from_arrays(T, {"F4": sigF4, "F5": sigF5})       # mapping
m = MeltAnalysis.from_arrays(T, arr2d, names=["A", "B", "C"])     # 2D
```

### Plotting (optional)

`pip install 'rnamelt[plots]'`. Each result class has a `.plot()` method
that lazy-imports `rnamelt.plots`, so matplotlib never enters the
browser/Pyodide bundle.

```python
fig, axes = m.single("F4").plot(figsize=(15, 5))      # raw / vH / full fit
fig, ax   = m.multi(oligo_multi).plot()
fig, axes = m.concentration(oligo_multi).plot()

# Convenience shortcut on the analyzer
fig, axes, result = m.plot("F4")
```

### Functional helpers (back-compat)

Thin wrappers around the same orchestrator that take a cleaned
`pandas.DataFrame` and return the legacy dict directly:

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

| `column` value         | Mode                                | What it does |
|------------------------|-------------------------------------|--------------|
| any signal-column name | **Single column**                   | Tm by baseline intersection, van't Hoff linearisation, and a full two-state nonlinear fit. Returns ΔH / ΔS / ΔG / Tm from each method. |
| `__multi__`            | **Shared-ΔH joint fit**             | Fit every column with common ΔH, ΔS but independent baselines. Each column carries its own concentration. |
| `__concentration__`    | **Concentration-series van't Hoff** | Three Tm values per column (raw / vH / full-fit) → `1/Tm = (R/ΔH)·ln(C_T/f) + ΔS/ΔH` regression for each. `f = 1` (homodimer) or `f = 4` (heterodimer). |

## Two-state model

Observed signal is a linear combination of folded and unfolded baselines,
weighted by the fraction unfolded θ(T):

```
A(T) = (m_F·T + b_F)·(1 − θ) + (m_U·T + b_U)·θ
K(T) = exp(−ΔG / RT),    ΔG = ΔH − T·ΔS
θ    = K / (1 + K)                       (monomer)
θ    = (√(1 + 8·c₀·K) − 1) / (4·c₀·K)    (dimer)
```

Units: ΔH in kcal/mol, ΔS in kcal/(mol·K), T in K
(`T_K = T_C − T0`, `T0 = −273.15`).

## Browser app

The same Python package runs client-side via [Pyodide](https://pyodide.org/);
no server, no upload, no install. Drop a CSV onto
<https://tspich.github.io/RNAmelt/> to try it. The source for the page is
`index.html` in the [repository](https://github.com/tspich/RNAmelt).

## Development

```bash
git clone https://github.com/tspich/RNAmelt
cd RNAmelt
pip install -e '.[plots]'

pytest tests/                 # or: python -m unittest discover tests
python -m build               # sdist + wheel into dist/
```

Module layering is strict: `functions` → `methods` → `analysis_melting` →
`api` → `__main__` / browser. `functions` stays stateless. `plots` is
lazy-imported so the browser bundle never pulls matplotlib.

## License

[MIT](LICENSE).

## Links

- Source: <https://github.com/tspich/RNAmelt>
- Issues: <https://github.com/tspich/RNAmelt/issues>
- Live demo: <https://tspich.github.io/RNAmelt/>
