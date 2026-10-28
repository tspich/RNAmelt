# rnamelt

[![PyPI](https://img.shields.io/pypi/v/rnamelt.svg)](https://pypi.org/project/rnamelt/)
[![Python versions](https://img.shields.io/pypi/pyversions/rnamelt.svg)](https://pypi.org/project/rnamelt/)
[![License: MIT](https://img.shields.io/pypi/l/rnamelt.svg)](https://github.com/tspich/RNAmelt/blob/main/LICENSE)

Two-state thermodynamic analysis (ΔH, ΔS, ΔG, Tm) of UV-absorbance and
fluorescence RNA / DNA melting curves. One small, dependency-light Python
package that works as a library and as a browser app.

- **Library** — `from rnamelt import SingleCurve, MultiCurves`, typed
  result objects, optional `matplotlib` plotting.
- **Browser** — the same Python pipeline runs client-side via Pyodide: no
  server, no upload. Live demo: <https://tspich.github.io/RNAmelt/>.

Full documentation (browser workflow, settings, standard errors, model):
the [help page](https://tspich.github.io/RNAmelt/help.html).

## Install

```bash
pip install rnamelt                 # base — numpy, pandas, scipy
pip install 'rnamelt[plots]'        # adds matplotlib for rnamelt.plots
```

Requires Python ≥ 3.10.

## Quickstart

```python
import pandas as pd
from rnamelt import SingleCurve, MultiCurves

df = pd.read_csv("melt.csv")         # column 0: temperature (°C); then one column per curve
T = df.iloc[:, 0].to_numpy(float)

c = SingleCurve(T, df["sample_1"].to_numpy(float), name="sample_1",
                oligoC=0.5)          # per-strand µM; (A, B) for unequal strands
r  = c.raw_tm()                      # Tm where the mean baseline crosses the data
vh = c.vant_hoff()                   # linear fit of ln K vs 1/T
f  = c.fit()                         # non-linear two-state fit
if f.ok:
    print(f.dH, f.dS, f.dG, f.Tm)    # kcal/mol, kcal/(mol·K), kcal/mol at 37 °C, °C
else:
    print(f.error)

m = MultiCurves([SingleCurve(T, df[col].to_numpy(float), name=col, oligoC=o)
                 for col, o in {"s1": 0.5, "s2": 5.0, "s3": 50.0}.items()])
gf  = m.global_fit()                 # one ΔH / ΔS shared by all curves
cdr = m.conc_dependency()            # 1/Tm vs ln(C_T/f)
```

Data and concentration are all a default run needs. Baseline windows are
detected automatically, or give all four:
`SingleCurve(..., fold_low_T=15, fold_high_T=35, unf_low_T=70, unf_high_T=90)`.
`struct_type` is `"heterodimer"` (default), `"homodimer"` or `"monomer"`.

Results are NamedTuples from `rnamelt.results`. A failed analysis is
**returned, not raised**: check `ok`, and `error` says why. Fits also report
`converged`, the starting values used (`seed`) and standard errors
(`SE_dH`, `SE_dS`, `SE_dG`).

Every tunable default lives in one settings class (`SolverSettings`,
`BoundsSettings`, `FitInitSettings`, …). Override it per curve or per call
with a dict:

```python
c = SingleCurve(T, signal, oligoC=0.5, vh={"border": 0.2})
c.fit(bounds_mode="unbounded", strategy="multistart")
```

Plotting (needs `rnamelt[plots]`):

```python
from rnamelt import plots
plots.plot_single(c); plots.plot_multi(gf); plots.plot_concentration(cdr)
```

### Upgrading from 0.1.x

The API was rewritten around `SingleCurve` / `MultiCurves`. `MeltAnalysis`,
the `analyze_*` helpers and the command-line tool are gone, and `oligoC` is
now required for dimers.

## Development

```bash
git clone https://github.com/tspich/RNAmelt
cd RNAmelt
pip install -e '.[plots]'

pytest tests/
python -m build               # sdist + wheel into dist/
python -m http.server 8000    # browser app at http://localhost:8000
```

The browser page is `index.html`; it talks to the package through
`rnamelt/web_bridge.py`.

## License

[MIT](LICENSE).

## Links

- Source: <https://github.com/tspich/RNAmelt>
- Issues: <https://github.com/tspich/RNAmelt/issues>
- Live demo: <https://tspich.github.io/RNAmelt/>
