# rnamelt

[![PyPI](https://img.shields.io/pypi/v/rnamelt.svg)](https://pypi.org/project/rnamelt/)
[![Python versions](https://img.shields.io/pypi/pyversions/rnamelt.svg)](https://pypi.org/project/rnamelt/)
[![License: MIT](https://img.shields.io/pypi/l/rnamelt.svg)](https://github.com/tspich/RNAmelt/blob/main/LICENSE)

Two-state thermodynamic analysis (ΔH, ΔS, ΔG, Tm) of UV-absorbance and
fluorescence RNA / DNA melting curves. One small, dependency-light Python
package that works as a library and as a browser app.

- **Library** — `from rnamelt import SingleCurve, MultiCurves`, typed
  result objects, optional `matplotlib` plotting.
- **Browser** — the same Python pipeline runs client-side via Pyodide.
  Live demo: <https://tspich.github.io/RNAmelt/>.

## Install

```bash
pip install rnamelt                 # base — numpy, pandas, scipy
pip install 'rnamelt[plots]'        # adds matplotlib for rnamelt.plots
```

Requires Python ≥ 3.10.

## Quickstart

```python
import pandas as pd
from rnamelt import SingleCurve

df = pd.read_csv("melt.csv")
T = df.iloc[:, 0].to_numpy(float)

c = SingleCurve(T, df["sample_1"].to_numpy(float), name="sample_1", oligoC=0.5)
f = c.fit()
if f.ok:
    print(f.dH, f.dS, f.dG, f.Tm)   # kcal/mol, kcal/(mol·K), kcal/mol at 37 °C, °C
else:
    print(f.error)
```

That is all a default run needs: data and the per-strand concentration.
Baseline windows are detected automatically, and the fit is seeded from a
van't Hoff estimate.

## CSV input format

```
temperature, sample_1, sample_2, ...
20.0,        0.412,    0.388, ...
20.5,        0.415,    0.391, ...
```

- Column 0: temperature in °C (any header name).
- Columns 1+: absorbance or fluorescence (any header name).
- `rnamelt.cleaning.clean(df)` drops all-NaN rows and rows without a
  temperature, and coerces every column to numbers.

## Python API

### One curve — `SingleCurve`

```python
from rnamelt import SingleCurve

c = SingleCurve(
    T, signal,
    name="sample_1",
    oligoC=0.5,                   # per-strand concentration, µM (required for dimers)
    struct_type="heterodimer",    # "heterodimer" | "homodimer" | "monomer"
)

r  = c.raw_tm()      # RawTm      — Tm where the mean baseline crosses the data
vh = c.vant_hoff()   # VanHoffFit — linear fit of ln K vs 1/T
f  = c.fit()         # SingleFit  — non-linear fit of the full two-state model
```

Each method returns a NamedTuple from `rnamelt.results`. A failed
analysis is **returned, not raised**: `ok` is False and `error` says why.
Always check `ok` before reading values.

`SingleFit` also reports:

- `converged` — False if the solver stopped at `max_nfev`. The best result
  is still returned, and a `ConvergenceWarning` is issued.
- `seed` — the starting values used, and where they came from:
  `{"dH", "dS", "dH_from", "dS_from"}`, with `*_from` in
  `"user" | "vh" | "fallback" | "multistart"`.

A fit whose model Tm lies more than `bounds["Tm_margin"]` (3 °C) outside
the fitted temperature range did not find the transition. It is returned
with `ok=False`.

### Baseline windows

Four temperatures (°C) define the baselines. The folded baseline is fitted
on `[fold_low_T, fold_high_T]`, the unfolded one on
`[unf_low_T, unf_high_T]`, and the full fit uses `[fold_low_T, unf_high_T]`.

```python
c = SingleCurve(T, signal, oligoC=0.5,
                fold_low_T=15, fold_high_T=35, unf_low_T=70, unf_high_T=90)
```

Give all four or none. When they are omitted, they are detected from the
curve's derivatives (`c.window_source == "auto"`). If detection fails, edge
windows of `windows["fallback_width"]` °C are used and a `BaselineWarning`
is issued (`window_source == "fallback"`). `c.auto_baselines()` returns the
detected windows without changing the curve.

### Several curves — `MultiCurves`

```python
from rnamelt import SingleCurve, MultiCurves

curves = [SingleCurve(T, df[col].to_numpy(float), name=col, oligoC=oligo)
          for col, oligo in {"s1": 0.5, "s2": 5.0, "s3": 50.0}.items()]
m = MultiCurves(curves)

gf = m.global_fit()          # GlobalFit — one ΔH/ΔS shared by all curves,
                             # baselines per curve
gf.dH, gf.dS, gf.dG
for cc in gf.curves:         # GlobalFitCurve
    print(cc.name, cc.oligoC, cc.Tm)

cdr = m.conc_dependency()    # ConcDepResult — 1/Tm vs ln(C_T/f)
for s in (cdr.series_raw, cdr.series_vh, cdr.series_fit):
    if s.ok:
        print(s.method, s.dH, s.dS, s.dG, s.r_squared)
```

Curves are independent: each keeps its own concentration, windows and
settings, and curves may have different lengths. All curves must share one
`struct_type`. `conc_dependency` extracts three Tm values per curve (raw,
van't Hoff, full fit) and runs one regression for each. The factor `f` is
1 for self-complementary strands (homodimer) and 4 otherwise. It does not
apply to monomers, whose Tm does not depend on concentration. Curves whose
Tm cannot be extracted are listed in `cdr.skipped`.

`SingleFit`, `GlobalFit` and each `ConcentrationSeries` carry standard
errors `SE_dH`, `SE_dS`, `SE_dG` (same units as the values) and
`cov_dH_dS`; `SingleFit` also has `SE_Tm`:

- single and global fits: from the covariance s²·(JᵀJ)⁻¹ at the solution;
  the Tm error is propagated from ΔH and ΔS through
  Tm = ΔH / (ΔS + R·ln(C_T/f));
- concentration series: from the scatter of the points about the line,
  propagated from slope and intercept (needs at least 3 curves).

The single-curve van't Hoff estimate and the raw Tm have no errors: their
points are derived from one curve through the chosen baselines, so a
formal error would be misleading.

The ΔG error keeps the ΔH–ΔS covariance,
var(ΔG) = var(ΔH) + T²·var(ΔS) − 2T·cov(ΔH, ΔS), which matters: ΔH and ΔS
are strongly correlated. When an error is undefined (2 curves, or ΔH / ΔS
at a fit bound), the SE fields are None and `SE_note` says why. These are
the formal errors from measurement noise; they do not include the effect
of the baseline choice.

### Settings

Every tunable default lives in one dict, exported from `rnamelt`:

| Group        | Dict                  | Controls |
|--------------|-----------------------|----------|
| `windows`    | `WINDOW_DEFAULTS`     | `detector` options passed to baseline detection, `fallback_width` |
| `solver`     | `SOLVER_DEFAULTS`     | `scipy.optimize.least_squares` options (`max_nfev`, tolerances, `loss`, `x_scale`, …) |
| `vh`         | `VH_DEFAULTS`         | van't Hoff: `border` (θ range kept), `t1_min` / `t1_max`, `T_scale` |
| `fit_init`   | `FIT_INIT_DEFAULTS`   | starting values: `seed_from`, `dH_init`, `dS_init`, `lin_init`, `b1_init`, `b2_init`, fallbacks and plausible ranges |
| `bounds`     | `BOUNDS_DEFAULTS`     | `mode` (baseline slopes), `dH` / `dS` boxes, `slope_cap_k`, `Tm_margin` |
| `multistart` | `MULTISTART_DEFAULTS` | `dH_seeds`, `fwhm_factors`, `seed_Tm_K`, `min_abs_dH` |
| `raw`        | `RAW_DEFAULTS`        | `edge_skip` for the raw-Tm crossing search |

Pass a partial dict to override any group, either **per curve** (it is
validated when the curve is built, and typos raise `ValueError`) or **per
call** (it applies to that call only):

```python
c = SingleCurve(T, signal, oligoC=0.5,
                vh={"border": 0.2},
                bounds={"mode": "loose"})

c.fit(solver={"max_nfev": 500})
c.fit(bounds_mode="unbounded")                  # shortcut for bounds={"mode": ...}
c.fit(strategy="multistart")                    # one fit per dH seed, best wins
c.vant_hoff(vh={"t1_min": 2.9, "t1_max": 3.1})
```

The dH / dS starting values follow `fit_init["seed_from"]`:

| `seed_from`  | Starting values |
|--------------|-----------------|
| `"auto"`     | your `dH_init` / `dS_init` where given, otherwise van't Hoff, otherwise the fallbacks |
| `"vh"`       | van't Hoff, otherwise the fallbacks (ignores `dH_init` / `dS_init`) |
| `"user"`     | `dH_init` and `dS_init` (both required) |
| `"fallback"` | `dH_fallback` / `dS_fallback`, without running van't Hoff |

The van't Hoff dH and dS are used together or not at all: if either one
falls outside its plausible range, both fall back.

`global_fit` and `conc_dependency` take the same per-call overrides:

```python
m.conc_dependency(fit_init={"seed_from": "user", "dH_init": -100, "dS_init": -0.28})
m.global_fit(bounds={"dH": (-250, 0)})
```

For `global_fit`, the settings that define the shared fit must agree
across curves: the solver, `seed_from` and the dH / dS starting values,
fallbacks and boxes. A conflict raises `ValueError` naming the setting;
setting that value in the call resolves it.

### Plotting (optional)

`pip install 'rnamelt[plots]'`. matplotlib is only imported by
`rnamelt.plots`, so it never enters the browser bundle.

```python
from rnamelt import plots

fig, axes = plots.plot_single(c)                 # raw / van't Hoff / fit
fig, ax   = plots.plot_multi(m.global_fit())
fig, axes = plots.plot_concentration(m.conc_dependency())
```

### Upgrading from 0.1.x

The API was rewritten. `MeltAnalysis`, the `analyze_*` helpers and the
`rnamelt` command-line tool are gone. Build `SingleCurve` / `MultiCurves`
objects instead.

- `T_low` / `T_high` / `bl_lower_offset` / `bl_upper_offset` are replaced
  by the four window temperatures (or automatic detection).
- `oligoC` is required for heterodimers and homodimers.
- `vantHoff` → `vant_hoff`; `structType` → `struct_type`.

## Two-state model

The observed signal is a mix of the folded and unfolded baselines, weighted
by the folded fraction θ(T):

```
A(T) = (m1·T + b1)·θ + (m2·T + b2)·(1 − θ)
ΔG   = ΔH − T·ΔS,     K = exp(−ΔG / RT)
θ    = K / (1 + K)                          monomer
θ    = (s − 1) / (s + 1),  s = √(1 + 2·C_T·K)    heterodimer
θ    = (s − 1) / (s + 1),  s = √(1 + 8·C_T·K)    homodimer
Tm   = ΔH / (ΔS + R·ln(C_T / f)),  f = 4 (hetero), 1 (homo);  ΔH / ΔS (monomer)
```

ΔH, ΔS and ΔG are for **folding**, so ΔH and ΔS are negative for a stable
duplex. `C_T` is the total strand concentration in M: 2 × `oligoC` for a
heterodimer, `oligoC` for a homodimer.

Units: ΔH in kcal/mol, ΔS in kcal/(mol·K), ΔG in kcal/mol at 37 °C.
Tm is reported in °C; the model works in K (`T_K = T_C + 273.15`).

## Browser app

The same Python package runs client-side via [Pyodide](https://pyodide.org/):
no server, no upload, no install. Drop a CSV onto
<https://tspich.github.io/RNAmelt/> to try it. The page is `index.html` in
the [repository](https://github.com/tspich/RNAmelt); it talks to the
package through `rnamelt/web_bridge.py`.

## Development

```bash
git clone https://github.com/tspich/RNAmelt
cd RNAmelt
pip install -e '.[plots]'

pytest tests/                 # or: python -m unittest discover tests
python -m build               # sdist + wheel into dist/
python -m http.server 8000    # browser app at http://localhost:8000
```

Module layering is strict: `functions` → `methods` → `api` →
`web_bridge` / `plots`. `functions` stays stateless. `plots` is
lazy-imported so the browser bundle never pulls in matplotlib.

## License

[MIT](LICENSE).

## Links

- Source: <https://github.com/tspich/RNAmelt>
- Issues: <https://github.com/tspich/RNAmelt/issues>
- Live demo: <https://tspich.github.io/RNAmelt/>
