# RNA Melt — In-Browser Van't Hoff Analysis

Fully serverless. Python (scipy curve fitting) runs via WebAssembly in the browser.
Data never leaves the user's machine.

## Project structure

```
rna/
├── index.html                    # UI, controls, charts
└── analysis/
    ├── __init__.py               # Module registry
    ├── utils.py                  # R constant, unit conversions, safe_json
    ├── cleaning.py               # CSV parsing, column renaming, sort by T
    └── analysis_melting.py       # Two-state fit + van't Hoff
```

## CSV format expected

```
temperature, sample_1, sample_2, ...
20.0,        0.412,    0.388, ...
20.5,        0.415,    0.391, ...
...
```

- Column 0 : Temperature in °C (any header name)
- Columns 1+: Absorbance or fluorescence signal (any header name)

## Running locally

Must be served via HTTP (not opened as file://).

```bash
cd rna
python -m http.server 8000
# open http://localhost:8000
```

## Interactive parameters

All adjustable in the sidebar without reloading:

| Parameter       | Description                                                  |
|-----------------|--------------------------------------------------------------|
| Signal column   | Choose which data column to fit (if multiple)                |
| T_low / T_high  | Transition window boundaries — only this region is fitted    |
| mL, bL          | Lower baseline slope & intercept: A_lower(T) = mL·T + bL    |
| mU, bU          | Upper baseline slope & intercept: A_upper(T) = mU·T + bU    |
| Fix baselines   | Hold mL/bL/mU/bU fixed; fit only ΔH and Tm                  |
| CT (µM)         | Strand concentration — adds a point to 1/Tm vs ln[CT] plot  |

Press **↺ Auto** to clear all overrides and re-detect baselines automatically.

## Two-state model

    A(T) = A_lower(T) · (1 − α) + A_upper(T) · α
    α(T) = K(T) / (1 + K(T))
    K(T) = exp(ΔH/R · (1/Tm − 1/T))

Fitted parameters: ΔH, Tm (and optionally mL, bL, mU, bU if not fixed).
Derived: ΔS = ΔH/Tm, ΔG°₂₅ = ΔH − 298.15·ΔS.

## Van't Hoff 1/Tm vs ln[CT]

For bimolecular / concentration-dependent melting:
Load multiple CSV files at different CT values, enter the CT for each run,
and the accumulated points build up the 1/Tm vs ln[CT] plot.
Once ≥ 3 points are collected, a linear regression gives ΔH and ΔS.

## Deployment

Drop the folder on any static host (GitHub Pages, Netlify, Cloudflare Pages).
No build step, no server, no cost.
# RNAmelt
