"""End-to-end demo of the rnamelt Python API against debug_data.csv.

Shows every public knob on `MeltAnalysis` plus every mode method and the
typed result objects they return. Run from the project root:

    python example_api.py
"""

from rnamelt import (
    MeltAnalysis,
    FIT_INIT_DEFAULTS,
    SOLVER_DEFAULTS,
    VH_DEFAULTS,
    FitFailed,
)

CSV_PATH = "debug_data.csv"


# ── 1. Build the analyzer ─────────────────────────────────────────────────
#
# Two equivalent ways to load + clean the CSV:
#
#   (a) MeltAnalysis.from_csv(path, **knobs)              — does the read+clean
#   (b) MeltAnalysis(clean(pd.read_csv(path)), **knobs)   — you do it yourself
#
# Every keyword shown below is optional except `df`. Defaults match the
# in-browser pipeline. `None` means "let the orchestrator decide".

m = MeltAnalysis.from_csv(
    CSV_PATH,

    # ── structural / experimental ────────────────────────────────────────
    struct_type     = "heterodimer",   # "heterodimer" | "homodimer" | "monomer"
    salt            = 150.0,           # NaCl (mM); metadata only — not in the model
    T_low           = None,            # transition-window lower edge (°C); None = full range
    T_high          = None,            # transition-window upper edge (°C); None = full range
    bl_lower_offset = 10.0,            # folded-baseline span above T_low   (°C)
    bl_upper_offset = 10.0,            # unfolded-baseline span below T_high (°C)

    # ── scipy.optimize.least_squares overrides ───────────────────────────
    # See rnamelt.SOLVER_DEFAULTS for the full default dict. Any key you
    # omit falls back to SOLVER_DEFAULTS. Pass None instead of a dict to
    # use the defaults wholesale.
    solver = {
        "max_nfev":         10000,       # int  — max function evaluations
        "ftol":             1e-8,        # float — cost-function tolerance
        "gtol":             1e-8,        # float — gradient tolerance
        "xtol":             1e-8,        # float — step-size tolerance
        "method":           "trf",       # "trf" | "dogbox" | "lm" (lm: no bounds)
        "loss":             "linear",    # "linear" | "soft_l1" | "huber" | "cauchy" | "arctan"
        "f_scale":          1.0,         # soft-margin for non-linear losses
        "jac":              "2-point",   # "2-point" | "3-point" | "cs"
        "verbose":          0,           # 0 silent | 1 term-report | 2 per-iter
        "residuals_method": "square",    # multi-fit only — how per-curve residuals
                                         # are aggregated. Ignored elsewhere.
    },

    # ── van't Hoff linearisation overrides ───────────────────────────────
    # See rnamelt.VH_DEFAULTS.
    vh = {
        "border":  0.15,    # θ cutoff; keep points with border ≤ θ ≤ 1-border
        "t1_min":  -1,      # lower clip on linearised x (-1 = no clip)
        "t1_max":  -1,      # upper clip on linearised x (-1 = no clip)
        "T_scale": 1000.0,  # numerical conditioning factor for 1/T
    },

    # ── full-fit initial guesses ─────────────────────────────────────────
    # See rnamelt.FIT_INIT_DEFAULTS. Leaving dH_init/dS_init as None lets
    # the orchestrator auto-seed them from the van't Hoff result.
    fit_init = {
        "dH_init":  None,   # kcal/mol;       None → auto-seed from vH
        "dS_init":  None,   # kcal/(mol·K);   None → auto-seed from vH
        "lin_init": 10,     # # of leading/trailing points for baseline-intercept seed
        "b1_init":  None,   # explicit (slope, intercept) for folded baseline; None = auto
        "b2_init":  None,   # explicit (slope, intercept) for unfolded baseline; None = auto
    },
)

print("signal columns:", m.signal_columns)
print()

# In-place reconfiguration (chainable). Unknown keys raise ValueError.
m.configure(salt=1000.0).configure(bl_lower_offset=8.0, bl_upper_offset=8.0)

# Independent copy if you want to fork the analyzer without affecting `m`.
m2 = m.copy()


# ── 2. Single-column mode ─────────────────────────────────────────────────

single = m.single("F4", oligo=0.5)   # oligo in µM

# Typed access — numpy arrays kept on the result object:
print(f"[single] Tm_raw     = {single.Tm_raw:.2f} °C")
print(f"[single] vH dH      = {single.vh.dH:.2f} kcal/mol  (ok={single.vh.ok})")
print(f"[single] fit dH     = {single.fit.dH:.2f} kcal/mol  (ok={single.fit.ok})")
print(f"[single] fit Tm     = {single.fit.Tm:.2f} °C")
print(f"[single] fit curve  : ndarray of length {len(single.fit.curve)}")
print()

# Failed sub-fits return None when accessed via attribute; `.require()`
# raises FitFailed instead. Useful for explicit error flow:
try:
    single.fit.require()
except FitFailed as e:
    print(f"[single] fit failed: {e}")

# Export forms:
legacy_dict = single.to_dict()           # byte-identical to analysis_melting.run output
df_summary  = single.to_dataframe()      # 1-row, 12-col tidy summary
# single.to_csv("single.csv")            # browser-download-format CSV (14-col legacy)


# ── 3. Batch single mode — every signal column ───────────────────────────

batch = m.single_all(oligo=0.5)          # dict[str, SingleResult]
for col, res in batch.items():
    tag = "ok" if res.fit and res.fit.ok else f"err: {res.error or 'fit failed'}"
    print(f"[batch] {col:>3}: fit Tm={res.fit.Tm if res.fit and res.fit.ok else float('nan'):>6.2f}  ({tag})")
print()


# ── 4. Multi mode — shared ΔH/ΔS across columns ──────────────────────────
# oligo_multi maps each signal column to its strand concentration (µM).

oligo_multi = {"F4": 0.5, "F5": 1.0, "F6": 2.0, "F7": 4.0, "F8": 8.0, "F9": 16.0}

multi = m.multi(oligo_multi)
if multi.error:
    print(f"[multi]  error: {multi.error}")
else:
    print(f"[multi]  shared dH = {multi.dH:.2f} kcal/mol")
    print(f"[multi]  shared dS = {multi.dS*1000:.2f} cal/(mol·K)")
    print(f"[multi]  columns   = {[c.name for c in multi.columns]}")
    for c in multi.columns:
        print(f"[multi]    {c.name}: Tm_fit={c.Tm_fit:.2f} °C  oligoC={c.oligoC} µM")
print()


# ── 5. Concentration-series mode (1/Tm vs ln(C_T / f)) ───────────────────

conc = m.concentration(oligo_multi)
if conc.error:
    print(f"[conc]   error: {conc.error}")
else:
    print(f"[conc]   per-curve count    = {len(conc.per_curve)}")
    print(f"[conc]   skipped curves     = {len(conc.skipped)}")
    for label, series in [
        ("raw", conc.series_raw),
        ("vH",  conc.series_vh),
        ("fit", conc.series_fit),
    ]:
        if series.ok:
            print(f"[conc]   {label:>3}-series   dH = {series.dH:>8.2f}  "
                  f"dG37 = {series.dG_37:>6.2f}  r² = {series.r_squared:.4f}")
        else:
            print(f"[conc]   {label:>3}-series   (not ok)")
print()


# ── 6. last_result + introspection ────────────────────────────────────────

print("last_result type:", type(m.last_result).__name__)
print("SOLVER_DEFAULTS keys:", sorted(SOLVER_DEFAULTS))
print("VH_DEFAULTS keys    :", sorted(VH_DEFAULTS))
print("FIT_INIT_DEFAULTS   :", sorted(FIT_INIT_DEFAULTS))


# ── 7. Plotting (optional — requires matplotlib) ─────────────────────────
# Each result class has a `.plot()` method that lazy-imports matplotlib.
# Use a non-interactive backend if you only want to save to disk.

import matplotlib
matplotlib.use("Agg")           # headless; comment out for interactive use
import matplotlib.pyplot as plt # noqa: E402

# Single result — 3-panel figure: raw / van't Hoff / full fit.
fig, _ = single.plot(figsize=(15, 5))
fig.savefig("plot_single.png", dpi=120)
plt.close(fig)
print("[plot]   wrote plot_single.png")

# MeltAnalysis convenience — equivalent to m.single(column).plot().
fig, _, _ = m.plot("F4")
fig.savefig("plot_F4.png", dpi=120)
plt.close(fig)
print("[plot]   wrote plot_F4.png")

# Multi result — one panel with all column raw + shared-ΔH fit curves.
if not multi.error:
    fig, _ = multi.plot(figsize=(10, 6))
    fig.savefig("plot_multi.png", dpi=120)
    plt.close(fig)
    print("[plot]   wrote plot_multi.png")

# Concentration result — two panels: per-curve overlay + 1/Tm vs ln(C_T/f).
if not conc.error:
    fig, _ = conc.plot(figsize=(13, 5))
    fig.savefig("plot_concentration.png", dpi=120)
    plt.close(fig)
    print("[plot]   wrote plot_concentration.png")
