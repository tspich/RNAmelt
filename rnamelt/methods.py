import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy import stats

from rnamelt import functions, constants


class ConvergenceWarning(UserWarning):
    """The nonlinear fit stopped at max_nfev before meeting its tolerances."""


class BaselineWarning(UserWarning):
    """Automatic baseline-window detection failed; fallback windows used."""


def _check_fit_status(res_lsq):
    """Return True if converged; warn and return False if max_nfev was hit;
    raise on any other solver failure."""
    if res_lsq.status > 0:
        return True
    if res_lsq.status == 0:
        warnings.warn(
            f"least_squares stopped at max_nfev={res_lsq.nfev} before "
            f"converging; returning the best parameters found",
            ConvergenceWarning, stacklevel=3,
        )
        return False
    raise ValueError(f"least_squares failed: {res_lsq.message}")

# ── Settings ─────────────────────────────────────────────────────────────────
# Every tunable default lives in one of the dicts below. SingleCurve takes
# one override dict per group (solver=, vh=, fit_init=, bounds=,
# multistart=, raw=). Unknown keys and invalid values are rejected when the
# curve is built; a value of None in an override means "keep the default".

# scipy.optimize.least_squares options for the nonlinear fits. All keys but
# `x_scale` are forwarded as-is. `x_scale` is the characteristic scale of
# each parameter kind, expanded to the full parameter vector per fit.
SOLVER_DEFAULTS = {
    "max_nfev": 1e12,
    "ftol":     1e-8,
    "gtol":     1e-8,
    "xtol":     1e-8,
    "method":   "trf",
    "loss":     "linear",
    "f_scale":  0.02,
    "jac":      "2-point",
    "verbose":  0,
    "x_scale":  {"dH": 1.0, "dS": 0.01, "slope": 1.0, "intercept": 100.0},
}

# Subset of SOLVER_DEFAULTS forwarded as-is to least_squares().
_SCIPY_LEAST_SQUARES_KEYS = (
    "max_nfev", "ftol", "gtol", "xtol",
    "method", "loss", "f_scale", "jac", "verbose",
)

# van't Hoff linearisation (ln K vs T_scale/T). Points with folded fraction
# outside [border, 1-border] are excluded; t1_min / t1_max clip the
# linearised x-axis (None = no clip).
VH_DEFAULTS = {
    "border":  0.15,
    "t1_min":  None,
    "t1_max":  None,
    "T_scale": 1000.0,
}

# Starting values for the nonlinear fit. `seed_from` picks where the dH /
# dS starting values come from:
#   "auto"     : dH_init / dS_init if set, else the van't Hoff estimate if it
#                lies inside dH_seed_range / dS_seed_range, else the fallback
#   "vh"       : van't Hoff estimate (if in range), else the fallback;
#                dH_init / dS_init are ignored
#   "user"     : dH_init / dS_init, both required
#   "fallback" : dH_fallback / dS_fallback (no van't Hoff is computed)
# lin_init = number of leading / trailing points averaged for the
# baseline-intercept seeds; b1_init / b2_init = explicit (slope, intercept)
# seeds for the folded / unfolded baselines.
SEED_POLICIES = ("auto", "vh", "user", "fallback")
FIT_INIT_DEFAULTS = {
    "seed_from":     "auto",
    "dH_init":       None,
    "dS_init":       None,
    "lin_init":      10,
    "b1_init":       None,
    "b2_init":       None,
    "dH_fallback":   -80.0,
    "dS_fallback":   -0.2,
    "dH_seed_range": (-150.0, 0.0),
    "dS_seed_range": (-5.0, 0.0),
}

# Parameter bounds for the nonlinear fit. dH (kcal/mol) and dS
# (kcal/(mol·K)) are boxed by `dH` / `dS`; `mode` bounds the baseline
# slopes (see _fit_bounds) and `slope_cap_k` sets the cap of "physical".
# A fit whose model Tm lies more than `Tm_margin` °C outside the fitted
# window did not find the transition and is reported as failed.
BOUNDS_MODES = ("physical", "loose", "positive", "unbounded")
BOUNDS_DEFAULTS = {
    "mode":        "physical",
    "dH":          (-200.0, 0.0),
    "dS":          (-5.0, 0.0),
    "slope_cap_k": 5.0,
    "Tm_margin":   3.0,
}

# strategy="multistart": one fit per dH seed (plus fwhm_factors × the
# FWHM-of-dS/dT estimate), dS seed = dH seed / seed_Tm_K. A result counts
# counts only if it passes the Tm check (bounds["Tm_margin"]) and
# |dH| >= min_abs_dH; the lowest-RMS such result wins.
MULTISTART_DEFAULTS = {
    "dH_seeds":     (-40.0, -80.0, -120.0, -160.0),
    "fwhm_factors": (1.0, 1.5),
    "seed_Tm_K":    350.0,
    "min_abs_dH":   10.0,
}

# Raw Tm (crossing of the mean baseline with the data): the first / last
# `edge_skip` points are ignored when searching for the crossing.
RAW_DEFAULTS = {
    "edge_skip": 10,
}


# Baseline windows used when a SingleCurve is built without explicit
# windows: detected with auto_baseline.find_baselines (`detector` = its
# keyword arguments); if detection fails, `fallback_width` °C at each scan
# edge, with a BaselineWarning.
WINDOW_DEFAULTS = {
    "detector":       {},
    "fallback_width": 10.0,
}


def _merge(group, defaults, overrides):
    """Overlay overrides on defaults; reject unknown keys, skip None values."""
    out = dict(defaults)
    if overrides:
        for k, v in dict(overrides).items():
            if k not in defaults:
                raise ValueError(
                    f"unknown {group} option {k!r}; valid keys: {sorted(defaults)}"
                )
            if v is not None:
                out[k] = v
    return out


def _finite(group, key, v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{group}[{key!r}] must be a number, got {v!r}")
    if not np.isfinite(f):
        raise ValueError(f"{group}[{key!r}] must be finite, got {v!r}")
    return f


def _interval(group, key, v):
    try:
        lo, hi = v
    except (TypeError, ValueError):
        raise ValueError(f"{group}[{key!r}] must be a (low, high) pair, got {v!r}")
    lo, hi = _finite(group, key, lo), _finite(group, key, hi)
    if not lo < hi:
        raise ValueError(f"{group}[{key!r}] needs low < high, got {v!r}")
    return (lo, hi)


def _positive_int(group, key, v, minimum=1):
    if isinstance(v, bool) or not float(v).is_integer() or int(v) < minimum:
        raise ValueError(f"{group}[{key!r}] must be an integer >= {minimum}, got {v!r}")
    return int(v)


def _solver_settings(overrides=None):
    """Validated solver settings (defaults + overrides)."""
    s = _merge("solver", SOLVER_DEFAULTS, overrides)
    xs = _merge("solver['x_scale']", SOLVER_DEFAULTS["x_scale"],
                (overrides or {}).get("x_scale"))
    for k, v in xs.items():
        if _finite("solver['x_scale']", k, v) <= 0:
            raise ValueError(f"solver['x_scale'][{k!r}] must be > 0, got {v!r}")
    s["x_scale"] = xs
    if _finite("solver", "max_nfev", s["max_nfev"]) < 1:
        raise ValueError(f"solver['max_nfev'] must be >= 1, got {s['max_nfev']!r}")
    return s


def _vh_settings(overrides=None):
    """Validated van't Hoff settings; the dict doubles as vant_hoff() kwargs."""
    s = _merge("vh", VH_DEFAULTS, overrides)
    b = _finite("vh", "border", s["border"])
    if not 0.0 <= b < 0.5:
        raise ValueError(f"vh['border'] must be in [0, 0.5), got {b}")
    if _finite("vh", "T_scale", s["T_scale"]) <= 0:
        raise ValueError(f"vh['T_scale'] must be > 0, got {s['T_scale']!r}")
    for k in ("t1_min", "t1_max"):
        if s[k] == -1:             # legacy "no clip" sentinel
            s[k] = None
        if s[k] is not None:
            s[k] = _finite("vh", k, s[k])
    if s["t1_min"] is not None and s["t1_max"] is not None and not s["t1_min"] < s["t1_max"]:
        raise ValueError(f"vh needs t1_min < t1_max, got {s['t1_min']}, {s['t1_max']}")
    return s


def _fit_init_settings(overrides=None):
    """Validated fit starting-value settings."""
    s = _merge("fit_init", FIT_INIT_DEFAULTS, overrides)
    if s["seed_from"] not in SEED_POLICIES:
        raise ValueError(f"unknown fit_init['seed_from'] {s['seed_from']!r}; "
                         f"valid: {list(SEED_POLICIES)}")
    if s["seed_from"] == "user" and (s["dH_init"] is None or s["dS_init"] is None):
        raise ValueError("fit_init['seed_from']='user' needs both dH_init and dS_init")
    for k in ("dH_init", "dS_init"):
        if s[k] is not None:
            s[k] = _finite("fit_init", k, s[k])
    for k in ("dH_fallback", "dS_fallback"):
        s[k] = _finite("fit_init", k, s[k])
    for k in ("dH_seed_range", "dS_seed_range"):
        s[k] = _interval("fit_init", k, s[k])
    s["lin_init"] = _positive_int("fit_init", "lin_init", s["lin_init"])
    for k in ("b1_init", "b2_init"):
        if s[k] is not None:
            try:
                m, b = s[k]
            except (TypeError, ValueError):
                raise ValueError(f"fit_init[{k!r}] must be a (slope, intercept) pair, got {s[k]!r}")
            s[k] = (_finite("fit_init", k, m), _finite("fit_init", k, b))
    return s


def _bounds_settings(overrides=None):
    """Validated fit-bounds settings."""
    s = _merge("bounds", BOUNDS_DEFAULTS, overrides)
    if s["mode"] not in BOUNDS_MODES:
        raise ValueError(f"unknown bounds mode {s['mode']!r}; valid: {list(BOUNDS_MODES)}")
    s["dH"] = _interval("bounds", "dH", s["dH"])
    s["dS"] = _interval("bounds", "dS", s["dS"])
    if _finite("bounds", "slope_cap_k", s["slope_cap_k"]) <= 0:
        raise ValueError(f"bounds['slope_cap_k'] must be > 0, got {s['slope_cap_k']!r}")
    if _finite("bounds", "Tm_margin", s["Tm_margin"]) < 0:
        raise ValueError(f"bounds['Tm_margin'] must be >= 0, got {s['Tm_margin']!r}")
    return s


def _multistart_settings(overrides=None):
    """Validated multistart settings."""
    s = _merge("multistart", MULTISTART_DEFAULTS, overrides)
    for k in ("dH_seeds", "fwhm_factors"):
        try:
            vals = tuple(_finite("multistart", k, v) for v in s[k])
        except TypeError:
            raise ValueError(f"multistart[{k!r}] must be a sequence of numbers, got {s[k]!r}")
        s[k] = vals
    if not s["dH_seeds"] and not s["fwhm_factors"]:
        raise ValueError("multistart needs at least one dH seed or fwhm factor")
    if _finite("multistart", "seed_Tm_K", s["seed_Tm_K"]) <= 0:
        raise ValueError(f"multistart['seed_Tm_K'] must be > 0, got {s['seed_Tm_K']!r}")
    if _finite("multistart", "min_abs_dH", s["min_abs_dH"]) < 0:
        raise ValueError(f"multistart['min_abs_dH'] must be >= 0, got {s['min_abs_dH']!r}")
    return s


def _raw_settings(overrides=None):
    """Validated raw-Tm settings."""
    s = _merge("raw", RAW_DEFAULTS, overrides)
    s["edge_skip"] = _positive_int("raw", "edge_skip", s["edge_skip"], minimum=0)
    return s


def _window_settings(overrides=None):
    """Validated window settings. The detector keys themselves are checked
    against find_baselines by the API (this module doesn't import it)."""
    s = _merge("windows", WINDOW_DEFAULTS, overrides)
    if not isinstance(s["detector"], dict):
        raise ValueError(f"windows['detector'] must be a dict, got {s['detector']!r}")
    s["detector"] = dict(s["detector"])
    if _finite("windows", "fallback_width", s["fallback_width"]) <= 0:
        raise ValueError(f"windows['fallback_width'] must be > 0, got {s['fallback_width']!r}")
    return s


_SETTINGS_FN = {
    "windows":    _window_settings,
    "solver":     _solver_settings,
    "vh":         _vh_settings,
    "fit_init":   _fit_init_settings,
    "bounds":     _bounds_settings,
    "multistart": _multistart_settings,
    "raw":        _raw_settings,
}


def _layer(group, base, override):
    """Validated settings for ``group``: ``override`` applied on top of the
    already-resolved ``base`` (e.g. a curve's settings). Nested dicts such as
    solver['x_scale'] are merged key by key; None values keep the base."""
    combined = dict(base)
    for k, v in dict(override or {}).items():
        if v is None:
            continue
        if isinstance(combined.get(k), dict) and isinstance(v, dict):
            combined[k] = {**combined[k], **v}
        else:
            combined[k] = v
    return _SETTINGS_FN[group](combined)


def _least_squares_kwargs(solver):
    """least_squares kwargs (without x_scale) from solver settings/overrides."""
    s = _solver_settings(solver)
    return {k: s[k] for k in _SCIPY_LEAST_SQUARES_KEYS}


def _x_scale(solver, n_curves=1):
    """x_scale vector for [dH, dS] + n_curves × [m1, b1, m2, b2]."""
    xs = _solver_settings(solver)["x_scale"]
    per_curve = [xs["slope"], xs["intercept"], xs["slope"], xs["intercept"]]
    return [xs["dH"], xs["dS"]] + per_curve * n_curves


def T_m_ds_raw(temperatures,
               ydata,
               baseline_bound_minT = None,
               baseline_bound_maxT = None,
               baseline_unbound_minT = None,
               baseline_unbound_maxT = None,
               edge_skip = None):
    """Raw Tm: crossing of the mean of the two OLS baselines with the data.

    The folded baseline is fitted on [baseline_bound_minT,
    baseline_bound_maxT], the unfolded one on [baseline_unbound_minT,
    baseline_unbound_maxT] (°C; the inner edges are required, missing outer
    edges default to the scan edges). ``edge_skip`` (default RAW_DEFAULTS)
    end points are ignored when searching for the crossing.
    Returns ``(Tm, y_Tm, base_b, base_ub, base_med)``.
    """
    T = np.array(temperatures)
    if edge_skip is None:
        edge_skip = RAW_DEFAULTS["edge_skip"]
    if baseline_bound_maxT is None or baseline_unbound_minT is None:
        raise ValueError("T_m_ds_raw needs baseline_bound_maxT and baseline_unbound_minT")

    bl_b_minT  = T[0]  if baseline_bound_minT   is None else baseline_bound_minT
    bl_ub_maxT = T[-1] if baseline_unbound_maxT is None else baseline_unbound_maxT
    bl_b_maxT  = baseline_bound_maxT
    bl_ub_minT = baseline_unbound_minT

    if bl_b_maxT <= bl_b_minT:
        raise ValueError(
            f"folded baseline window is empty or inverted: "
            f"[{bl_b_minT}, {bl_b_maxT}] °C"
        )
    if bl_ub_maxT <= bl_ub_minT:
        raise ValueError(
            f"unfolded baseline window is empty or inverted: "
            f"[{bl_ub_minT}, {bl_ub_maxT}] °C"
        )

    m_bound, b_bound = functions.fit_linear(T, ydata, bl_b_minT, bl_b_maxT)
    m_unbound, b_unbound = functions.fit_linear(T, ydata, bl_ub_minT, bl_ub_maxT)

    # median of base lines
    m_med, b_med = (m_bound + m_unbound) / 2, (b_bound + b_unbound) / 2

    T_m, y_dat = functions.intersect_lin(m_med, b_med, T, ydata, edge_skip=edge_skip)

    return T_m, y_dat, (m_bound, b_bound), (m_unbound, b_unbound), (m_med, b_med)

def vant_hoff(T,
             signal,
             m_bound,
             b_bound,
             m_unbound,
             b_unbound,
             c0,
             border = None,
             T_scale = None,
             t1_min = None,
             t1_max = None,
             struct_type = 'heterodimer'):
    """Linearised van't Hoff analysis. Unset options take VH_DEFAULTS."""
    vh = _vh_settings({"border": border, "T_scale": T_scale,
                       "t1_min": t1_min, "t1_max": t1_max})
    border, T_scale = vh["border"], vh["T_scale"]
    t1_min, t1_max = vh["t1_min"], vh["t1_max"]
    # ln K(θ) = ln(a · θ / (c0 (1-θ)²)) with a = 2 for A + B ⇌ AB and
    # a = 1/2 for 2A ⇌ A₂ (θ = folded fraction); K at Tm (θ = 1/2) is
    # 4/c0 and 1/c0 respectively — see functions.tm_from_therm.
    if struct_type == 'heterodimer':
        a, k_tm = 2.0, 4.0
    elif struct_type == 'homodimer':
        a, k_tm = 0.5, 1.0
    elif struct_type == 'monomer':
        raise ValueError("Van't Hoff analysis not possible for monomers")
    else:
        raise ValueError(
            f"unknown struct_type {struct_type!r}; "
            f"valid: 'heterodimer', 'homodimer', 'monomer'"
        )

    # fraction folded from the two linear baselines
    f_folded = [
        (dd - functions.linear(T[i], m_unbound, b_unbound))
        / (
            functions.linear(T[i], m_bound, b_bound)
            - functions.linear(T[i], m_unbound, b_unbound)
        )
        for i, dd in enumerate(signal)
    ]

    K = [
        np.log(a * ff / (c0 * ((1.0 - ff) ** 2)))
        if border <= ff <= (1 - border)
        else None
        for ff in f_folded
    ]

    t1 = [T_scale / (t - constants.T0) for t in T]

    # keep points with a defined ln K and t1 inside [t1_min, t1_max]
    # (-1 or None = no clip on that side)
    lo = None if t1_min in (-1, None) else t1_min
    hi = None if t1_max in (-1, None) else t1_max
    tt, lnK = [], []
    for x, k in zip(t1, K):
        if k is None:
            continue
        if lo is not None and x < lo:
            continue
        if hi is not None and x > hi:
            continue
        tt.append(x)
        lnK.append(k)

    if len(tt) < 2:
        raise ValueError(
            f"van't Hoff regression needs at least 2 points with "
            f"{border} ≤ θ ≤ {1 - border}"
            + (f" and t1 in [{lo}, {hi}]" if lo is not None or hi is not None else "")
            + f"; got {len(tt)}"
        )

    xdata = np.array(tt)
    ydata = np.array(lnK)

    res_lsq = least_squares(functions.linear_res,
                            [1, 1],
                            args=(xdata, ydata))

    t_m = (np.log(k_tm / c0) - res_lsq.x[1]) / res_lsq.x[0]
    T_m = T_scale / t_m + constants.T0
    dH  = -res_lsq.x[0] * constants.R * T_scale
    dS    = dH / (T_m - constants.T0) + constants.R * np.log(k_tm / c0)
    dG_37 = dH - (37. - constants.T0) * dS

    return T_m, dG_37, dH, dS, t1, K, xdata, ydata, res_lsq.x

def vant_hoff_concentration(
    tm_values,
    concentrations,
    self_complementary: bool = False,
) -> dict:
    """Linear-regression van't Hoff fit of 1/Tm vs ln(C_T/f) → ΔH, ΔS, ΔG(37°C)."""
    tm = np.asarray(tm_values, dtype=float)
    ct = np.asarray(concentrations, dtype=float)

    if tm.shape != ct.shape:
        raise ValueError(
            f"tm_values and concentrations must have the same shape "
            f"(got {tm.shape} and {ct.shape})"
        )
    if tm.ndim != 1:
        raise ValueError("tm_values and concentrations must be 1-D")
    if tm.size < 2:
        raise ValueError("Need at least 2 (Tm, C_T) points for regression")
    if np.any(~np.isfinite(tm)) or np.any(tm <= 0):
        raise ValueError("Tm values must be finite and > 0 (Kelvin)")
    if np.any(~np.isfinite(ct)) or np.any(ct <= 0):
        raise ValueError("Concentrations must be finite and > 0 (Molar)")

    f = 1.0 if self_complementary else 4.0
    ln_ct = np.log(ct / f)
    inv_tm = 1.0 / tm

    if np.allclose(ln_ct, ln_ct[0]):
        raise ValueError(
            "All concentrations are identical — cannot fit slope of "
            "1/Tm vs ln(C_T)"
        )

    #slope, intercept, r, _, _ = stats.linregress(ln_ct, inv_tm)
    reg = stats.linregress(ln_ct, inv_tm)
    slope     = float(reg.slope)
    intercept = float(reg.intercept)
    r_value   = float(reg.rvalue)


    if slope == 0:
        raise ValueError("Regression slope is zero; cannot derive ΔH")

    dH = constants.R / slope
    dS = intercept * dH
    dG_37 = dH - (37.0 - constants.T0) * dS

    return {
        "dH":             float(dH),
        "dS":             float(dS),
        "dG_37":          float(dG_37),
        "r_squared":      r_value ** 2,
        "slope":          slope,
        "intercept":      intercept,
        "ln_ct":          ln_ct,
        "inv_tm":         inv_tm,
        "tm_values":      tm,
        "concentrations": ct,
        "self_complementary": bool(self_complementary),
    }

def _slope_cap(T, d, k=5.0):
    """Baseline-slope magnitude cap: k× the naive whole-data slope amplitude.

    Under a two-state model the folded/unfolded baselines are linear regions
    of the signal; a fitted slope much larger than the overall
    (F_max-F_min)/(T_max-T_min) implies the baseline is absorbing part of the
    transition and is unphysical. k=5 leaves plenty of headroom for real
    baseline drift while ruling out the pathological "baseline explains the
    whole curve" minima that trap the solver at the initial dH seed.
    """
    span_T = float(T[-1] - T[0])
    span_S = float(np.nanmax(d) - np.nanmin(d))
    if span_T <= 0 or span_S <= 0:
        return np.inf
    return k * span_S / span_T


def _fit_bounds(T, d, mode=None, bounds=None):
    """Return (lo, hi) parameter bounds for [dH, dS, m1, b1, m2, b2].

    ``bounds`` is a BOUNDS_DEFAULTS-style override dict (dH / dS boxes,
    slope_cap_k); ``mode`` overrides ``bounds['mode']``. Baseline modes:
      "physical"  : |m1|, |m2| <= slope_cap_k × (ΔS/ΔT of the data)
      "loose"     : m1 >= 0, m2 free
      "positive"  : m1 >= 0, m2 >= 0
      "unbounded" : both slopes free
    Intercepts are always free.
    """
    b = _bounds_settings({**(bounds or {}), **({"mode": mode} if mode else {})})
    (dH_lo, dH_hi), (dS_lo, dS_hi) = b["dH"], b["dS"]
    inf = np.inf
    if b["mode"] == "physical":
        cap = _slope_cap(T, d, k=b["slope_cap_k"])
        m1, m2 = (-cap, cap), (-cap, cap)
    elif b["mode"] == "loose":
        m1, m2 = (0.0, inf), (-inf, inf)
    elif b["mode"] == "positive":
        m1, m2 = (0.0, inf), (0.0, inf)
    else:  # "unbounded"
        m1, m2 = (-inf, inf), (-inf, inf)
    return ([dH_lo, dS_lo, m1[0], -inf, m2[0], -inf],
            [dH_hi, dS_hi, m1[1],  inf, m2[1],  inf])


def dH_seed_from_fwhm(T, S, struct_type="heterodimer", folding=True,
                      smooth_window=21, smooth_polyorder=3):
    """Estimate ΔH (kcal/mol) from the FWHM of dS/dT — no baseline required.

    For a two-state transition centred at Tm, dθ/dT is peaked and the peak
    FWHM is a fixed multiple of R·Tm²/|ΔH|. The formula
    ``|ΔH| ≈ 3.526 R Tm² / FWHM``
    applies to both mono- and bimolecular two-state melts. Returns None if a
    peak cannot be located. ``folding=True`` returns a negative value
    (RNAmelt's helix-formation sign convention).

    A sloping baseline broadens the observed dS/dT peak, so this estimator
    systematically UNDER-predicts |ΔH| by a factor typically 1.3–1.8. It is
    intended as a robust starting point for the non-linear fit, not as a
    final result.
    """
    from scipy.signal import savgol_filter
    T = np.asarray(T, dtype=float); S = np.asarray(S, dtype=float)
    n = len(T)
    if n < 11:
        return None
    dt = float(np.median(np.diff(T)))
    w = min(smooth_window, n if n % 2 else n - 1)
    w = w if w % 2 else w - 1
    p = min(smooth_polyorder, w - 1)
    dS = savgol_filter(S, w, p, deriv=1) / dt

    up = float(S[-max(3, n // 10):].mean()) > float(S[:max(3, n // 10)].mean())
    if not up:
        dS = -dS
    ipk = int(np.argmax(dS))
    peak = float(dS[ipk])
    if peak <= 0:
        return None

    half = peak / 2.0
    left, right = ipk, ipk
    while left > 0 and dS[left] > half:
        left -= 1
    while right < n - 1 and dS[right] > half:
        right += 1

    def _cross(i0, i1):
        y0, y1 = dS[i0], dS[i1]
        return T[i0] if y0 == y1 else T[i0] + (half - y0) / (y1 - y0) * (T[i1] - T[i0])

    T_lo = _cross(left, left + 1) if left + 1 <= ipk else T[left]
    T_hi = _cross(right - 1, right) if right - 1 >= ipk else T[right]
    fwhm = float(T_hi - T_lo)
    if fwhm <= 0:
        return None

    Tm_K = float(T[ipk]) + 273.15
    dH_mag = 3.526 * constants.R * Tm_K * Tm_K / fwhm
    dH = -dH_mag if folding else dH_mag
    return float(dH)


def _intercept_seed(d, lin_init, head=True):
    """(0, mean of the first / last lin_init points) as a baseline seed."""
    d = np.asarray(d, dtype=float)
    n = min(int(lin_init), d.size)
    part = d[:n] if head else d[-n:]
    return (0.0, float(np.mean(part)))


def fit_full_function(
    T,
    d,
    c0=1e-6,
    dH_init=None,
    dS_init=None,
    b1_init=None,
    b2_init=None,
    lin_init=None,
    struct_type="heterodimer",
    bounds_mode=None,
    solver=None,
    return_converged=False,
    bounds=None,
):
    """Nonlinear two-state fit of one curve.

    Unset starting values take FIT_INIT_DEFAULTS (dH/dS fallbacks, lin_init),
    unset bounds take BOUNDS_DEFAULTS; ``bounds_mode`` overrides
    ``bounds['mode']``.

    Returns ``(dG_37, dH, dS, Tm, y_Tm, base_b, base_ub, base_med)``; with
    ``return_converged=True`` a trailing ``converged`` bool is appended.
    ``Tm`` is the model melting temperature from the fitted ΔH/ΔS and c0
    (θ = 1/2), and ``y_Tm`` the fitted signal there (the mean baseline).
    Hitting ``max_nfev`` emits a ConvergenceWarning and returns the best
    parameters found (``converged=False``) instead of raising.
    """
    xdata = np.array(T)
    ydata = np.array(d)
    init = _fit_init_settings({"dH_init": dH_init, "dS_init": dS_init,
                               "lin_init": lin_init})
    dH0 = init["dH_init"] if init["dH_init"] is not None else init["dH_fallback"]
    dS0 = init["dS_init"] if init["dS_init"] is not None else init["dS_fallback"]

    if b1_init is None:
        b1_init = _intercept_seed(ydata, init["lin_init"], head=True)
    if b2_init is None:
        b2_init = _intercept_seed(ydata, init["lin_init"], head=False)

    x_init = np.array([dH0, dS0, *b1_init, *b2_init])  # dH, dS, m1, b1, m2, b2

    fit_bounds = _fit_bounds(xdata, ydata, bounds_mode, bounds)
    # scipy refuses x0 that violates the bounds — clip aggressive seeds
    # (e.g. an FWHM-derived dH near the lower bound) into the feasible box.
    x_init = np.clip(x_init, fit_bounds[0], fit_bounds[1])

    res_lsq = least_squares(functions.full_function_res,
                            x_init,
                            bounds = fit_bounds,
                            x_scale = _x_scale(solver),
                            args = (xdata, ydata),
                            kwargs = { 'c0': c0,
                                       'struct_type': struct_type},
                            **_least_squares_kwargs(solver))

    converged = _check_fit_status(res_lsq)

    dH, dS, m_b, b_b, m_ub, b_ub = res_lsq.x
    dG_37 = dH - (37. - constants.T0) * dS

    # median of base lines
    m_med, b_med = (m_b + m_ub) / 2, (b_b + b_ub) / 2

    T_m = functions.tm_from_therm(dH, dS, c0, struct_type)
    y_Tm = functions.linear(T_m, m_med, b_med)

    out = (dG_37, dH, dS, T_m, y_Tm, (m_b, b_b), (m_ub, b_ub), (m_med, b_med))
    return (*out, converged) if return_converged else out

def fit_full_function_multi(
    T,
    ds,
    cs=None,
    c0=1e-6,
    dH_init=None,
    dS_init=None,
    b1_inits=None,
    b2_inits=None,
    lin_init=None,
    struct_type="heterodimer",
    bounds_mode=None,
    solver=None,
    return_converged=False,
    bounds=None,
):
    """Joint two-state fit of several curves sharing ΔH / ΔS.

    Returns ``(dG_37, dH, dS, Tms, y_Tms, base_lo, base_hi, base_med)``
    (lists per curve); ``return_converged=True`` appends a ``converged``
    bool. Tms are model melting temperatures from ΔH/ΔS and each curve's
    concentration; see fit_full_function for the max_nfev behaviour.
    """

    xdata = T
    ydata = ds
    n = len(ds)
    # bounds / lin_init / baseline seeds may be given per curve (sequence)
    # or once for all curves. The shared dH / dS box comes from curve 0.
    per = lambda v: list(v) if isinstance(v, (list, tuple)) else [v] * n
    bounds_list, lin_list = per(bounds), per(lin_init)
    b1_list = list(b1_inits) if b1_inits is not None else [None] * n
    b2_list = list(b2_inits) if b2_inits is not None else [None] * n
    if not len(bounds_list) == len(lin_list) == len(b1_list) == len(b2_list) == n:
        raise ValueError("per-curve bounds / lin_init / baseline seeds must "
                         f"have one entry per curve ({n})")

    init = _fit_init_settings({"dH_init": dH_init, "dS_init": dS_init})
    dH0 = init["dH_init"] if init["dH_init"] is not None else init["dH_fallback"]
    dS0 = init["dS_init"] if init["dS_init"] is not None else init["dS_fallback"]

    x_inits = [dH0, dS0]
    for i in range(n):
        li = _fit_init_settings({"lin_init": lin_list[i]})["lin_init"]
        b1 = b1_list[i] if b1_list[i] is not None else _intercept_seed(ds[i], li, head=True)
        b2 = b2_list[i] if b2_list[i] is not None else _intercept_seed(ds[i], li, head=False)
        x_inits.extend([*b1, *b2])
    x_init = np.array(x_inits)

    # shared dH/dS box, then per-curve baseline bounds from the same helper
    lo0, hi0 = _fit_bounds(np.asarray(T[0]), np.asarray(ds[0]), bounds_mode, bounds_list[0])
    bounds_lo, bounds_hi = lo0[:2], hi0[:2]
    for i in range(n):
        lo_i, hi_i = _fit_bounds(np.asarray(T[i]), np.asarray(ds[i]), bounds_mode, bounds_list[i])
        bounds_lo.extend(lo_i[2:])
        bounds_hi.extend(hi_i[2:])
    fit_bounds = (bounds_lo, bounds_hi)
    x_init = np.clip(x_init, fit_bounds[0], fit_bounds[1])

    res_lsq = least_squares(functions.full_function_multi_res,
                            x_init,
                            bounds = fit_bounds,
                            x_scale = _x_scale(solver, n_curves=len(ds)),
                            args = (xdata, ydata),
                            kwargs = { 'cs': cs,
                                       'c0': c0,
                                       'struct_type': struct_type},
                            **_least_squares_kwargs(solver))

    converged = _check_fit_status(res_lsq)

    dH, dS = res_lsq.x[0], res_lsq.x[1]
    dG_37 = dH - (37 - constants.T0) * dS

    curve_cs = list(cs) if cs is not None and len(cs) else [c0] * len(ds)
    baselines_lo = []
    baselines_me = []
    baselines_hi = []
    T_ms         = []
    y_dats       = []
    for j, i in enumerate(range(2, len(res_lsq.x), 4)):
        m_b, b_b    = res_lsq.x[i], res_lsq.x[i + 1]
        m_ub, b_ub  = res_lsq.x[i + 2], res_lsq.x[i + 3]
        m_m, b_m    = (m_b + m_ub) / 2, (b_b + b_ub) / 2
        T_m         = functions.tm_from_therm(dH, dS, curve_cs[j], struct_type)
        baselines_lo.append( (m_b, b_b) )
        baselines_hi.append( (m_ub, b_ub) )
        baselines_me.append( (m_m, b_m) )
        T_ms.append(T_m)
        y_dats.append(functions.linear(T_m, m_m, b_m))

    out = (dG_37, dH, dS, T_ms, y_dats, baselines_lo, baselines_hi, baselines_me)
    return (*out, converged) if return_converged else out

