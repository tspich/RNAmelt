import numpy as np
from scipy.optimize import least_squares
from scipy import stats

from rnamelt import functions, constants

# Defaults for the nonlinear full-function fits. Most keys forward to
# scipy.optimize.least_squares; `residuals_method` is multi-fit-specific
# (chooses how per-curve residuals are combined — see functions.res_diffs)
# and is silently ignored by the single-curve fit. Tuned for the
# in-browser (Pyodide) case where ~10k function evaluations is the
# practical ceiling; CLI / API users can raise via the `solver=` argument.
SOLVER_DEFAULTS = {
    "max_nfev":         10000,
    "ftol":             1e-8,
    "gtol":             1e-8,
    "xtol":             1e-8,
    "method":           "trf",
    "loss":             "linear",
    "f_scale":          1.0,
    "jac":              "2-point",
    "verbose":          0,
    "residuals_method": "square",  # multi-fit only — see functions.res_diffs
}

# Subset of SOLVER_DEFAULTS that is forwarded as-is to least_squares().
# Anything else in the bundle is consumed locally (see _solver_extra).
_SCIPY_LEAST_SQUARES_KEYS = (
    "max_nfev", "ftol", "gtol", "xtol",
    "method", "loss", "f_scale", "jac", "verbose",
)


def _validate_solver(overrides):
    """Raise on unknown solver keys."""
    if not overrides:
        return
    for k, v in overrides.items():
        if k not in SOLVER_DEFAULTS:
            raise ValueError(
                f"unknown solver option {k!r}; valid keys: "
                f"{sorted(SOLVER_DEFAULTS)}"
            )


def _solver_kwargs(overrides):
    """Return scipy.optimize.least_squares kwargs (scipy-only subset)."""
    _validate_solver(overrides)
    out = {k: SOLVER_DEFAULTS[k] for k in _SCIPY_LEAST_SQUARES_KEYS}
    if overrides:
        for k, v in overrides.items():
            if v is None or k not in _SCIPY_LEAST_SQUARES_KEYS:
                continue
            out[k] = v
    return out


def _solver_extra(overrides, key):
    """Get a non-scipy solver-bundle value with default fallback."""
    if overrides and overrides.get(key) is not None:
        _validate_solver(overrides)
        return overrides[key]
    return SOLVER_DEFAULTS[key]


# Defaults for the van't Hoff linearisation (1/T vs ln K). `border` is
# the fraction-folded cutoff that gates which points enter the
# regression — points with θ outside [border, 1-border] are excluded.
# `t1_min` / `t1_max` clip the linearised x-axis (in T_scale / (T_C - T0)
# units; -1 = no clip). `T_scale` is a numerical-conditioning factor and
# rarely needs changing.
VH_DEFAULTS = {
    "border":  0.15,
    "t1_min":  -1,
    "t1_max":  -1,
    "T_scale": 1000.0,
}


def _vh_kwargs(overrides):
    """Merge user-supplied overrides with VH_DEFAULTS; return kwargs for vantHoff()."""
    out = dict(VH_DEFAULTS)
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            if k not in VH_DEFAULTS:
                raise ValueError(
                    f"unknown vh option {k!r}; valid keys: "
                    f"{sorted(VH_DEFAULTS)}"
                )
            out[k] = v
    return out


# Defaults for the full-function fit initial guesses. `dH_init` / `dS_init`
# left as None mean "let the orchestrator auto-seed" (from van't Hoff
# linearisation in single / concentration modes, or a hardcoded fallback
# in multi mode). A non-None value here overrides that auto-seeding.
# `lin_init` is the number of leading / trailing data points averaged
# for the baseline-intercept seed. `b1_init` / `b2_init` are explicit
# (slope, intercept) tuples for the folded / unfolded baselines;
# multi-fit and concentration modes ignore them and auto-compute.
FIT_INIT_DEFAULTS = {
    "dH_init":  None,
    "dS_init":  None,
    "lin_init": 10,
    "b1_init":  None,
    "b2_init":  None,
}


def _fit_init_kwargs(overrides):
    """Validate fit_init overrides and merge with FIT_INIT_DEFAULTS."""
    out = dict(FIT_INIT_DEFAULTS)
    if overrides:
        for k, v in overrides.items():
            if k not in FIT_INIT_DEFAULTS:
                raise ValueError(
                    f"unknown fit_init option {k!r}; valid keys: "
                    f"{sorted(FIT_INIT_DEFAULTS)}"
                )
            out[k] = v
    return out


def T_m_ds_raw(temperatures,
               ydata,
               baseline_bound_minT = -1,
               baseline_bound_maxT = 20.0,
               baseline_unbound_minT = 70,
               baseline_unbound_maxT = -1,
               debug = False):
    T = np.array(temperatures)

    # set bounds first
    bl_b_minT   = T[0] if baseline_bound_minT == -1 else baseline_bound_minT
    bl_ub_maxT  = T[-1] if baseline_unbound_maxT == -1 else baseline_unbound_maxT

    bl_b_maxT   = T[-1] if baseline_bound_maxT not in T else baseline_bound_maxT
    bl_ub_minT  = T[-1] if baseline_unbound_minT not in T else baseline_unbound_minT

    if debug:
        print(f"bounds_bound: [{bl_b_minT}, {bl_b_maxT}]")
        print(f"bounds_unbound: [{bl_ub_minT}, {bl_ub_maxT}]")

    # create upper base line (bound state)
    if bl_b_maxT <= bl_b_minT:
        m_bound = 0
        b_bound = ydata[0]
    else:
        m_bound, b_bound = functions.fit_linear(T, ydata, bl_b_minT, bl_b_maxT)

    # lower base line
    if bl_ub_minT >= bl_ub_maxT :
        m_unbound = 0
        b_unbound = ydata[-1]
        if debug:
            print('lower base line: ', m_unbound, b_unbound)
    else:
        m_unbound, b_unbound = functions.fit_linear(T, ydata, bl_ub_minT, bl_ub_maxT)


    # median of base lines
    m_med, b_med = (m_bound + m_unbound) / 2, (b_bound + b_unbound) / 2

    T_m, y_dat = functions.intersect_lin(m_med, b_med, T, ydata)

    return T_m, y_dat, (m_bound, b_bound), (m_unbound, b_unbound), (m_med, b_med)

def vantHoff(T,
             signal,
             m_bound,
             b_bound,
             m_unbound,
             b_unbound,
             c0,
             border = 0.1,
             T_scale = 1000.,
             t1_min = -1,
             t1_max = -1,
             structType = 'heterodimer'):
    if structType == 'heterodimer':
        # normalize signal to obtain fraction of folded
        f_folded = [
            (dd - functions.linear(T[i], m_unbound, b_unbound))
            / (
                functions.linear(T[i], m_bound, b_bound)
                - functions.linear(T[i], m_unbound, b_unbound)
            )
            for i, dd in enumerate(signal)
        ]

        K = [
            np.log(2 * ff / (c0 * ((1.0 - ff) ** 2)))
            if ff <= (1 - border) and ff >= border
            else None
            for ff in f_folded
        ]

        t1 = [ T_scale / (t - constants.T0) for t in T ]


        lnK = []
        tt  = []

        # compile actual data without None values
        for i in range(0, len(K)):
            if K[i] != None:
                tt.append(t1[i])
                lnK.append(K[i])

        if t1_min == -1:
            t1_min = len(t1)
        else:
            t1_min = next(i for i,v in enumerate(tt) if v < t1_min)

        if t1_max == -1:
            t1_max = 0
        else:
            t1_max = next(i for i,v in enumerate(tt) if v <= t1_max)

        xdata = np.array(tt[t1_max:t1_min])
        ydata = np.array(lnK[t1_max:t1_min])

        res_lsq = least_squares(functions.linear_res,
                                [1, 1],
                                args=(xdata, ydata))

        t_m = (np.log(4 / c0) - res_lsq.x[1]) / res_lsq.x[0]
        T_m = T_scale / t_m + constants.T0
        dH  = -res_lsq.x[0] * constants.R * T_scale
        dG_Tm = constants.R * (T_m - constants.T0) * np.log(4 / c0) + dH

        dS    = dH / (T_m - constants.T0) + constants.R * np.log(4 / c0)
        dG_37 = dH - (37. - constants.T0) * dS

        return T_m, dG_37, dH, dS, t1, K, xdata, ydata, res_lsq.x

    elif structType == 'homodimer':
        # normalize signal to obtain fraction of folded
        f_folded  = [ (dd - functions.linear(T[i], m_unbound, b_unbound)) / (functions.linear(T[i], m_bound, b_bound) - functions.linear(T[i], m_unbound, b_unbound)) for i, dd in enumerate(signal) ]

        K         = [ np.log(2 * ff / (c0 * ((1. - ff) ** 2))) if ff <= (1 - border) and ff >= border else None for ff in f_folded ]
        t1        = [ T_scale / (t - constants.T0) for t in T ]


        lnK = []
        tt  = []

        # compile actual data without None values
        for i in range(0, len(K)):
            if K[i] != None:
                tt.append(t1[i])
                lnK.append(K[i])

        if t1_min == -1:
            t1_min = len(t1)
        else:
            t1_min = next(i for i,v in enumerate(tt) if v < t1_min)

        if t1_max == -1:
            t1_max = 0
        else:
            t1_max = next(i for i,v in enumerate(tt) if v <= t1_max)

        xdata = np.array(tt[t1_max:t1_min])
        ydata = np.array(lnK[t1_max:t1_min])

        res_lsq = least_squares(functions.linear_res,
                                [1, 1],
                                args=(xdata, ydata))

        t_m = (np.log(1 / c0) - res_lsq.x[1]) / res_lsq.x[0]
        T_m = T_scale / t_m + constants.T0
        dH  = -res_lsq.x[0] * constants.R * T_scale
        dG_Tm = constants.R * (T_m - constants.T0) * np.log(1 / c0) + dH

        dS    = dH / (T_m - constants.T0) + constants.R * np.log(1 / c0)
        dG_37 = dH - (37. - constants.T0) * dS

        return T_m, dG_37, dH, dS, t1, K, xdata, ydata, res_lsq.x


    elif structType == 'monomer':
        raise ValueError("Van't Hoff analysis not possible for monomere!")

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

    slope, intercept, r, _, _ = stats.linregress(ln_ct, inv_tm)

    if slope == 0:
        raise ValueError("Regression slope is zero; cannot derive ΔH")

    dH = constants.R / slope
    dS = intercept * dH
    dG_37 = dH - (37.0 - constants.T0) * dS

    return {
        "dH":             float(dH),
        "dS":             float(dS),
        "dG_37":          float(dG_37),
        "r_squared":      float(r ** 2),
        "slope":          float(slope),
        "intercept":      float(intercept),
        "ln_ct":          ln_ct,
        "inv_tm":         inv_tm,
        "tm_values":      tm,
        "concentrations": ct,
        "self_complementary": bool(self_complementary),
    }

def fit_full_function(
    T,
    d,
    c0=1e-6,
    dH_init=-100,
    dS_init=-0.2,
    b1_init=None,
    b2_init=None,
    lin_init=10,
    #max_v=np.inf,
    structType="heterodimer",
    solver=None,
):
    xdata = np.array(T)
    ydata = np.array(d)

    # use the first and last lin_init data values
    # for initializing the linear intercepts of L_1 and L_2
    if b1_init == None:
      b1_init = (0, sum(d[0:lin_init]) / lin_init)

    if b2_init == None:
      b2_init = (0, sum(d[-lin_init:]) / lin_init)

    x_init = np.array([ dH_init, dS_init, *b1_init, *b2_init]) # dH, dS, m1, b1, m2, b2

    bounds = ([ -200, -5,       0, -np.inf, 0,      -np.inf ],
              [0,      0,  np.inf,  np.inf, np.inf,  np.inf ])

    scales  = [1., 0.01, 1., 100., 1., 100.]

    sk = _solver_kwargs(solver)
    res_lsq = least_squares(functions.full_function_res,
                            x_init,
                            bounds = bounds,
                            x_scale = scales,
                            args = (xdata, ydata),
                            kwargs = { 'c0': c0,
                                       'structType': structType},
                            **sk)

    if res_lsq.success>0:
        dG_37 = res_lsq.x[0] - (37 - constants.T0) * res_lsq.x[1]
        dH, dS, m_b, b_b, m_ub, b_ub = res_lsq.x
        dG_37 = dH - (37. - constants.T0) * dS

        # median of base lines
        m_med, b_med = (m_b + m_ub) / 2, (b_b + b_ub) / 2

        T_m, y_dat = functions.intersect_lin(m_med, b_med, xdata, ydata)

        return dG_37, dH, dS, T_m, y_dat, (m_b, b_b), (m_ub, b_ub), (m_med, b_med)
    else:
        raise ValueError("The least square didn't converge")

def fit_full_function_multi(
    T,
    ds,
    cs=None,
    c0=1e-6,
    dH_init=-100,
    dS_init=-0.2,
    b1_inits=None,
    b2_inits=None,
    lin_init=10,
    residuals_method="square",
    structType="heterodimer",
    solver=None,
):

    xdata = np.array(T)
    ydata = ds #np.array(d)

    # use the first and last lin_init data values
    # for initializing the linear intercepts of L_1 and L_2
    if b1_inits == None:
      b1_inits = [ (0, sum(dd[0:lin_init]) / lin_init) for dd in ds ]

    if b2_inits == None:
      b2_inits = [ (0, sum(dd[-lin_init:]) / lin_init) for dd in ds ]

    x_inits = [ dH_init, dS_init ]
    for i in range(len(ds)):
        x_inits.extend([*b1_inits[i], *b2_inits[i]])

    x_init = np.array(x_inits)

    bounds_lo = [ -200, -5 ]
    bounds_hi = [ 0,     0 ]
    for i in range(len(ds)):
        bounds_lo.extend([ 0, -np.inf, 0, -np.inf ])
        bounds_hi.extend([ np.inf, np.inf, np.inf, np.inf ])

    bounds = (bounds_lo, bounds_hi)

    scales = [ 1., 0.01 ]

    for i in range(len(ds)):
        scales.extend([1, 100., 1, 100.])

    # `solver["residuals_method"]` overrides the positional kwarg if given.
    if solver and solver.get("residuals_method") is not None:
        residuals_method = solver["residuals_method"]

    if residuals_method not in functions.res_diffs:
        #print("Falling back to linear diff sum")
        res_diff_fun = functions.res_diffs['linear']
    else:
        res_diff_fun = functions.res_diffs[residuals_method]

    sk = _solver_kwargs(solver)
    res_lsq = least_squares(functions.full_function_multi_res,
                            x_init,
                            bounds = bounds,
                            x_scale = scales,
                            args = (xdata, ydata),
                            kwargs = { 'cs': cs,
                                       'c0': c0,
                                       'res_func': res_diff_fun,
                                       'structType': structType},
                            **sk)

    if res_lsq.success>0:
        dH, dS = res_lsq.x[0], res_lsq.x[1]
        dG_37 = dH - (37 - constants.T0) * dS

        baselines_lo = []
        baselines_me = []
        baselines_hi = []
        T_ms         = []
        y_dats       = []
        for j, i in enumerate(range(2, len(res_lsq.x), 4)):
            m_b, b_b    = res_lsq.x[i], res_lsq.x[i + 1]
            m_ub, b_ub  = res_lsq.x[i + 2], res_lsq.x[i + 3]
            m_m, b_m    = (m_b + m_ub) / 2, (b_b + b_ub) / 2
            T_m, y_dat  = functions.intersect_lin(m_m, b_m, xdata, ydata[j])
            baselines_lo.append( (m_b, b_b) )
            baselines_hi.append( (m_ub, b_ub) )
            baselines_me.append( (m_m, b_m) )
            T_ms.append(T_m)
            y_dats.append(y_dat)

        return dG_37, dH, dS, T_ms, y_dats, baselines_lo, baselines_hi, baselines_me
    else:
        raise ValueError("The least square didn't converge")

