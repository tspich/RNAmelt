import numpy as np
from scipy.optimize import least_squares

from rnamelt import constants

def linear(x, a, b):
    return a * x + b

def linear_res(x, T, y):
    return linear(T, *x) - y

def fit_linear(T, d, minT, maxT):
    # Locate the window [start, stop] with defensive clamps. When a caller
    # supplies a boundary just above/below the scan (e.g. a JS toFixed(1)
    # rounding a T[-1]=89.6999969 up to 89.7) the naive `next()` raises
    # StopIteration with an empty message and blows up the whole pipeline
    # silently. Clamp gracefully instead.
    start = next((i for i, v in enumerate(T) if v >= minT), None)
    if start is None:
        raise ValueError(
            f"fit_linear: minT={minT} above scan max T={T[-1]} — no data in window"
        )
    stop = next((i for i, v in enumerate(T) if v >= maxT), len(T) - 1)
    if stop - start + 1 < 2:
        raise ValueError(
            f"fit_linear: window [{minT}, {maxT}] holds fewer than 2 points "
            f"within T=[{T[0]}, {T[-1]}]"
        )

    xdata = np.array(T[start:stop + 1])
    ydata = np.array(d[start:stop + 1])

    res_lsq = least_squares(linear_res,
                            [1, 1],
                            args=(xdata, ydata))

    return (res_lsq.x[0], res_lsq.x[1])

def intersect_lin(m, b, xs, ys, edge_skip=10):
    """First crossing (nearest the window centre if several) of the line
    y = m·x + b with the data (xs, ys), by linear interpolation. The first
    and last ``edge_skip`` points are ignored."""
    diffs = [y - linear(x, m, b) for x, y in zip(xs, ys)]
    hits = [i for i in range(edge_skip, len(ys) - edge_skip - 1)
            if diffs[i] == 0 or diffs[i] * diffs[i + 1] < 0]
    if not hits:
        lo = xs[min(edge_skip, len(xs) - 1)]
        hi = xs[max(-edge_skip - 1, -len(xs))]
        raise ValueError(
            f"baseline does not intersect data in window [{lo:.1f}, {hi:.1f}]"
        )

    i = hits[0]
    if len(hits) > 1:
        midT = 0.5 * (xs[0] + xs[-1])
        i = min(hits, key=lambda k: abs(xs[k] - midT))

    x1, y1 = xs[i], ys[i]
    x2, y2 = xs[i + 1], ys[i + 1]
    m_int = (y2 - y1) / (x2 - x1)
    b_int = y1 - m_int * x1
    x = (b_int - b) / (m - m_int)
    return x, linear(x, m, b)

def full_function(T, dH, dS, m1, b1, m2, b2, c0 = 1e-6, struct_type='heterodimer'):
    if struct_type == 'heterodimer':
        theta = theta_from_therm(T, dH, dS, c0)
    elif struct_type == 'homodimer':
        theta = theta_from_therm_homo(T, dH, dS, c0)
    elif struct_type == 'monomer':
        theta = theta_from_therm_mono(T, dH, dS)
    else:
        raise ValueError(
            f"unknown struct_type {struct_type!r}; "
            f"valid: 'heterodimer', 'homodimer', 'monomer'"
        )
    return theta * (m1 * T + b1) + (1 - theta) * (m2 * T + b2)

def full_function_res(x, T, y, c0 = 1e-6, struct_type = 'heterodimer'):
    dH, dS, m1, b1, m2, b2 = x
    return y - full_function(T, dH, dS, m1, b1, m2, b2, c0, struct_type)

def full_function_multi(temps, *args, **kwargs):
    default_c0 = 1e-6
    default_struct_type = 'heterodimer'
    # prepare variables and constants
    f = []
    m1s = []
    b1s = []
    m2s = []
    b2s = []

    c0 = kwargs.get("c0", default_c0)
    # just in case, c0 is None
    if not c0:
        c0 = default_c0
    # collect arguments
    dH = args[0][0]
    dS = args[0][1]

    for i in range(2, len(args[0]), 4):
        m1s.append(args[0][i])
        b1s.append(args[0][i + 1])
        m2s.append(args[0][i + 2])
        b2s.append(args[0][i + 3])

    # create default concentrations if not specified differently
    default_cs = [ c0 for _ in range(len(m1s)) ]
    cs = kwargs.get("cs", default_cs)
    if not cs:
        cs = default_cs

    struct_type = kwargs.get('struct_type', default_struct_type)
    # compose result function fit
    for i, c in enumerate(cs):
        if struct_type == 'heterodimer':
            theta = theta_from_therm(temps[i], dH, dS, c)
        elif struct_type == 'homodimer':
            theta = theta_from_therm_homo(temps[i], dH, dS, c)
        elif struct_type == 'monomer':
            theta = theta_from_therm_mono(temps[i], dH, dS)
        else:
            raise ValueError(
                f"unknown struct_type {struct_type!r}; "
                f"valid: 'heterodimer', 'homodimer', 'monomer'"
            )

        f.append(theta * (m1s[i] * temps[i] + b1s[i]) + (1 - theta) * (m2s[i] * temps[i] + b2s[i]))

    return f

def full_function_multi_res(x, temps, ys, **kwargs):
    """Concatenated per-curve residuals (data − model) of the joint fit."""
    fits = full_function_multi(temps, x, **kwargs)
    return np.concatenate([
        np.asarray(y, dtype=float).ravel() - np.asarray(f, dtype=float).ravel()
        for y, f in zip(ys, fits)
    ])

# Equilibrium constant of folding at the melting temperature (θ = 1/2),
# in units of 1/c0 for the bimolecular cases:
#   heterodimer  A + B ⇌ AB : K = 2θ / (c0 (1-θ)²)  →  K(Tm) = 4 / c0
#   homodimer    2A ⇌ A₂    : K = θ / (2 c0 (1-θ)²) →  K(Tm) = 1 / c0
#   monomer      A ⇌ A*     : K = θ / (1-θ)         →  K(Tm) = 1
_K_AT_TM_FACTOR = {"heterodimer": 4.0, "homodimer": 1.0}


def tm_from_therm(dH, dS, c0=1e-6, struct_type='heterodimer'):
    """Model melting temperature (°C) where the folded fraction θ is 1/2.

    dH in kcal/mol, dS in kcal/(mol·K), c0 = total strand concentration (M).
    Solves ΔH − Tm·ΔS = −R·Tm·ln K(Tm) for Tm.
    """
    if struct_type == 'monomer':
        denom = dS
    elif struct_type in _K_AT_TM_FACTOR:
        denom = dS + constants.R * np.log(c0 / _K_AT_TM_FACTOR[struct_type])
    else:
        raise ValueError(
            f"unknown struct_type {struct_type!r}; "
            f"valid: 'heterodimer', 'homodimer', 'monomer'"
        )
    return dH / denom + constants.T0


def theta_from_therm(T, dH, dS, c0 = 1e-6):
    tt    = T - constants.T0
    dG    = dH - tt * dS
    term = np.sqrt(2 * c0 * np.exp(- dG / (constants.R * tt)) + 1)
    return (term - 1) / (term + 1)

def theta_from_therm_homo(T, dH, dS, c0 = 1e-6):
    tt    = T - constants.T0
    dG    = dH - tt * dS
    term = np.sqrt(8 * c0 * np.exp(- dG / (constants.R * tt)) + 1)
    return (term - 1) / (term + 1)

def theta_from_therm_mono(T, dH, dS):
    tt    = T - constants.T0
    dG    = dH - tt * dS
    return np.exp(- dG / (constants.R * tt))/(1+np.exp(- dG / (constants.R * tt)))



