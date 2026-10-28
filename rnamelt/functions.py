import numpy as np
from scipy.optimize import least_squares

from rnamelt import constants

def linear(x, a, b):
    return a * x + b

def linear_res(x, T, y):
    return linear(T, *x) - y

def fit_linear(T, d, minT, maxT):
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
    theta = folded_fraction(T, dH, dS, c0, struct_type)
    return theta * (m1 * T + b1) + (1 - theta) * (m2 * T + b2)

def full_function_res(x, T, y, c0 = 1e-6, struct_type = 'heterodimer'):
    dH, dS, m1, b1, m2, b2 = x
    return y - full_function(T, dH, dS, m1, b1, m2, b2, c0, struct_type)

def full_function_multi(temps, *args, **kwargs):
    default_c0 = 1e-6
    default_struct_type = 'heterodimer'
    f = []
    m1s = []
    b1s = []
    m2s = []
    b2s = []

    c0 = kwargs.get("c0", default_c0)
    if not c0:
        c0 = default_c0
    dH = args[0][0]
    dS = args[0][1]

    for i in range(2, len(args[0]), 4):
        m1s.append(args[0][i])
        b1s.append(args[0][i + 1])
        m2s.append(args[0][i + 2])
        b2s.append(args[0][i + 3])

    default_cs = [ c0 for _ in range(len(m1s)) ]
    cs = kwargs.get("cs", default_cs)
    if not cs:
        cs = default_cs

    struct_type = kwargs.get('struct_type', default_struct_type)
    for i, c in enumerate(cs):
        theta = folded_fraction(temps[i], dH, dS, c, struct_type)
        f.append(theta * (m1s[i] * temps[i] + b1s[i]) + (1 - theta) * (m2s[i] * temps[i] + b2s[i]))

    return f

def full_function_multi_res(x, temps, ys, **kwargs):
    """Concatenated per-curve residuals (data − model) of the joint fit."""
    fits = full_function_multi(temps, x, **kwargs)
    return np.concatenate([
        np.asarray(y, dtype=float).ravel() - np.asarray(f, dtype=float).ravel()
        for y, f in zip(ys, fits)
    ])


def _conc(c0, struct_type):
    """The concentration in the form each model uses: (c_A, c_B) in M for a
    heterodimer, the total c0 for a homodimer, None for a monomer."""
    pair = np.size(c0) == 2
    if struct_type == 'heterodimer':
        return (float(c0[0]), float(c0[1])) if pair else (0.5 * c0, 0.5 * c0)
    if pair:
        raise ValueError(f"unequal strand concentrations only apply to "
                         f"heterodimers, not struct_type={struct_type!r}")
    if struct_type == 'homodimer':
        return c0
    if struct_type == 'monomer':
        return None
    raise ValueError(f"unknown struct_type {struct_type!r}; "
                     f"valid: 'heterodimer', 'homodimer', 'monomer'")


def c_eff(c0, struct_type='heterodimer'):
    """Effective concentration (M) with K(Tm) = 1/c_eff, i.e.
    Tm = ΔH / (ΔS + R·ln c_eff); None for a monomer."""
    c = _conc(c0, struct_type)
    if struct_type == 'heterodimer':
        a, b = c
        return max(a, b) - 0.5 * min(a, b)
    return c


def folded_fraction(T, dH, dS, c0=1e-6, struct_type='heterodimer'):
    """Folded fraction θ(T) of the two-state model (T in °C, ΔH in kcal/mol,
    ΔS in kcal/(mol·K)); written in 1/K so it stays finite at both ends."""
    c = _conc(c0, struct_type)
    tt = T - constants.T0
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        inv_K = np.exp((dH - tt * dS) / (constants.R * tt))
        if struct_type == 'monomer':
            return 1.0 / (1.0 + inv_K)
        if struct_type == 'homodimer':
            # θ = (s − 1)/(s + 1) with s = √(1 + 8 c0 K), written in r = 1/s
            r = 1.0 / np.sqrt(1.0 + 8.0 * c / inv_K)
            return (1.0 - r) / (1.0 + r)
        # [AB] = 2 c_A c_B / (S + √(S² − 4 c_A c_B)), S = c_A + c_B + 1/K,
        # with the discriminant written without cancellation
        a, b = c
        disc = (a - b) ** 2 + inv_K * (2.0 * (a + b) + inv_K)
        AB = 2.0 * a * b / (a + b + inv_K + np.sqrt(disc))
        return AB / min(a, b)


def ln_k_from_theta(theta, c0, struct_type='heterodimer'):
    """ln K of the folding equilibrium from the folded fraction θ
    (the inverse of folded_fraction at one temperature)."""
    c = _conc(c0, struct_type)
    if struct_type == 'monomer':
        return np.log(theta / (1.0 - theta))
    if struct_type == 'homodimer':
        return np.log(theta / (2.0 * c * (1.0 - theta) ** 2))
    a, b = c
    m = min(a, b)
    return np.log(theta * m / ((a - theta * m) * (b - theta * m)))


def _tm_denominator(dS, c0, struct_type):
    """ΔS + R·ln c_eff (ΔS for a monomer): Tm_K = ΔH / this."""
    if struct_type == 'monomer':
        return dS
    return dS + constants.R * np.log(c_eff(c0, struct_type))


def tm_from_therm(dH, dS, c0=1e-6, struct_type='heterodimer'):
    """Model melting temperature (°C) where the folded fraction θ is 1/2.

    dH in kcal/mol, dS in kcal/(mol·K), c0 = total strand concentration (M)
    or (c_A, c_B) per strand (M). Solves ΔH − Tm·ΔS = −R·Tm·ln K(Tm) for Tm.
    """
    return dH / _tm_denominator(dS, c0, struct_type) + constants.T0


def tm_error(dH: float, dS: float, cov_HS, c0: float = 1e-6,
             struct_type: str = 'heterodimer') -> float:
    """Standard error (°C = K) of the model Tm from the 2×2 covariance of
    (ΔH [kcal/mol], ΔS [kcal/(mol·K)]), by the delta method on
    Tm = ΔH / D with D = ΔS + R·ln c_eff."""
    D = _tm_denominator(dS, c0, struct_type)
    g = np.array([1.0 / D, -dH / D**2])
    return float(np.sqrt(max(g @ np.asarray(cov_HS, dtype=float) @ g, 0.0)))


def thermo_cov_from_regression(slope: float, intercept: float, cov_ab) -> np.ndarray:
    """2×2 covariance of (ΔH [kcal/mol], ΔS [kcal/(mol·K)]) from the covariance
    of slope a and intercept b of 1/Tm = a·ln(C_T/f) + b (delta method on
    ΔH = R/a, ΔS = R·b/a)."""
    a, b, R = float(slope), float(intercept), constants.R
    J = np.array([[-R / a**2,      0.0],
                  [-R * b / a**2,  R / a]])
    return J @ np.asarray(cov_ab, dtype=float) @ J.T


def thermo_errors(cov_HS, T_C: float = 37.0) -> tuple[float, float, float]:
    """Standard errors (ΔH, ΔS, ΔG at T_C °C) from the 2×2 covariance of
    (ΔH [kcal/mol], ΔS [kcal/(mol·K)]). The ΔG error keeps the ΔH–ΔS
    covariance: var(ΔG) = var(ΔH) + T²·var(ΔS) − 2T·cov(ΔH, ΔS)."""
    cov = np.asarray(cov_HS, dtype=float)
    T_K = T_C - constants.T0
    var_G = cov[0, 0] + T_K**2 * cov[1, 1] - 2 * T_K * cov[0, 1]
    return (float(np.sqrt(max(cov[0, 0], 0.0))),
            float(np.sqrt(max(cov[1, 1], 0.0))),
            float(np.sqrt(max(var_G, 0.0))))
