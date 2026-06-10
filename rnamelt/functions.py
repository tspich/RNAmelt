from functools import reduce

import numpy as np
from scipy.optimize import least_squares

from rnamelt import constants

def res_diff(vs):
    return vs[0] - vs[1]

def res_diff_square(vs):
    return (vs[0] - vs[1]) * (vs[0] - vs[1])

def res_diff_cubic(vs):
    return (vs[0] - vs[1]) * (vs[0] - vs[1]) * (vs[0] - vs[1])

def res_diff_quadratic(vs):
    return (vs[0] - vs[1]) * (vs[0] - vs[1]) * (vs[0] - vs[1]) * (vs[0] - vs[1])

res_diffs = {
    'linear': res_diff,
    'square': res_diff_square,
    'cubic':  res_diff_cubic,
    'quadratic' : res_diff_quadratic,
}

def linear(x, a, b):
    return a * x + b

def linear_res(x, T, y):
    return linear(T, *x) - y

def fit_linear(T, d, minT, maxT):
    start = next(i for i,v in enumerate(T) if v >= minT) # TS: change > to >= to include first temperature (10.0)
    stop  = next(i for i,v in enumerate(T) if v >= maxT)

    xdata = np.array(T[start:stop + 1])
    ydata = np.array(d[start:stop + 1])

    res_lsq = least_squares(linear_res,
                            [1, 1],
                            args=(xdata, ydata))

    return (res_lsq.x[0], res_lsq.x[1])

def intersect_lin(m, b, xs, ys, border = 10, min_value = 0, max_value = 10000):
    """
    Find the intersection of a linear function with
    slope m and intersect b and a set of X-Y coordinates
    given in lists xs and ys.
    """
    # 1. find point closest to the data
    closest = (min_value, max_value)
    for (i, y) in enumerate(ys[border:-border], start = border):
        v = linear(xs[i], m, b)
        if y < v and ys[i + 1] >= v:
            closest = (i, abs(y - v))
            break

    # 2. perform linear interpolation
    x1, y1 = xs[closest[0]], ys[closest[0]]
    if y1 < linear(x1, m, b):
        x2, y2 = x1, y1
        x1, y1 = xs[closest[0] - 1], ys[closest[0] - 1]
    else:
        x2, y2 = xs[closest[0] + 1], ys[closest[0] + 1]

    m_int = (y2 - y1) / (x2 - x1)
    b_int = y1 - m_int * x1

    x     = (b_int - b) / (m - m_int)
    y     = linear(x, m, b)

    return x, y

def full_function(T, dH, dS, m1, b1, m2, b2, c0 = 1e-6, structType='heterodimer'):
    if structType == 'heterodimer':
        theta = theta_from_therm(T, dH, dS, c0)
    elif structType == 'homodimer':
        theta = theta_from_therm_homo(T, dH, dS, c0)
    elif structType == 'monomer':
        theta = theta_from_therm_mono(T, dH, dS, c0)
    return theta * (m1 * T + b1) + (1 - theta) * (m2 * T + b2)

def full_function_res(x, T, y, c0 = 1e-6, structType = 'heterodimer'):
    return y - full_function(T, *x, c0, structType)

def full_function_multi(T, *args, **kwargs):
    default_c0 = 1e-6
    default_structType = 'hetero'
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

    structType = kwargs.get('structType', default_structType)
    # compose result function fit
    for i, c in enumerate(cs):
        if structType == 'heterodimer':
            theta = theta_from_therm(T, dH, dS, c)
        elif structType == 'homodimer':
            theta = theta_from_therm_homo(T, dH, dS, c)
        elif structType == 'monomer':
            theta = theta_from_therm_mono(T, dH, dS, c)

        f.append(theta * (m1s[i] * T + b1s[i]) + (1 - theta) * (m2s[i] * T + b2s[i]))

    return f

def full_function_multi_res(x, T, ys, **kwargs):
    # set residual function, default to simple differences
    res_func = kwargs.get('res_func', res_diff)

    if not res_func:
        res_func = res_diff

    # compute and return actual sum of residuals
    return reduce(lambda x, y: x + y, map(res_func, zip(ys, full_function_multi(T, x, **kwargs))))

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

def theta_from_therm_mono(T, dH, dS, c0 = 1e-6):
    tt    = T - constants.T0
    dG    = dH - tt * dS
    return np.exp(- dG / (constants.R * tt))/(1+np.exp(- dG / (constants.R * tt)))
