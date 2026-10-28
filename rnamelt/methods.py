import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Optional

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
# Every tunable default is a field of one of the frozen dataclasses below.
# SingleCurve takes one per group (solver=, vh=, fit_init=, bounds=,
# multistart=, raw=, windows=), as an instance or as a dict of overrides;
# unknown keys and invalid values are rejected when the curve is built, and
# None in an override keeps the current value.

SEED_POLICIES = ("auto", "vh", "user", "fallback")
BOUNDS_MODES = ("physical", "loose", "positive", "unbounded")

# Options of SolverSettings forwarded as-is to least_squares().
_SCIPY_LEAST_SQUARES_KEYS = (
    "max_nfev", "ftol", "gtol", "xtol",
    "method", "loss", "f_scale", "jac", "verbose",
)


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


class _Settings:
    """Shared behaviour of the settings dataclasses."""
    _group = ""

    @classmethod
    def of(cls, value=None, **changes):
        """Settings from None (defaults), an instance, or a dict of
        overrides, with ``changes`` applied on top."""
        base = value if isinstance(value, cls) else cls()
        if isinstance(value, Mapping):
            changes = {**value, **changes}
        elif value is not None and not isinstance(value, cls):
            raise ValueError(f"{cls._group} settings must be a dict or {cls.__name__}, "
                             f"got {value!r}")
        return base.updated(changes)

    def updated(self, changes=None):
        """A copy with ``changes`` applied: None values keep the current value,
        dict fields (x_scale, detector) are merged key by key."""
        if isinstance(changes, type(self)):
            return changes
        changes = dict(changes or {})
        names = [f.name for f in fields(self)]
        unknown = sorted(set(changes) - set(names))
        if unknown:
            raise ValueError(f"unknown {self._group} option(s) {unknown}; "
                             f"valid keys: {names}")
        kept = {}
        for k, v in changes.items():
            if v is None:
                continue
            cur = getattr(self, k)
            kept[k] = {**cur, **dict(v)} if isinstance(cur, Mapping) and isinstance(v, Mapping) else v
        return replace(self, **kept) if kept else self

    def _set(self, name, value):
        object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SolverSettings(_Settings):
    """scipy.optimize.least_squares options for the nonlinear fits;
    ``x_scale`` is the characteristic scale of each parameter kind."""
    _group = "solver"
    max_nfev: float = 1e12
    ftol: float = 1e-8
    gtol: float = 1e-8
    xtol: float = 1e-8
    method: str = "trf"
    loss: str = "linear"
    f_scale: float = 0.02
    jac: str = "2-point"
    verbose: int = 0
    x_scale: Mapping = field(default_factory=lambda: {
        "dH": 1.0, "dS": 0.01, "slope": 1.0, "intercept": 100.0})

    def __post_init__(self):
        xs = dict(self.x_scale)
        unknown = sorted(set(xs) - {"dH", "dS", "slope", "intercept"})
        if unknown:
            raise ValueError(f"unknown solver['x_scale'] option(s) {unknown}; "
                             f"valid keys: ['dH', 'dS', 'slope', 'intercept']")
        for k, v in xs.items():
            if _finite("solver['x_scale']", k, v) <= 0:
                raise ValueError(f"solver['x_scale'][{k!r}] must be > 0, got {v!r}")
        self._set("x_scale", xs)
        if _finite("solver", "max_nfev", self.max_nfev) < 1:
            raise ValueError(f"solver['max_nfev'] must be >= 1, got {self.max_nfev!r}")


@dataclass(frozen=True)
class VanHoffSettings(_Settings):
    """van't Hoff linearisation (ln K vs T_scale/T): points with folded
    fraction outside [border, 1 − border] are excluded; t1_min / t1_max clip
    the linearised x-axis (None = no clip)."""
    _group = "vh"
    border: float = 0.15
    t1_min: Optional[float] = None
    t1_max: Optional[float] = None
    T_scale: float = 1000.0

    def __post_init__(self):
        if not 0.0 <= _finite("vh", "border", self.border) < 0.5:
            raise ValueError(f"vh['border'] must be in [0, 0.5), got {self.border}")
        if _finite("vh", "T_scale", self.T_scale) <= 0:
            raise ValueError(f"vh['T_scale'] must be > 0, got {self.T_scale!r}")
        for k in ("t1_min", "t1_max"):
            if getattr(self, k) is not None:
                self._set(k, _finite("vh", k, getattr(self, k)))
        if self.t1_min is not None and self.t1_max is not None and not self.t1_min < self.t1_max:
            raise ValueError(f"vh needs t1_min < t1_max, got {self.t1_min}, {self.t1_max}")


@dataclass(frozen=True)
class FitInitSettings(_Settings):
    """Starting values of the nonlinear fit. ``seed_from`` picks the dH / dS
    starting values:
      "auto"     dH_init / dS_init if set, else the van't Hoff estimate if it
                 lies inside dH_seed_range / dS_seed_range, else the fallback
      "vh"       van't Hoff estimate (if in range), else the fallback
      "user"     dH_init / dS_init, both required
      "fallback" dH_fallback / dS_fallback (no van't Hoff is computed)
    ``lin_init`` points are averaged for the baseline-intercept seeds;
    ``b1_init`` / ``b2_init`` are explicit (slope, intercept) seeds."""
    _group = "fit_init"
    seed_from: str = "auto"
    dH_init: Optional[float] = None
    dS_init: Optional[float] = None
    lin_init: int = 10
    b1_init: Optional[tuple] = None
    b2_init: Optional[tuple] = None
    dH_fallback: float = -80.0
    dS_fallback: float = -0.2
    dH_seed_range: tuple = (-150.0, 0.0)
    dS_seed_range: tuple = (-5.0, 0.0)

    def __post_init__(self):
        if self.seed_from not in SEED_POLICIES:
            raise ValueError(f"unknown fit_init['seed_from'] {self.seed_from!r}; "
                             f"valid: {list(SEED_POLICIES)}")
        if self.seed_from == "user" and (self.dH_init is None or self.dS_init is None):
            raise ValueError("fit_init['seed_from']='user' needs both dH_init and dS_init")
        for k in ("dH_init", "dS_init"):
            if getattr(self, k) is not None:
                self._set(k, _finite("fit_init", k, getattr(self, k)))
        for k in ("dH_fallback", "dS_fallback"):
            self._set(k, _finite("fit_init", k, getattr(self, k)))
        for k in ("dH_seed_range", "dS_seed_range"):
            self._set(k, _interval("fit_init", k, getattr(self, k)))
        self._set("lin_init", _positive_int("fit_init", "lin_init", self.lin_init))
        for k in ("b1_init", "b2_init"):
            v = getattr(self, k)
            if v is not None:
                try:
                    m, b = v
                except (TypeError, ValueError):
                    raise ValueError(f"fit_init[{k!r}] must be a (slope, intercept) pair, got {v!r}")
                self._set(k, (_finite("fit_init", k, m), _finite("fit_init", k, b)))

    @property
    def needs_vh(self) -> bool:
        """Does picking the dH / dS seeds need a van't Hoff estimate?"""
        return self.seed_from == "vh" or (
            self.seed_from == "auto" and (self.dH_init is None or self.dS_init is None))

    def seeds(self, vh_estimates=()) -> dict:
        """dH (kcal/mol) / dS (kcal/(mol·K)) starting values under seed_from,
        as {"dH", "dS", "dH_from", "dS_from"} with sources "user", "vh" or
        "fallback". ``vh_estimates`` are van't Hoff (dH, dS) pairs; those with
        both values inside the seed ranges are used, through their median."""
        (hlo, hhi), (slo, shi) = self.dH_seed_range, self.dS_seed_range
        vh_ok = [(h, s) for h, s in vh_estimates if hlo < h < hhi and slo < s < shi]
        seeds = {}
        for i, (key, user, fallback) in enumerate((("dH", self.dH_init, self.dH_fallback),
                                                   ("dS", self.dS_init, self.dS_fallback))):
            if self.seed_from in ("auto", "user") and user is not None:
                seeds[key], seeds[key + "_from"] = user, "user"
            elif self.seed_from in ("auto", "vh") and vh_ok:
                seeds[key] = float(np.median([p[i] for p in vh_ok]))
                seeds[key + "_from"] = "vh"
            else:
                seeds[key], seeds[key + "_from"] = fallback, "fallback"
        return seeds


@dataclass(frozen=True)
class BoundsSettings(_Settings):
    """Bounds of the nonlinear fit: dH (kcal/mol) and dS (kcal/(mol·K))
    boxes; ``mode`` bounds the baseline slopes (see _fit_bounds) and
    ``slope_cap_k`` sets the cap of "physical". A fit whose model Tm lies
    more than ``Tm_margin`` °C outside the fitted window is reported as failed."""
    _group = "bounds"
    mode: str = "physical"
    dH: tuple = (-200.0, 0.0)
    dS: tuple = (-5.0, 0.0)
    slope_cap_k: float = 5.0
    Tm_margin: float = 3.0

    def __post_init__(self):
        if self.mode not in BOUNDS_MODES:
            raise ValueError(f"unknown bounds mode {self.mode!r}; valid: {list(BOUNDS_MODES)}")
        self._set("dH", _interval("bounds", "dH", self.dH))
        self._set("dS", _interval("bounds", "dS", self.dS))
        if _finite("bounds", "slope_cap_k", self.slope_cap_k) <= 0:
            raise ValueError(f"bounds['slope_cap_k'] must be > 0, got {self.slope_cap_k!r}")
        if _finite("bounds", "Tm_margin", self.Tm_margin) < 0:
            raise ValueError(f"bounds['Tm_margin'] must be >= 0, got {self.Tm_margin!r}")


@dataclass(frozen=True)
class MultistartSettings(_Settings):
    """strategy="multistart": one fit per dH seed (plus fwhm_factors × the
    FWHM-of-dS/dT estimate), dS seed = dH seed / seed_Tm_K. A result counts
    only if it passes the Tm check and |dH| >= min_abs_dH; the lowest-RMS
    such result wins."""
    _group = "multistart"
    dH_seeds: tuple = (-40.0, -80.0, -120.0, -160.0)
    fwhm_factors: tuple = (1.0, 1.5)
    seed_Tm_K: float = 350.0
    min_abs_dH: float = 10.0

    def __post_init__(self):
        for k in ("dH_seeds", "fwhm_factors"):
            try:
                self._set(k, tuple(_finite("multistart", k, v) for v in getattr(self, k)))
            except TypeError:
                raise ValueError(f"multistart[{k!r}] must be a sequence of numbers, "
                                 f"got {getattr(self, k)!r}")
        if not self.dH_seeds and not self.fwhm_factors:
            raise ValueError("multistart needs at least one dH seed or fwhm factor")
        if _finite("multistart", "seed_Tm_K", self.seed_Tm_K) <= 0:
            raise ValueError(f"multistart['seed_Tm_K'] must be > 0, got {self.seed_Tm_K!r}")
        if _finite("multistart", "min_abs_dH", self.min_abs_dH) < 0:
            raise ValueError(f"multistart['min_abs_dH'] must be >= 0, got {self.min_abs_dH!r}")


@dataclass(frozen=True)
class RawSettings(_Settings):
    """Raw Tm (crossing of the mean baseline with the data): the first / last
    ``edge_skip`` points are ignored when searching for the crossing."""
    _group = "raw"
    edge_skip: int = 10

    def __post_init__(self):
        self._set("edge_skip", _positive_int("raw", "edge_skip", self.edge_skip, minimum=0))


@dataclass(frozen=True)
class WindowSettings(_Settings):
    """Baseline windows of a SingleCurve built without explicit windows:
    detected with auto_baseline.find_baselines (``detector`` = its keyword
    arguments); if detection fails, ``fallback_width`` °C at each scan edge,
    with a BaselineWarning. The detector keys are checked by the API."""
    _group = "windows"
    detector: Mapping = field(default_factory=dict)
    fallback_width: float = 10.0

    def __post_init__(self):
        if not isinstance(self.detector, Mapping):
            raise ValueError(f"windows['detector'] must be a dict, got {self.detector!r}")
        self._set("detector", dict(self.detector))
        if _finite("windows", "fallback_width", self.fallback_width) <= 0:
            raise ValueError(f"windows['fallback_width'] must be > 0, got {self.fallback_width!r}")


def _least_squares_kwargs(solver):
    """least_squares kwargs (without x_scale) from solver settings/overrides."""
    s = SolverSettings.of(solver)
    return {k: getattr(s, k) for k in _SCIPY_LEAST_SQUARES_KEYS}


def _x_scale(solver, n_curves=1):
    """x_scale vector for [dH, dS] + n_curves × [m1, b1, m2, b2]."""
    xs = SolverSettings.of(solver).x_scale
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
    edges default to the scan edges). A zero-width window (equal edges)
    gives a flat baseline at the signal measured nearest that temperature,
    the best guess when the plateau is missing. ``edge_skip`` (default
    RawSettings) end points are ignored when searching for the crossing.
    Returns ``(Tm, y_Tm, base_b, base_ub, base_med)``.
    """
    T = np.array(temperatures)
    if edge_skip is None:
        edge_skip = RawSettings().edge_skip
    if baseline_bound_maxT is None or baseline_unbound_minT is None:
        raise ValueError("T_m_ds_raw needs baseline_bound_maxT and baseline_unbound_minT")

    bl_b_minT  = T[0]  if baseline_bound_minT   is None else baseline_bound_minT
    bl_ub_maxT = T[-1] if baseline_unbound_maxT is None else baseline_unbound_maxT
    bl_b_maxT  = baseline_bound_maxT
    bl_ub_minT = baseline_unbound_minT

    if bl_b_maxT < bl_b_minT:
        raise ValueError(
            f"folded baseline window is empty or inverted: "
            f"[{bl_b_minT}, {bl_b_maxT}] °C"
        )
    elif bl_b_maxT == bl_b_minT:
        m_bound = 0.0
        b_bound = float(ydata[np.argmin(np.abs(T - bl_b_maxT))])
    else:
        m_bound, b_bound = functions.fit_linear(T, ydata, bl_b_minT, bl_b_maxT)

    if bl_ub_maxT < bl_ub_minT:
        raise ValueError(
            f"unfolded baseline window is empty or inverted: "
            f"[{bl_ub_minT}, {bl_ub_maxT}] °C"
        )
    elif bl_ub_maxT == bl_ub_minT:
        m_unbound = 0.0
        b_unbound = float(ydata[np.argmin(np.abs(T - bl_ub_minT))])
    else:
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
    """Linearised van't Hoff analysis. Unset options take VanHoffSettings.

    ``c0`` is the total strand concentration (M) or, for a heterodimer with
    unequal strands, the pair (c_A, c_B) per strand (M)."""
    vh = VanHoffSettings.of(border=border, T_scale=T_scale, t1_min=t1_min, t1_max=t1_max)
    border, T_scale, lo, hi = vh.border, vh.T_scale, vh.t1_min, vh.t1_max
    # ln K(θ) from functions.ln_k_from_theta; at Tm (θ = 1/2) K = 1/c_eff
    # (4/c0 for a heterodimer, 1/c0 for a homodimer) — see
    # functions.tm_from_therm.
    if struct_type == 'monomer':
        raise ValueError("Van't Hoff analysis not possible for monomers")
    if struct_type not in ('heterodimer', 'homodimer'):
        raise ValueError(
            f"unknown struct_type {struct_type!r}; "
            f"valid: 'heterodimer', 'homodimer', 'monomer'"
        )
    ln_k_tm = -np.log(functions.c_eff(c0, struct_type))

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
        functions.ln_k_from_theta(ff, c0, struct_type)
        if border <= ff <= (1 - border)
        else None
        for ff in f_folded
    ]

    t1 = [T_scale / (t - constants.T0) for t in T]

    # keep points with a defined ln K and t1 inside [t1_min, t1_max]
    # (None = no clip on that side)
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

    t_m = (ln_k_tm - res_lsq.x[1]) / res_lsq.x[0]
    T_m = T_scale / t_m + constants.T0
    dH  = -res_lsq.x[0] * constants.R * T_scale
    dS    = dH / (T_m - constants.T0) + constants.R * ln_k_tm
    dG_37 = dH - (37. - constants.T0) * dS

    return T_m, dG_37, dH, dS, t1, K, xdata, ydata, tuple(res_lsq.x.tolist())

def vant_hoff_concentration(
    tm_values,
    concentrations=None,
    self_complementary: bool = False,
    c_eff=None,
) -> dict:
    """Linear-regression van't Hoff fit of 1/Tm vs ln(C_T/f) → ΔH, ΔS, ΔG(37°C).

    Give either the total strand concentrations (M) with
    ``self_complementary`` (f = 1 or 4), or ``c_eff`` (M), the effective
    concentrations of functions.c_eff, regressed as ln c_eff (needed when
    strand ratios differ between curves)."""
    if (concentrations is None) == (c_eff is None):
        raise ValueError("give either concentrations or c_eff")
    tm = np.asarray(tm_values, dtype=float)
    ct = np.asarray(concentrations if c_eff is None else c_eff, dtype=float)

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

    f = 1.0 if (self_complementary or c_eff is not None) else 4.0
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

    # Standard errors from the scatter of the points about the line; with
    # two points the line is exact and the error undefined.
    n = tm.size
    if n > 2:
        x_mean = ln_ct.mean()
        s_xx = float(np.sum((ln_ct - x_mean) ** 2))
        resid = inv_tm - (slope * ln_ct + intercept)
        s2 = float(resid @ resid) / (n - 2)
        var_a = s2 / s_xx
        cov_ab = np.array([[var_a,           -x_mean * var_a],
                           [-x_mean * var_a, s2 / n + x_mean**2 * var_a]])
        errors = _thermo_error_fields(
            functions.thermo_cov_from_regression(slope, intercept, cov_ab), None)
    else:
        errors = _thermo_error_fields(
            None, "2 points fit the line exactly: no degrees of freedom for an error")

    return {
        "dH":             float(dH),
        "dS":             float(dS),
        "dG_37":          float(dG_37),
        **errors,
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

    ``bounds`` is a BoundsSettings or an override dict (dH / dS boxes,
    slope_cap_k); ``mode`` overrides ``bounds['mode']``. Baseline modes:
      "physical"  : |m1|, |m2| <= slope_cap_k × (ΔS/ΔT of the data)
      "loose"     : m1 >= 0, m2 free
      "positive"  : m1 >= 0, m2 >= 0
      "unbounded" : both slopes free
    Intercepts are always free.
    """
    b = BoundsSettings.of(bounds, mode=mode)
    (dH_lo, dH_hi), (dS_lo, dS_hi) = b.dH, b.dS
    inf = np.inf
    if b.mode == "physical":
        cap = _slope_cap(T, d, k=b.slope_cap_k)
        m1, m2 = (-cap, cap), (-cap, cap)
    elif b.mode == "loose":
        m1, m2 = (0.0, inf), (-inf, inf)
    elif b.mode == "positive":
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
    return_errors=False,
):
    """Nonlinear two-state fit of one curve.

    Unset starting values take FitInitSettings (dH/dS fallbacks, lin_init),
    unset bounds take BoundsSettings; ``bounds_mode`` overrides
    ``bounds['mode']``.

    Returns ``(dG_37, dH, dS, Tm, y_Tm, base_b, base_ub, base_med)``; with
    ``return_converged=True`` a trailing ``converged`` bool is appended,
    then with ``return_errors=True`` a dict of standard errors (SE_dH,
    SE_dS, SE_dG at 37 °C, SE_Tm, cov_dH_dS, SE_note) from s²·(JᵀJ)⁻¹.
    ``Tm`` is the model melting temperature from the fitted ΔH/ΔS and c0
    (θ = 1/2), and ``y_Tm`` the fitted signal there (the mean baseline).
    Hitting ``max_nfev`` emits a ConvergenceWarning and returns the best
    parameters found (``converged=False``) instead of raising.
    """
    xdata = np.array(T)
    ydata = np.array(d)
    init = FitInitSettings.of(dH_init=dH_init, dS_init=dS_init, lin_init=lin_init)
    dH0 = init.dH_init if init.dH_init is not None else init.dH_fallback
    dS0 = init.dS_init if init.dS_init is not None else init.dS_fallback

    if b1_init is None:
        b1_init = _intercept_seed(ydata, init.lin_init, head=True)
    if b2_init is None:
        b2_init = _intercept_seed(ydata, init.lin_init, head=False)

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

    dH, dS, m_b, b_b, m_ub, b_ub = res_lsq.x.tolist()
    dG_37 = dH - (37. - constants.T0) * dS

    # median of base lines
    m_med, b_med = (m_b + m_ub) / 2, (b_b + b_ub) / 2

    T_m = functions.tm_from_therm(dH, dS, c0, struct_type)
    y_Tm = functions.linear(T_m, m_med, b_med)

    out = (dG_37, dH, dS, T_m, y_Tm, (m_b, b_b), (m_ub, b_ub), (m_med, b_med))
    if return_converged:
        out = (*out, converged)
    if return_errors:
        cov_HS, note = _dH_dS_covariance(res_lsq, SolverSettings.of(solver).loss)
        errors = _thermo_error_fields(cov_HS, note)
        errors["SE_Tm"] = (None if cov_HS is None
                           else functions.tm_error(dH, dS, cov_HS, c0, struct_type))
        out = (*out, errors)
    return out

def _thermo_error_fields(cov_HS, note) -> dict:
    """SE_dH / SE_dS / SE_dG (37 °C) and cov_dH_dS from a 2×2 (ΔH, ΔS)
    covariance; all None, with `note` saying why, when it is undefined."""
    if cov_HS is None:
        return {"SE_dH": None, "SE_dS": None, "SE_dG": None,
                "cov_dH_dS": None, "SE_note": note}
    se_H, se_S, se_G = functions.thermo_errors(cov_HS, 37.0)
    return {"SE_dH": se_H, "SE_dS": se_S, "SE_dG": se_G,
            "cov_dH_dS": float(cov_HS[0, 1]), "SE_note": note}


def _dH_dS_covariance(res_lsq, loss="linear"):
    """(2×2 covariance of ΔH, ΔS, note) from the Jacobian at a least_squares
    solution whose first two parameters are ΔH and ΔS: s²·(JᵀJ)⁻¹ over the
    parameters not held at a bound. Returns (None, reason) when undefined."""
    J, r = np.asarray(res_lsq.jac, dtype=float), np.asarray(res_lsq.fun, dtype=float)
    free = np.asarray(res_lsq.active_mask) == 0
    if not (free[0] and free[1]):
        return None, "ΔH or ΔS is at a fit bound: its error is undefined"
    dof = J.shape[0] - int(free.sum())
    if dof <= 0:
        return None, "no degrees of freedom for an error"
    Jf = J[:, free]
    # Column scaling keeps the inversion well-posed: dS, slopes and
    # intercepts differ by orders of magnitude.
    scale = np.linalg.norm(Jf, axis=0)
    if np.any(scale == 0):
        return None, "a fit parameter does not affect the curve: error undefined"
    Jn = Jf / scale
    A = Jn.T @ Jn
    if np.linalg.cond(A) > 1e12:
        return None, "fit parameters are too strongly correlated: error undefined"
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.inv(A) / np.outer(scale, scale)
    note = None if loss == "linear" else f"approximate (loss={loss!r})"
    return cov[:2, :2], note


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
    return_errors=False,
):
    """Joint two-state fit of several curves sharing ΔH / ΔS.

    Returns ``(dG_37, dH, dS, Tms, y_Tms, base_lo, base_hi, base_med)``
    (lists per curve); ``return_converged=True`` appends a ``converged``
    bool, then ``return_errors=True`` a dict with SE_dH, SE_dS, SE_dG
    (37 °C), cov_dH_dS and SE_note, from the covariance s²·(JᵀJ)⁻¹ at the
    solution. Tms are model melting temperatures from ΔH/ΔS and each
    curve's concentration; see fit_full_function for the max_nfev behaviour.
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

    init = FitInitSettings.of(dH_init=dH_init, dS_init=dS_init)
    dH0 = init.dH_init if init.dH_init is not None else init.dH_fallback
    dS0 = init.dS_init if init.dS_init is not None else init.dS_fallback

    x_inits = [dH0, dS0]
    for i in range(n):
        li = FitInitSettings.of(lin_init=lin_list[i]).lin_init
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

    x = res_lsq.x.tolist()
    dH, dS = x[0], x[1]
    dG_37 = dH - (37 - constants.T0) * dS

    curve_cs = list(cs) if cs is not None and len(cs) else [c0] * len(ds)
    baselines_lo = []
    baselines_me = []
    baselines_hi = []
    T_ms         = []
    y_dats       = []
    for j, i in enumerate(range(2, len(x), 4)):
        m_b, b_b    = x[i], x[i + 1]
        m_ub, b_ub  = x[i + 2], x[i + 3]
        m_m, b_m    = (m_b + m_ub) / 2, (b_b + b_ub) / 2
        T_m         = functions.tm_from_therm(dH, dS, curve_cs[j], struct_type)
        baselines_lo.append( (m_b, b_b) )
        baselines_hi.append( (m_ub, b_ub) )
        baselines_me.append( (m_m, b_m) )
        T_ms.append(T_m)
        y_dats.append(functions.linear(T_m, m_m, b_m))

    out = (dG_37, dH, dS, T_ms, y_dats, baselines_lo, baselines_hi, baselines_me)
    if return_converged:
        out = (*out, converged)
    if return_errors:
        loss = SolverSettings.of(solver).loss
        out = (*out, _thermo_error_fields(*_dH_dS_covariance(res_lsq, loss)))
    return out

