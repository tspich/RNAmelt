import inspect
import warnings
from dataclasses import fields

import numpy as np
from typing import Any, Mapping, Optional, Union, Iterator, overload
from collections.abc import Sequence
from rnamelt import methods, constants, functions, auto_baseline
from rnamelt.results import (
    RawTm,
    VanHoffFit,
    SingleFit,
    GlobalFit,
    GlobalFitCurve,
    ConcDepResult,
    ConcentrationCurve,
    ConcentrationSeries,
)

def _tm_outside(Tm, T, margin):
    """Error message if the model Tm lies more than `margin` °C outside
    the fitted temperatures T, else None."""
    lo, hi = float(np.min(T)), float(np.max(T))
    if lo - margin <= Tm <= hi + margin:
        return None
    return (f"fit Tm {Tm:.1f} °C lies outside the fit window "
            f"[{lo:.1f}, {hi:.1f}] °C ± {margin:g} — the fit did not find the "
            "transition; check the baseline windows or the starting values")


def _check_seeds_in_bounds(init, bnd):
    """Reject user dH_init / dS_init that lie outside the fit bounds."""
    for key, box in (("dH_init", "dH"), ("dS_init", "dS")):
        v = getattr(init, key)
        lo, hi = getattr(bnd, box)
        if v is not None and not lo <= v <= hi:
            raise ValueError(
                f"fit_init[{key!r}]={v} lies outside bounds[{box!r}]=({lo}, {hi})"
            )


def _check_detector_kwargs(kwargs):
    """Reject options that auto_baseline.find_baselines doesn't take."""
    unknown = sorted(set(kwargs) - _DETECTOR_KEYS)
    if unknown:
        raise ValueError(f"unknown baseline detector option(s) {unknown}; "
                         f"valid: {sorted(_DETECTOR_KEYS)}")


_WINDOW_KEYS = ("fold_low_T", "fold_high_T", "unf_low_T", "unf_high_T")
_DETECTOR_KEYS = frozenset(
    inspect.signature(auto_baseline.find_baselines).parameters) - {"T", "signal"}


class SingleCurve:
    def __init__(
        self,
        temps,
        signal,
        *,
        name: str = "unnamed",
        oligoC: Optional[Union[float, tuple[float, float]]] = None,
        struct_type: str = "heterodimer",
        fold_low_T: Optional[float] = None,
        fold_high_T: Optional[float] = None,
        unf_low_T: Optional[float] = None,
        unf_high_T: Optional[float] = None,
        windows: Optional[Mapping[str, Any]] = None,
        solver: Optional[Mapping[str, Any]] = None,
        vh: Optional[Mapping[str, Any]] = None,
        fit_init: Optional[Mapping[str, Any]] = None,
        bounds: Optional[Mapping[str, Any]] = None,
        multistart: Optional[Mapping[str, Any]] = None,
        raw: Optional[Mapping[str, Any]] = None,
    ):
        """One melting curve.

        ``oligoC`` is the per-strand concentration in µM, required for
        heterodimers and homodimers (ignored for monomers). For a
        heterodimer whose strands are not in equal amounts, give the pair
        ``(A, B)`` of both strand concentrations in µM; the folded
        fraction then refers to the limiting strand.

        Baseline windows (°C): the folded baseline is fitted on
        [fold_low_T, fold_high_T], the unfolded one on [unf_low_T,
        unf_high_T], and [fold_low_T, unf_high_T] is the fit window. Give
        all four, or none: then they are detected with
        ``auto_baseline.find_baselines`` (``windows['detector']`` options),
        falling back to ``windows['fallback_width']`` °C at each scan edge
        with a BaselineWarning. ``window_source`` records which happened
        ("user", "auto" or "fallback").

        The settings groups ``windows``, ``solver``, ``vh``, ``fit_init``,
        ``bounds``, ``multistart`` and ``raw`` override the matching
        settings classes in ``rnamelt`` (e.g. ``rnamelt.BoundsSettings``),
        given as an instance or a dict of overrides, and are validated here.
        """
        signal          = np.asarray(signal,  dtype=float)
        temps           = np.asarray(temps,  dtype=float)

        if temps.ndim != 1 or signal.ndim != 1:
            raise ValueError("temps and signal must be 1-D arrays")
        if temps.shape != signal.shape:
            raise ValueError(
                f"temps and signal must have the same length "
                f"(got {temps.shape[0]} and {signal.shape[0]})"
            )
        if temps.size < 10:
            raise ValueError(f"need at least 10 data points, got {temps.size}")
        if struct_type not in constants.STRAND_STOICHIOMETRY:
            raise ValueError(
                f"unknown struct_type {struct_type!r}; valid: "
                f"{sorted(constants.STRAND_STOICHIOMETRY)}"
            )
        if oligoC is None:
            if struct_type != "monomer":
                raise ValueError(
                    f"oligoC (per-strand concentration, µM) is required "
                    f"for struct_type={struct_type!r}"
                )
        oligoC_B = None
        if oligoC is not None:
            is_pair = isinstance(oligoC, (tuple, list, np.ndarray))
            if is_pair and len(oligoC) != 2:
                raise ValueError(f"oligoC must be a number or a pair (A, B), got {oligoC!r}")
            if is_pair and struct_type != "heterodimer":
                raise ValueError(
                    f"a pair of strand concentrations only applies to "
                    f"heterodimers, not struct_type={struct_type!r}"
                )
            concs = []
            for v in (oligoC if is_pair else (oligoC,)):
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    f = np.nan
                if not np.isfinite(f) or f <= 0:
                    raise ValueError(f"oligoC must be a positive finite number, got {v!r}")
                concs.append(f)
            oligoC, oligoC_B = concs if is_pair else (concs[0], None)

        # Drop NaN signal points once — every consumer downstream can assume
        # clean arrays.
        keep = ~np.isnan(signal)
        if keep.sum() < 10:
            raise ValueError(
                f"need at least 10 non-NaN signal points, got {int(keep.sum())}"
            )

        self.name            = name
        self.temps           = temps[keep]
        self.signal          = signal[keep]
        self.oligoC          = oligoC
        self.oligoC_B        = oligoC_B     # strand B (µM) when unequal; else None
        self.struct_type     = struct_type
        # Fully resolved settings (defaults + overrides), validated now.
        self.windows         = methods.WindowSettings.of(windows)
        _check_detector_kwargs(self.windows.detector)
        self.solver          = methods.SolverSettings.of(solver)
        self.vh              = methods.VanHoffSettings.of(vh)
        self.fit_init        = methods.FitInitSettings.of(fit_init)
        self.bounds          = methods.BoundsSettings.of(bounds)
        self.multistart      = methods.MultistartSettings.of(multistart)
        self.raw             = methods.RawSettings.of(raw)
        _check_seeds_in_bounds(self.fit_init, self.bounds)

        given = dict(zip(_WINDOW_KEYS, (fold_low_T, fold_high_T, unf_low_T, unf_high_T)))
        set_keys = [k for k, v in given.items() if v is not None]
        if set_keys and len(set_keys) < 4:
            missing = [k for k in _WINDOW_KEYS if given[k] is None]
            raise ValueError(
                f"partial baseline windows: got {set_keys}, missing {missing} — "
                f"give all four or none (auto-detect)"
            )
        if set_keys:
            windows_T = tuple(float(given[k]) for k in _WINDOW_KEYS)
            self.window_source = "user"
        else:
            try:
                windows_T = self.auto_baselines(**self.windows.detector)
                fl, fh, ul, uh = windows_T
                if not (np.all(np.isfinite(windows_T)) and fl <= fh < ul <= uh):
                    raise ValueError(f"detected windows out of order: {windows_T}")
                self.window_source = "auto"
            except Exception as e:
                windows_T = self._edge_windows()
                self.window_source = "fallback"
                warnings.warn(
                    f"{name}: automatic baseline detection failed ({e}); using "
                    f"windows of {windows_T[1] - windows_T[0]:g} °C at each scan edge",
                    methods.BaselineWarning, stacklevel=2,
                )
        fl, fh, ul, uh = windows_T
        if not (np.all(np.isfinite(windows_T)) and fl <= fh < ul <= uh):
            raise ValueError(
                f"baseline windows must satisfy fold_low_T <= fold_high_T < "
                f"unf_low_T <= unf_high_T; got {windows_T}"
            )
        self.fold_low_T, self.fold_high_T, self.unf_low_T, self.unf_high_T = windows_T

    def _edge_windows(self) -> tuple[float, float, float, float]:
        """Fallback windows: fallback_width °C at each scan edge, shrunk to a
        quarter of the scan on short curves."""
        T_min, T_max = float(self.temps.min()), float(self.temps.max())
        w = min(float(self.windows.fallback_width), (T_max - T_min) / 4)
        return (T_min, T_min + w, T_max - w, T_max)

    def auto_baselines(self, **detector_kwargs) -> tuple[float, float, float, float]:
        """(fold_low_T, fold_high_T, unf_low_T, unf_high_T) detected with
        ``auto_baseline.find_baselines`` on the full curve. Keyword arguments
        go to the detector. Pass the result back into a new SingleCurve to
        use it, e.g. ``SingleCurve(T, S, oligoC=c, **dict(zip(keys, w)))``."""
        _check_detector_kwargs(detector_kwargs)
        fl, fh, ul, uh = auto_baseline.find_baselines(
            self.temps, self.signal, **detector_kwargs
        )
        return (float(fl), float(fh), float(ul), float(uh))

    @property
    def window_temps(self) -> tuple[float, float, float, float]:
        """(fold_low_T, fold_high_T, unf_low_T, unf_high_T)."""
        return (self.fold_low_T, self.fold_high_T, self.unf_low_T, self.unf_high_T)

    def __repr__(self) -> str:
        return (
            f"SingleCurve(name={self.name!r}, n={self.temps.size}, "
            f"T=[{self.temps[0]:.1f}, {self.temps[-1]:.1f}]°C, "
            f"oligoC={self._oligo_repr()}µM, struct_type={self.struct_type!r})"
        )

    def _oligo_repr(self):
        return self.oligoC if self.oligoC_B is None else (self.oligoC, self.oligoC_B)

    @property
    def c0(self) -> Optional[float]:
        """Total strand concentration in M (oligo µM × stoichiometry, or
        A + B for unequal strands); None for a monomer built without oligoC."""
        if self.oligoC is None:
            return None
        if self.oligoC_B is not None:
            return 1e-6 * (self.oligoC + self.oligoC_B)
        return 1e-6 * self.oligoC * constants.STRAND_STOICHIOMETRY[self.struct_type]

    @property
    def conc(self):
        """Concentration passed to the model functions: the total strand
        concentration c0 (M), or the pair (c_A, c_B) in M for unequal strands."""
        if self.oligoC_B is not None:
            return (1e-6 * self.oligoC, 1e-6 * self.oligoC_B)
        return self.c0

    @property
    def c_eff(self) -> Optional[float]:
        """Effective concentration (M) with K(Tm) = 1/c_eff (see
        functions.c_eff); None for a monomer."""
        if self.conc is None:
            return None
        return functions.c_eff(self.conc, self.struct_type)

    def windowed_data(self) -> tuple[np.ndarray, np.ndarray]:
        """(T, signal) restricted to the fit window [fold_low_T, unf_high_T]."""
        mask = (self.temps >= self.fold_low_T) & (self.temps <= self.unf_high_T)
        if mask.sum() < 10:
            raise ValueError(
                f"need at least 10 data points in the fit window "
                f"[{self.fold_low_T}, {self.unf_high_T}]°C, got {int(mask.sum())}"
            )
        return self.temps[mask], self.signal[mask]

    def raw_tm(self, *, raw: Optional[Mapping[str, Any]] = None) -> RawTm:
        """Raw Tm: crossing of the mean of the two OLS baselines with the data.

        ``raw`` overrides the curve's raw settings for this call only.
        """
        raw_s = self.raw.updated(raw)
        TT, y = self.windowed_data()
        fl, fh, ul, uh = self.window_temps

        try:
            T_m_raw, y_r, base_b, base_ub, base_med = methods.T_m_ds_raw(
                TT, y,
                baseline_bound_minT=fl,
                baseline_bound_maxT=fh,
                baseline_unbound_minT=ul,
                baseline_unbound_maxT=uh,
                edge_skip=raw_s.edge_skip,
            )
        except Exception as e:
            return RawTm(ok=False, error=str(e))

        return RawTm(
            ok = True,
            Tm = T_m_raw,
            ym = y_r,
            base_b = base_b,
            base_ub = base_ub,
            base_med = base_med
        )

    def vant_hoff(
        self,
        *,
        base_b: Optional[tuple[float, float]] = None,
        base_ub: Optional[tuple[float, float]] = None,
        vh: Optional[Mapping[str, Any]] = None,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> VanHoffFit:
        """Linearised van't Hoff analysis (ln K vs 1/T).

        ``base_b`` / ``base_ub`` are the folded / unfolded (slope, intercept)
        baselines; missing ones come from ``raw_tm()``. ``vh`` and ``raw``
        override the curve's settings for this call only.
        """
        vh_s = self.vh.updated(vh)
        if self.struct_type == "monomer":
            return VanHoffFit(
                ok=False,
                error="van't Hoff linearisation is not defined for monomers "
                      "— Tm is concentration-independent for intramolecular folding",
            )

        if base_b is None or base_ub is None:
            r = self.raw_tm(raw=raw)
            if not r.ok:
                return VanHoffFit(ok=False, error=f"baselines unavailable: {r.error}")
            base_b = base_b if base_b is not None else r.base_b
            base_ub = base_ub if base_ub is not None else r.base_ub

        TT, y = self.windowed_data()

        try:
            T_m_vH, dG_37_vH, dH_vH, dS_vH, t1, K, xdata, ydata, fit_vh = methods.vant_hoff(
                TT, y,
                *base_b,
                *base_ub,
                self.conc,
                struct_type = self.struct_type,
                border=vh_s.border, T_scale=vh_s.T_scale,
                t1_min=vh_s.t1_min, t1_max=vh_s.t1_max,
            )
        except Exception as e:
            return VanHoffFit(ok=False, error=str(e))

        return  VanHoffFit(
            ok         = True,
            dH         = dH_vH,
            dS         = dS_vH,
            dG         = dG_37_vH,
            Tm         = T_m_vH,
            t1         = np.asarray(t1, dtype=float),
            K          = np.asarray([np.nan if k is None else k for k in K], dtype=float),
            xdata      = xdata,
            ydata      = ydata,
            fit_params = fit_vh,
        )

    def fit(
        self,
        *,
        fit_init: Optional[Mapping[str, Any]] = None,
        bounds: Optional[Mapping[str, Any]] = None,
        bounds_mode: Optional[str] = None,
        solver: Optional[Mapping[str, Any]] = None,
        multistart: Optional[Mapping[str, Any]] = None,
        strategy: str = "single",
        seeds: Optional[Sequence[float]] = None,
        vh_fit: Optional[VanHoffFit] = None,
        vh: Optional[Mapping[str, Any]] = None,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> SingleFit:
        """Non-linear two-state fit.

        Every settings argument overrides the curve's own settings for this
        call only (see the settings classes, e.g. ``rnamelt.FitInitSettings``):

        fit_init : starting values and ``seed_from`` policy
            ("auto" / "vh" / "user" / "fallback"), e.g.
            ``fit_init={"seed_from": "user", "dH_init": -100, "dS_init": -0.28}``.
        bounds, bounds_mode : fit bounds; ``bounds_mode`` is a shortcut for
            ``bounds={"mode": ...}`` ("physical", "loose", "positive",
            "unbounded").
        solver : scipy least_squares options.
        multistart : settings used when ``strategy="multistart"``.
        strategy : "single" (one fit) or "multistart" (one fit per seed,
            lowest-RMS physical result wins; ignores ``seed_from``).
        seeds : explicit multistart dH seeds (kcal/mol).
        vh_fit : precomputed van't Hoff result used for van't Hoff seeding;
            otherwise ``vant_hoff(vh=vh, raw=raw)`` is run when needed.

        The starting values used are reported in ``SingleFit.seed``.
        """
        init = self.fit_init.updated(fit_init)
        bnd  = self.bounds.updated(bounds).updated({"mode": bounds_mode})
        sol  = self.solver.updated(solver)
        ms   = self.multistart.updated(multistart)
        _check_seeds_in_bounds(init, bnd)
        if strategy not in ("single", "multistart"):
            raise ValueError(f"unknown strategy {strategy!r}; "
                             "valid: 'single', 'multistart'")

        TT, y = self.windowed_data()

        def _run_one(dH0, dS0, seed_info):
            try:
                (dG_37, dH, dS, Tm, ym, base_b, base_ub, base_med, converged,
                 errors) = methods.fit_full_function(
                    TT, y,
                    c0          = self.conc,
                    dH_init     = dH0,
                    dS_init     = dS0,
                    lin_init    = init.lin_init,
                    b1_init     = init.b1_init,
                    b2_init     = init.b2_init,
                    struct_type  = self.struct_type,
                    bounds      = bnd,
                    solver      = sol,
                    return_converged = True,
                    return_errors    = True,
                )
            except Exception as e:
                return SingleFit(ok=False, error=str(e), seed=seed_info)
            err = _tm_outside(Tm, TT, bnd.Tm_margin)
            if err:
                return SingleFit(ok=False, error=f"{err} (dH={dH:.1f})",
                                 converged=bool(converged), seed=seed_info)
            curve_fit = np.array([
                functions.full_function(t, dH, dS, *base_b, *base_ub,
                                        c0=self.conc, struct_type=self.struct_type)
                for t in TT
            ])
            return SingleFit(
                ok         = True,
                dH         = float(dH),
                dS         = float(dS),
                dG         = float(dG_37),
                Tm         = float(Tm),
                ym         = float(ym),
                base_b     = base_b,
                base_ub    = base_ub,
                base_med   = base_med,
                curve_fit  = curve_fit,
                converged  = bool(converged),
                seed       = seed_info,
                **errors,
            )

        if strategy == "single":
            vh_estimates = []
            if init.needs_vh:
                if vh_fit is None:
                    vh_fit = self.vant_hoff(vh=vh, raw=raw)
                if vh_fit.ok:
                    vh_estimates.append((vh_fit.dH, vh_fit.dS))
            seed_info = init.seeds(vh_estimates)
            return _run_one(seed_info["dH"], seed_info["dS"], seed_info)

        if seeds is None:
            seeds = list(ms.dH_seeds)
            fwhm_seed = methods.dH_seed_from_fwhm(
                TT, y, struct_type=self.struct_type, folding=True,
            )
            if fwhm_seed is not None:
                # FWHM under-predicts |dH|, hence the extra multiples
                seeds = [k * fwhm_seed for k in ms.fwhm_factors] + seeds
        if not len(seeds):
            raise ValueError("multistart needs at least one dH seed")

        best: Optional[SingleFit] = None
        best_cost = np.inf
        last_error = None
        for dH_seed in seeds:
            dS_seed = dH_seed / ms.seed_Tm_K
            f = _run_one(dH_seed, dS_seed, {"dH": float(dH_seed), "dS": float(dS_seed),
                                            "dH_from": "multistart",
                                            "dS_from": "multistart"})
            if not f.ok:
                last_error = f.error
                continue
            rms = float(np.sqrt(np.mean((y - f.curve_fit) ** 2)))
            cost = rms if abs(f.dH) >= ms.min_abs_dH else rms * 1e6
            if cost < best_cost:
                best_cost, best = cost, f
        return best if best is not None else SingleFit(
            ok=False, error=f"no multistart seed gave a valid fit (last: {last_error})"
        )

class MultiCurves(Sequence[SingleCurve]):
    def __init__(self, curves: Sequence[SingleCurve]):
        curves = list(curves)
        if len(curves) < 2:
            raise ValueError(f"need at least 2 curves, got {len(curves)}")
        for c in curves:
            if not isinstance(c, SingleCurve):
                raise TypeError(f"expected SingleCurve, got {type(c).__name__}")

        struct_types = {c.struct_type for c in curves}
        if len(struct_types) > 1:
            raise ValueError(
            f"all curves must have the same struct_type, got\
                    {sorted(struct_types)}"
            )
        self.curves = curves

    def __len__(self) -> int:
        return len(self.curves)

    def __iter__(self) -> Iterator[SingleCurve]:
        return iter(self.curves)

    @overload
    def __getitem__(self, key: int) -> SingleCurve: ...
    @overload
    def __getitem__(self, key: slice) -> "MultiCurves": ...
    @overload
    def __getitem__(self, key: str) -> SingleCurve: ...

    def __getitem__(self, key) -> Union[SingleCurve, "MultiCurves"]:
        """Index by position, slice (→ `MultiCurves`), or curve name."""
        if isinstance(key, slice):
            return MultiCurves(self.curves[key])
        if isinstance(key, str):
            for c in self.curves:
                if c.name == key:
                    return c
            raise KeyError(key)
        return self.curves[key]

    @property
    def names(self) -> list[str]:
        """Curve names, in order."""
        return [c.name for c in self.curves]

    @property
    def struct_type(self) -> str:
        """The `struct_type` shared by every curve."""
        return self.curves[0].struct_type

    @property
    def oligos(self) -> list[float]:
        """Per-curve strand concentrations in µM."""
        return [c.oligoC for c in self.curves]

    @property
    def self_complementary(self) -> bool:
        """True for homodimers, where C_T enters the model undivided."""
        return self.struct_type == "homodimer"

    # fit_init / bounds keys that belong to the shared dH / dS parameters
    # of the joint fit; every curve must agree on them.
    _SHARED_FIT_INIT = ("seed_from", "dH_init", "dS_init", "dH_fallback",
                        "dS_fallback", "dH_seed_range", "dS_seed_range")
    _SHARED_BOUNDS = ("dH", "dS")

    def _shared(self, group, settings, keys=None):
        """Return settings[0] after checking that all curves agree on ``keys``
        (all keys if None); raise ValueError naming the conflict otherwise."""
        keys = keys if keys is not None else [f.name for f in fields(settings[0])]
        for k in keys:
            vals = [getattr(s, k) for s in settings]
            if any(v != vals[0] for v in vals[1:]):
                per_curve = {c.name: v for c, v in zip(self.curves, vals)}
                raise ValueError(
                    f"curves disagree on shared setting {group}[{k!r}]: {per_curve}; "
                    f"pass {group}={{{k!r}: ...}} to set it for all curves"
                )
        return settings[0]

    def global_fit(
        self,
        *,
        fit_init: Optional[Mapping[str, Any]] = None,
        bounds: Optional[Mapping[str, Any]] = None,
        bounds_mode: Optional[str] = None,
        solver: Optional[Mapping[str, Any]] = None,
        vh: Optional[Mapping[str, Any]] = None,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> GlobalFit:
        """Joint two-state fit: one dH / dS shared by all curves, baselines
        per curve.

        Settings arguments override every curve's own settings for this call.
        Per-curve settings (bounds mode and slope cap, lin_init, b1_init /
        b2_init) keep each curve's value. Settings of the shared parameters
        (dH / dS seeds and seed_from, the dH / dS bounds, solver) must agree
        across curves; otherwise a ValueError asks you to pass them here.

        seed_from "auto" / "vh" seeds dH / dS with the median of the per-curve
        van't Hoff estimates that fall inside the seed ranges
        (``vant_hoff(vh=vh, raw=raw)`` per curve). The seeds used are
        reported in ``GlobalFit.seed``.
        """
        inits = [c.fit_init.updated(fit_init) for c in self.curves]
        bnds  = [c.bounds.updated(bounds).updated({"mode": bounds_mode}) for c in self.curves]
        sols  = [c.solver.updated(solver) for c in self.curves]
        init = self._shared("fit_init", inits, self._SHARED_FIT_INIT)
        self._shared("bounds", bnds, self._SHARED_BOUNDS)
        sol = self._shared("solver", sols)
        for i, b in zip(inits, bnds):
            _check_seeds_in_bounds(i, b)

        Ts, ys, cs = [], [], []
        for c in self.curves:
            try:
                T, y = c.windowed_data()
            except ValueError as e:
                return GlobalFit(error=f"curve {c.name!r}: {e}")
            Ts.append(T)
            ys.append(y)
            cs.append(c.conc)

        vh_estimates = []
        if init.needs_vh:
            for c in self.curves:
                v = c.vant_hoff(vh=vh, raw=raw)
                if v.ok:
                    vh_estimates.append((v.dH, v.dS))
        seed_info = init.seeds(vh_estimates)

        try:
            (dG_37, dH, dS, Tms, _y, bl_lo, bl_hi, bl_med,
             converged, errors) = methods.fit_full_function_multi(
                Ts, ys,
                cs         = cs,
                dH_init    = seed_info["dH"],
                dS_init    = seed_info["dS"],
                lin_init   = [i.lin_init for i in inits],
                b1_inits   = [i.b1_init for i in inits],
                b2_inits   = [i.b2_init for i in inits],
                struct_type = self.struct_type,
                bounds     = bnds,
                solver     = sol,
                return_converged = True,
                return_errors    = True,
            )
        except Exception as e:
            return GlobalFit(error=f"global fit failed: {e}", seed=seed_info)

        for i, c in enumerate(self.curves):
            err = _tm_outside(Tms[i], Ts[i], bnds[i].Tm_margin)
            if err:
                return GlobalFit(error=f"curve {c.name!r}: {err} (dH={dH:.1f})",
                                 converged=bool(converged), seed=seed_info)

        curves = []
        for i, c in enumerate(self.curves):
            base_b, base_ub = bl_lo[i], bl_hi[i]
            curves.append(GlobalFitCurve(
                name       = c.name,
                oligoC     = c.oligoC,
                c0         = c.c0,
                Tm         = float(Tms[i]),
                base_b     = base_b,
                base_ub    = base_ub,
                base_med   = bl_med[i],
                fit        = np.array([
                    functions.full_function(t, dH, dS, *base_b, *base_ub,
                                            c0=cs[i], struct_type=self.struct_type)
                    for t in Ts[i]
                ]),
                T_used     = Ts[i],
                signal     = ys[i],
                signal_all = c.signal,
                oligoC_B   = c.oligoC_B,
                c_eff      = c.c_eff,
            ))

        return GlobalFit(
            ok     = True,
            dH     = float(dH),
            dS     = float(dS),
            dG     = float(dG_37),
            T_used = Ts,
            curves = curves,
            converged = bool(converged),
            seed   = seed_info,
            **errors,
        )

    def conc_dependency(
        self,
        precomputed_tms: Optional[Mapping[str, Mapping[str, Any]]] = None,
        *,
        fit_init: Optional[Mapping[str, Any]] = None,
        bounds: Optional[Mapping[str, Any]] = None,
        bounds_mode: Optional[str] = None,
        solver: Optional[Mapping[str, Any]] = None,
        multistart: Optional[Mapping[str, Any]] = None,
        strategy: str = "single",
        seeds: Optional[Sequence[float]] = None,
        vh: Optional[Mapping[str, Any]] = None,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> ConcDepResult:
        """Aggregate per-curve Tms and regress ``1/Tm vs ln(C_T/f)``.

        For each curve: ``raw_tm(raw=raw)``, ``vant_hoff(vh=vh)`` on those
        baselines, then ``fit(...)`` with the remaining arguments, which
        override every curve's settings exactly as in ``SingleCurve.fit``.
        E.g. ``fit_init={"seed_from": "user", "dH_init": -100,
        "dS_init": -0.28}`` seeds every per-curve fit with your own values
        instead of van't Hoff.

        Parameters
        ----------
        precomputed_tms : optional {curve_name: {"Tm_raw": ..., "Tm_vH": ...,
            "Tm_fit": ..., "base_b": (m, b), "base_ub": (m, b), "converged": bool}}
            (floats, or None where unknown). When present for a given curve name, the expensive per-curve
            ``raw_tm`` / ``vant_hoff`` / ``fit`` pipeline is skipped and the
            supplied values are used directly. Fields other than the three
            ``Tm_*`` values are optional. Intended for the web UI, which
            already has these results cached from prior single-curve runs
            and doesn't want them recomputed on every concentration-mode
            switch. Callers are responsible for making sure the cached
            values were produced with the same baseline windows as the
            current curves.
        """
        if self.struct_type == "monomer":
            return ConcDepResult(
                struct_type = self.struct_type,
                error="concentration-series van't Hoff is not applicable to "
                      "monomers — Tm is concentration-independent for "
                      "intramolecular folding",
            )

        per_curve: list[ConcentrationCurve] = []
        skipped: list[dict] = []
        precomputed_tms = dict(precomputed_tms) if precomputed_tms else {}

        for c in self.curves:
            if not np.isfinite(c.oligoC) or c.oligoC <= 0:
                skipped.append({"name": c.name,
                                "reason": f"non-positive concentration {c.oligoC}"})
                continue
            try:
                TT, y = c.windowed_data()
            except ValueError as e:
                skipped.append({"name": c.name, "reason": str(e)})
                continue

            cache = precomputed_tms.get(c.name)
            if cache is not None:
                # Trust the caller-supplied per-curve Tms; skip the fit pipeline.
                per_curve.append(ConcentrationCurve(
                    name       = c.name,
                    oligoC     = c.oligoC,
                    c0         = c.c0,
                    Tm_raw     = cache.get("Tm_raw"),
                    Tm_vH      = cache.get("Tm_vH"),
                    Tm_fit     = cache.get("Tm_fit"),
                    lnCT       = float(np.log(c.c_eff)),
                    T_used     = TT,
                    signal     = y,
                    signal_all = c.signal,
                    base_b     = cache.get("base_b"),
                    base_ub    = cache.get("base_ub"),
                    oligoC_B   = c.oligoC_B,
                    c_eff      = c.c_eff,
                    converged  = cache.get("converged"),
                ))
                continue

            raw_res = c.raw_tm(raw=raw)
            if not raw_res.ok:
                skipped.append({"name": c.name,
                                "reason": f"raw Tm extraction failed: {raw_res.error}"})
                continue

            if (raw_res.Tm is None or not np.isfinite(raw_res.Tm)
                    or not (c.fold_low_T <= raw_res.Tm <= c.unf_high_T)):
                skipped.append({"name": c.name,
                                "reason": f"raw Tm out of window ({raw_res.Tm})"})
                continue

            vh_res = c.vant_hoff(base_b=raw_res.base_b, base_ub=raw_res.base_ub, vh=vh)
            fit = c.fit(vh_fit=vh_res, fit_init=fit_init, bounds=bounds,
                        bounds_mode=bounds_mode, solver=solver,
                        multistart=multistart, strategy=strategy, seeds=seeds)

            per_curve.append(ConcentrationCurve(
                name       = c.name,
                oligoC     = c.oligoC,
                c0         = c.c0,
                Tm_raw     = raw_res.Tm,
                Tm_vH      = vh_res.Tm if vh_res.ok else None,
                Tm_fit     = fit.Tm    if fit.ok    else None,
                lnCT       = float(np.log(c.c_eff)),
                T_used     = TT,
                signal     = y,
                signal_all = c.signal,
                base_b     = raw_res.base_b,
                base_ub    = raw_res.base_ub,
                oligoC_B   = c.oligoC_B,
                c_eff      = c.c_eff,
                converged  = fit.converged if fit.ok else None,
            ))


        if len(per_curve) < 2:
            error = (f"need at least 2 valid (Tm, C_T) points; "
                            f"got {len(per_curve)}")
        else:
            error = None

        series_raw = self._regress(per_curve, "raw")
        series_vh  = self._regress(per_curve, "vh")
        series_fit = self._regress(per_curve, "fit")

        result = ConcDepResult(
            ok                 = error is None,
            struct_type        = self.struct_type,
            self_complementary = self.self_complementary,
            per_curve          = per_curve,
            skipped            = skipped,
            series_raw         = series_raw,
            series_vh          = series_vh,
            series_fit         = series_fit,
            error              = error
        )
        return result


    def _regress(self, per_curve, method: str) -> ConcentrationSeries:
        """Regress 1/Tm vs ln c_eff (= ln(C_T/f) for equal strands) for one
        Tm-extraction method."""
        tm_K, ct_M = [], []
        for p in per_curve:
            v = p.Tm_K(method)
            if v is not None and np.isfinite(v) and v > 0:
                tm_K.append(v)
                ct_M.append(p.c_eff)
        if len(tm_K) < 2:
            return ConcentrationSeries(
                method=method,
                error=f"only {len(tm_K)} usable Tm value(s); need at least 2")
        try:
            vh = methods.vant_hoff_concentration(np.array(tm_K), c_eff=np.array(ct_M))
        except ValueError as e:
            return ConcentrationSeries(method=method, error=str(e))

        ln_ct_line = np.linspace(float(vh["ln_ct"].min()), float(vh["ln_ct"].max()), 50)
        return ConcentrationSeries(
            method      = method,
            ok          = True,
            n           = len(tm_K),
            dH          = vh["dH"],
            dS          = vh["dS"],
            dG          = vh["dG_37"],
            r_squared   = vh["r_squared"],
            slope       = vh["slope"],
            intercept   = vh["intercept"],
            ln_ct       = vh["ln_ct"],
            inv_tm      = vh["inv_tm"],
            ln_ct_line  = ln_ct_line,
            inv_tm_line = vh["slope"] * ln_ct_line + vh["intercept"],
            SE_dH       = vh["SE_dH"],
            SE_dS       = vh["SE_dS"],
            SE_dG       = vh["SE_dG"],
            cov_dH_dS   = vh["cov_dH_dS"],
            SE_note     = vh["SE_note"],
        )



__all__ = [
    "SingleCurve",
    "MultiCurves",
]
