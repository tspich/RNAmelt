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

# Fallback ΔH / ΔS seeds for the non-linear fit when no van't Hoff
# estimate is available (monomers, or a failed linearisation), plus the
# plausibility window a vH estimate must fall in to be used as a seed.
_DH_FALLBACK = -80.0
_DS_FALLBACK = -0.2
_DH_SEED_RANGE = (-150.0, 0.0)
_DS_SEED_RANGE = (-5.0, 0.0)

def _seed(user_value, vh_value, valid_range, fallback):
    """Pick a fit seed: explicit user value > plausible vH estimate > fallback."""
    if user_value is not None:
        return user_value
    lo, hi = valid_range
    if vh_value is not None and np.isfinite(vh_value) and lo < vh_value < hi:
        return float(vh_value)
    return fallback

def _pair(v) -> tuple[float, float]:
    """Coerce a (slope, intercept) baseline to a plain float 2-tuple."""
    a, b = v
    return float(a), float(b)


class SingleCurve:
    def __init__(
        self,
        temps,
        signal,
        *,
        name: str = "unnamed",
        oligoC: float = 0.5,
        struct_type: str = "heterodimer",
        T_low: Optional[float] = None,
        T_high: Optional[float] = None,
        bl_lower_offset: float = 10.0,
        bl_upper_offset: float = 10.0,
        baseline_mode: str = "manual",
        baseline_kwargs: Optional[Mapping[str, Any]] = None,
        solver: Optional[Mapping[str, Any]] = None,
        vh: Optional[Mapping[str, Any]] = None,
        fit_init: Optional[Mapping[str, Any]] = None,
    ):
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
        if baseline_mode not in ("manual", "auto"):
            raise ValueError(
                f"unknown baseline_mode {baseline_mode!r}; "
                "valid: 'manual', 'auto'"
            )

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
        self.struct_type     = struct_type
        self.bl_lower_offset = bl_lower_offset
        self.bl_upper_offset = bl_upper_offset
        self.baseline_mode   = baseline_mode
        self.baseline_kwargs = dict(baseline_kwargs) if baseline_kwargs else {}
        self._auto_windows: Optional[tuple[float, float, float, float]] = None
        self.solver          = dict(solver)   if solver   else None
        self.vh              = dict(vh)       if vh       else None
        self.fit_init        = dict(fit_init) if fit_init else None

        T_min, T_max = float(self.temps.min()), float(self.temps.max())

        self.T_low  = T_low if T_low is not None else T_min
        self.T_high = T_high if T_high is not None else T_max

    def __repr__(self) -> str:
        return (
            f"SingleCurve(name={self.name!r}, n={self.temps.size}, "
            f"T=[{self.temps[0]:.1f}, {self.temps[-1]:.1f}]°C, "
            f"oligoC={self.oligoC}µM, struct_type={self.struct_type!r})"
        )

    @property
    def c0(self) -> float:
        """Total strand concentration in M (oligo µM × stoichiometry)."""
        return 1e-6 * self.oligoC * constants.STRAND_STOICHIOMETRY[self.struct_type]

    def windowed_data(self) -> tuple[np.ndarray, np.ndarray]:
        mask = (self.temps >= self.T_low) & (self.temps <= self.T_high)
        if mask.sum() < 10:
            raise ValueError(
                f"need at least 10 data points in the window "
                f"[{self.T_low}, {self.T_high}]°C, got {int(mask.sum())}"
            )
        return self.temps[mask], self.signal[mask]

    def auto_baselines(self) -> tuple[float, float, float, float]:
        """(fold_low_T, fold_high_T, unf_low_T, unf_high_T) from
        ``auto_baseline.find_baselines`` run on the windowed data. Cached
        per instance; forwards ``self.baseline_kwargs`` to the detector."""
        if self._auto_windows is None:
            TT, y = self.windowed_data()
            fl, fh, ul, uh = auto_baseline.find_baselines(
                TT, y, **self.baseline_kwargs
            )
            self._auto_windows = (float(fl), float(fh), float(ul), float(uh))
        return self._auto_windows

    def raw_tm(self) -> RawTm:
        TT, y = self.windowed_data()

        if self.baseline_mode == "auto":
            try:
                fl, fh, ul, uh = self.auto_baselines()
            except Exception as e:
                return RawTm(ok=False, error=f"auto baselines failed: {e}")
            kw = dict(
                baseline_bound_minT=fl,
                baseline_bound_maxT=fh,
                baseline_unbound_minT=ul,
                baseline_unbound_maxT=uh,
            )
        else:
            kw = dict(
                baseline_bound_maxT=self.T_low + self.bl_lower_offset,
                baseline_unbound_minT=self.T_high - self.bl_upper_offset,
            )

        try:
            T_m_raw, y_r, base_b, base_ub, base_med = methods.T_m_ds_raw(
                TT, y, **kw
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


    #TODO: Need to check what prrams are really needed.
    def vantHoff(self, *,
                 base_b: Optional[tuple[float, float]] = None,
                 base_ub: Optional[tuple[float, float]] = None) -> VanHoffFit:

        if self.struct_type == "monomer":
            return VanHoffFit(
                ok=False,
                error="van't Hoff linearisation is not defined for monomers "
                      "— Tm is concentration-independent for intramolecular folding",
            )

        raw = self.raw_tm()
        base_b = base_b if base_b is not None else raw.base_b
        base_ub = base_ub if base_ub is not None else raw.base_ub

        if not raw.ok or base_b is None or base_ub is None:
            return VanHoffFit(ok=False, error=f"baselines unavailable: {raw.error}")

        TT, y = self.windowed_data()

        try:
            T_m_vH, dG_37_vH, dH_vH, dS_vH, t1, K, xdata, ydata, fit_vh = methods.vantHoff(
                TT, y,
                *base_b,
                *base_ub,
                self.c0,
                structType = self.struct_type,
                **methods._vh_kwargs(self.vh),
            )

        except Exception as e:
            return VanHoffFit(
                ok         = False,
                error      = str(e),
            )

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
            fit_params = _pair(fit_vh),
        )

    def fit(
        self,
        *,
        vh: Optional[VanHoffFit] = None,
        bounds_mode: str = "physical",
        strategy: str = "single",
        seeds: Optional[Sequence[float]] = None,
    ) -> SingleFit:
        """Non-linear two-state fit.

        Parameters
        ----------
        vh : VanHoffFit, optional
            Pre-computed van't Hoff result whose (dH, dS) are used to seed
            the fit when ``fit_init`` leaves either as ``None``. If not given,
            it is computed on demand from the raw baselines.
        bounds_mode : {"loose", "unbounded", "physical", "positive"}
            Bounds on the four baseline parameters. See
            ``methods._fit_bounds`` for the exact box.
              * ``"loose"``    (default) — folded slope ≥ 0, unfolded free
              * ``"unbounded"`` — all four free
              * ``"physical"`` — |m₁|, |m₂| ≤ 5 · (ΔS/ΔT); ideal against
                the "baseline eats the transition" failure mode
              * ``"positive"`` — folded and unfolded slopes both ≥ 0
        strategy : {"single", "multistart"}
            ``"single"`` runs one non-linear fit (behaviour matches the
            historical default). ``"multistart"`` runs one fit per seed in
            ``seeds`` and keeps the best-by-residual result whose Tm sits
            inside the data window; use this when the vH linearisation may
            land the solver in a shallow local minimum (e.g. cut windows
            that clip the folded plateau).
        seeds : sequence of float, optional
            Only used when ``strategy="multistart"``. Each entry is a
            ``dH_init`` (kcal/mol). ``dS_init`` per seed is set to
            ``dH_init / 350`` — a rough van't Hoff scaling assuming
            Tm ≈ 350 K. Defaults to seeds spanning the physical range
            plus the FWHM-of-dS/dT estimate.
        """
        TT, y = self.windowed_data()
        init = methods._fit_init_kwargs(self.fit_init)

        need_seed = init["dH_init"] is None or init["dS_init"] is None
        if vh is None and need_seed:
            vh = self.vantHoff()
        vh_dH = vh.dH if (vh is not None and vh.ok) else None
        vh_dS = vh.dS if (vh is not None and vh.ok) else None

        def _run_one(dH_seed_override=None, dS_seed_override=None):
            try:
                dG_37, dH, dS, Tm, ym, base_b, base_ub, base_med = methods.fit_full_function(
                    TT, y,
                    c0          = self.c0,
                    dH_init     = _seed(dH_seed_override if dH_seed_override is not None
                                        else init["dH_init"], vh_dH, _DH_SEED_RANGE, _DH_FALLBACK),
                    dS_init     = _seed(dS_seed_override if dS_seed_override is not None
                                        else init["dS_init"], vh_dS, _DS_SEED_RANGE, _DS_FALLBACK),
                    lin_init    = init["lin_init"],
                    b1_init     = init["b1_init"],
                    b2_init     = init["b2_init"],
                    structType  = self.struct_type,
                    bounds_mode = bounds_mode,
                    solver      = self.solver,
                )
            except Exception as e:
                return SingleFit(ok=False, error=str(e))
            curve_fit = np.array([
                functions.full_function(t, dH, dS, *base_b, *base_ub,
                                        c0=self.c0, structType=self.struct_type)
                for t in TT
            ])
            return SingleFit(
                ok         = True,
                dH         = float(dH),
                dS         = float(dS),
                dG         = float(dG_37),
                Tm         = float(Tm),
                ym         = float(ym),
                base_b     = _pair(base_b),
                base_ub    = _pair(base_ub),
                base_med   = _pair(base_med),
                curve_fit  = curve_fit,
                derivative = -np.gradient(y, TT),
            )

        if strategy == "single":
            return _run_one()

        if strategy != "multistart":
            raise ValueError(f"unknown strategy {strategy!r}; "
                             "valid: 'single', 'multistart'")

        if seeds is None:
            fwhm_seed = methods.dH_seed_from_fwhm(
                TT, y, struct_type=self.struct_type, folding=True,
            )
            seeds = [-40.0, -80.0, -120.0, -160.0]
            if fwhm_seed is not None:
                # add the FWHM seed AND a stronger multiple (FWHM under-predicts)
                seeds = [fwhm_seed, 1.5 * fwhm_seed, *seeds]

        best: Optional[SingleFit] = None
        best_cost = np.inf
        T_lo_win, T_hi_win = float(TT.min()), float(TT.max())
        for dH_seed in seeds:
            dS_seed = dH_seed / 350.0 #NOTE: Does that make sense?
            f = _run_one(dH_seed_override=dH_seed, dS_seed_override=dS_seed)
            if not f.ok:
                continue
            rms = float(np.sqrt(np.mean((y - f.curve_fit) ** 2)))
            physical = (T_lo_win - 3.0 <= f.Tm <= T_hi_win + 3.0) and (f.dH < -10.0)
            cost = rms if physical else rms * 1e6
            if cost < best_cost:
                best_cost, best = cost, f
        return best if best is not None else SingleFit(
            ok=False, error="all multistart seeds failed to converge"
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
        """Index by position, slice (→ `MeltSet`), or curve name."""
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

    def global_fit(self) -> GlobalFit:
        head = self.curves[0]
        init = methods._fit_init_kwargs(head.fit_init)

        Ts, ys, cs = [], [], []
        for c in self.curves:
            try:
                T, y = c.windowed_data()
            except ValueError as e:
                return GlobalFit(error=f"curve {c.name!r}: {e}")
            Ts.append(T)
            ys.append(y)
            cs.append(c.c0)

        #NOTE: Make sure all the curves doesn't need to have the same number of
        #      points.

        try:
            dG_37, dH, dS, Tms, _y, bl_lo, bl_hi, bl_med = methods.fit_full_function_multi(
                Ts, ys,
                cs         = cs,
                dH_init    = _seed(init["dH_init"], None, _DH_SEED_RANGE, _DH_FALLBACK),
                dS_init    = _seed(init["dS_init"], None, _DS_SEED_RANGE, _DS_FALLBACK),
                lin_init   = init["lin_init"],
                structType = self.struct_type,
                solver     = head.solver,
            )
        except Exception as e:
            return GlobalFit(error=f"global fit failed: {e}")

        curves = []
        for i, c in enumerate(self.curves):
            base_b  = _pair(bl_lo[i])
            base_ub = _pair(bl_hi[i])
            curves.append(GlobalFitCurve(
                name       = c.name,
                oligoC     = c.oligoC,
                c0         = cs[i],
                Tm         = float(Tms[i]),
                base_b     = base_b,
                base_ub    = base_ub,
                base_med   = _pair(bl_med[i]),
                fit        = np.array([
                    functions.full_function(t, dH, dS, *base_b, *base_ub,
                                            c0=cs[i], structType=self.struct_type)
                    for t in Ts[i]
                ]),
                T_used     = Ts[i],
                signal     = ys[i],
                signal_all = c.signal,
            ))

        return GlobalFit(
            dH     = float(dH),
            dS     = float(dS),
            dG     = float(dG_37),
            T_used = Ts,
            curves = curves,
        )

    def conc_dependency(self) -> ConcDepResult:
        if self.struct_type == "monomer":
            return ConcDepResult(
                struct_type = self.struct_type,
                error="concentration-series van't Hoff is not applicable to "
                      "monomers — Tm is concentration-independent for "
                      "intramolecular folding",
            )

        f_factor = 1.0 if self.self_complementary else 4.0
        per_curve: list[ConcentrationCurve] = []
        skipped: list[dict] = []

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

            raw = c.raw_tm()
            if not raw.ok:
                skipped.append({"name": c.name,
                                "reason": f"raw Tm extraction failed: {raw.error}"})
                continue

            if raw.Tm is None or not np.isfinite(raw.Tm) or not (c.T_low <= raw.Tm <= c.T_high):
                skipped.append({"name": c.name,
                                "reason": f"raw Tm out of window ({raw.Tm})"})
                continue

            vh  = c.vantHoff(base_b=raw.base_b, base_ub=raw.base_ub)
            fit = c.fit(vh=vh)

            per_curve.append(ConcentrationCurve(
                name       = c.name,
                oligoC     = c.oligoC,
                c0         = c.c0,
                Tm_raw     = raw.Tm,
                Tm_vH      = vh.Tm  if vh.ok  else None,
                Tm_fit     = fit.Tm if fit.ok else None,
                lnCT       = float(np.log(c.c0 / f_factor)),
                T_used     = TT,
                signal     = y,
                signal_all = c.signal,
                base_b     = raw.base_b,
                base_ub    = raw.base_ub,
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
        """Regress 1/Tm vs ln(C_T/f) for one Tm-extraction method."""
        tm_K, ct_M = [], []
        for p in per_curve:
            v = p.Tm_K(method)
            if v is not None and np.isfinite(v) and v > 0:
                tm_K.append(v)
                ct_M.append(p.c0)
        if len(tm_K) < 2:
            return ConcentrationSeries(
                method=method,
                error=f"only {len(tm_K)} usable Tm value(s); need at least 2")
        try:
            vh = methods.vant_hoff_concentration(
                np.array(tm_K), np.array(ct_M),
                self_complementary=self.self_complementary,
            )
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
        )



__all__ = [
    "SingleCurve",
    "MultiCurves",
    #"SOLVER_DEFAULTS",
    #"VH_DEFAULTS",
    #"FIT_INIT_DEFAULTS",
]

#NOTE: Next we want another class, which is construct from a set of
#      SingleCurve and on which one can perform either global fit or Vant Hoff
#      over the concentration.
#      Are the dataclass in results really needed, what are the advantage of
#      it?
#      Important is to be able to rerun single analysis with different
#      parameters. How should be the multi class handled? Automatic update on
#      change of the single element (is that even possible?) or one need to
#      remember to update if changes happened?

