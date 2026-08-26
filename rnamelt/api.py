import numpy as np
from typing import Any, Mapping, Optional, Union, Iterator
from collections.abc import Sequence
from rnamelt import methods, constants, functions
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


class SingleCurve:
    def __init__(
        self,
        signal,
        temps,
        *,
        name: str = "unnamed",
        oligo: float = 0.5,
        struct_type: str = "heterodimer",
        T_low: Optional[float] = None,
        T_high: Optional[float] = None,
        bl_lower_offset: float = 10.0,
        bl_upper_offset: float = 10.0,
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
        if temps.size < 4:
            raise ValueError(f"need at least 4 data points, got {temps.size}")
        if struct_type not in constants.STRAND_STOICHIOMETRY:
            raise ValueError(
                f"unknown struct_type {struct_type!r}; valid: "
                f"{sorted(constants.STRAND_STOICHIOMETRY)}"
            )

        # Drop NaN signal points once — every consumer downstream can assume
        # clean arrays.
        keep = ~np.isnan(signal)
        if keep.sum() < 4:
            raise ValueError(
                f"need at least 4 non-NaN signal points, got {int(keep.sum())}"
            )

        self.name            = name
        self.temps           = temps[keep]
        self.signal          = signal[keep]
        self.oligo           = oligo
        self.struct_type     = struct_type
        #self.T_low           = T_low
        #self.T_high          = T_high
        self.bl_lower_offset = bl_lower_offset
        self.bl_upper_offset = bl_upper_offset
        self.solver          = dict(solver)   if solver   else None
        self.vh              = dict(vh)       if vh       else None
        self.fit_init        = dict(fit_init) if fit_init else None

        #valid = ~np.isnan(self.signal)
        #T_all      = self.temps[valid]
        #signal_all = self.signal[valid]

        T_min, T_max = float(self.temps.min()), float(self.temps.max())

        self.T_low  = T_low if T_low is not None else T_min
        self.T_high = T_high if T_high is not None else T_max

        pos_start = np.where(self.temps == self.T_low)[0][0]
        pos_end   = np.where(self.temps == self.T_high)[0][0]+1

        #self.TT = T_all[pos_start:pos_end]
        #self.used_data = signal_all[pos_start:pos_end]
        #self.tmRaw = self.__tmRaw()

    def __repr__(self) -> str:
        return (
            f"SingleCurve(name={self.name!r}, n={self.temps.size}, "
            f"T=[{self.temps[0]:.1f}, {self.temps[-1]:.1f}]°C, "
            f"oligo={self.oligo}µM, struct_type={self.struct_type!r})"
        )

    @property
    def c0(self) -> float:
        """Total strand concentration in M (oligo µM × stoichiometry)."""
        return 1e-6 * self.oligo * constants.STRAND_STOICHIOMETRY[self.struct_type]

    def windowed_data(self) -> tuple[np.ndarray, np.ndarray]:
        mask = (self.temps >= self.T_low) & (self.temps <= self.T_high)
        if mask.sum() < 10:
            raise ValueError(
                f"need at least 10 data points in the window "
                f"[{self.T_low}, {self.T_high}]°C, got {int(mask.sum())}"
            )
        return self.temps[mask], self.signal[mask]

    def raw_tm(self) -> RawTm:
        r_base_b_maxT=self.T_low+self.bl_lower_offset
        r_base_ub_minT=self.T_high-self.bl_upper_offset

        TT, y = self.windowed_data()

        try:
            T_m_raw, y_r, base_b_r, base_ub_r, base_med_r = methods.T_m_ds_raw(
                TT, y,
                baseline_bound_maxT=r_base_b_maxT,
                baseline_unbound_minT=r_base_ub_minT,
                #debug=True #NOTE: Should we add debugging?
            )
        except Exception as e:
            return RawTm(ok=False, error=str(e))

        return RawTm(
            ok = True,
            Tm = T_m_raw,
            signal = y_r,
            base_b_r = base_b_r,
            base_ub_r = base_ub_r,
            base_med_r = base_med_r
        )


    #TODO: Need to check what prrams are really needed.
    def vantHoff(self, *, raw: Optional[RawTm] = None) -> VanHoffFit:
        if self.struct_type == "monomer":
            return VanHoffFit(
                ok=False,
                error="van't Hoff linearisation is not defined for monomers "
                      "— Tm is concentration-independent for intramolecular folding",
            )
        raw = raw if raw is not None else self.raw_tm()
        if not raw.ok:
            return VanHoffFit(ok=False, error=f"baselines unavailable: {raw.error}")

        TT, y = self.windowed_data()

        try:
            T_m_vH, dG_37_vH, dH_vH, dS_vH, t1, K, xdata, ydata, fit_vh = methods.vantHoff(
                TT, y,
                *raw.base_b_r,
                *raw.base_ub_r,
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
            fit_params = fit_vh,
        )

    def fit(self, *, vh: Optional[VanHoffFit] = None) -> SingleFit:
        TT, y = self.windowed_data()
        init = methods._fit_init_kwargs(self.fit_init)

        need_seed = init["dH_init"] is None or init["dS_init"] is None
        if vh is None and need_seed:
            vh = self.vantHoff()
        vh_dH = vh.dH if (vh is not None and vh.ok) else None
        vh_dS = vh.dS if (vh is not None and vh.ok) else None

        try:
            dG_37, dH, dS, Tm, _y, base_b, base_ub, base_med = methods.fit_full_function(
                TT, y,
                c0         = self.c0,
                dH_init    = _seed(init["dH_init"], vh_dH, _DH_SEED_RANGE, _DH_FALLBACK),
                dS_init    = _seed(init["dS_init"], vh_dS, _DS_SEED_RANGE, _DS_FALLBACK),
                lin_init   = init["lin_init"],
                b1_init    = init["b1_init"],
                b2_init    = init["b2_init"],
                structType = self.struct_type,
                solver     = self.solver,
            )
        except Exception as e:
            return SingleFit(
                ok         = False,
                error      = str(e),
            )
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
            base_b     = tuple(float(v) for v in base_b),
            base_ub    = tuple(float(v) for v in base_ub),
            base_med   = tuple(float(v) for v in base_med),
            curve_fit  = curve_fit,
            derivative = -np.gradient(y, TT),
        )

class MultiCurves(Sequence):
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
        return [c.oligo for c in self.curves]

    @property
    def self_complementary(self) -> bool:
        """True for homodimers, where C_T enters the model undivided."""
        return self.struct_type == "homodimer"

    def global_fit(self) -> GlobalFit:
        if len(self.curves) < 2:
            return GlobalFit(
                error=f"global fit needs at least 2 curves, got {len(self.curves)}")

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
            base_b  = tuple(float(v) for v in bl_lo[i])
            base_ub = tuple(float(v) for v in bl_hi[i])
            curves.append(GlobalFitCurve(
                name       = c.name,
                oligoC     = c.oligo,
                c0         = cs[i],
                Tm_fit     = float(Tms[i]),
                base_b     = base_b,
                base_ub    = base_ub,
                base_med   = tuple(float(v) for v in bl_med[i]),
                curve      = np.array([
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
            if not np.isfinite(c.oligo) or c.oligo <= 0:
                skipped.append({"name": c.name,
                                "reason": f"non-positive concentration {c.oligo}"})
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
            if not np.isfinite(raw.Tm) or not (c.T_low <= raw.Tm <= c.T_high):
                skipped.append({"name": c.name,
                                "reason": f"raw Tm out of window ({raw.Tm})"})
                continue

            vh  = c.vantHoff(raw=raw)
            fit = c.fit(vh=vh)

            per_curve.append(ConcentrationCurve(
                name       = c.name,
                oligoC     = c.oligo,
                c0         = c.c0,
                Tm_raw     = raw.Tm,
                Tm_vH      = vh.Tm  if vh.ok  else None,
                Tm_fit     = fit.Tm if fit.ok else None,
                lnCT       = float(np.log(c.c0 / f_factor)),
                T_used     = TT,
                signal     = y,
                signal_all = c.signal,
                base_b     = raw.base_b_r,
                base_ub    = raw.base_ub_r,
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
            dG_37       = vh["dG_37"],
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

