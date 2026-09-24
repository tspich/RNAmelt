"""Typed results returned by `rnamelt.api`.

Every analysis returns a NamedTuple from this module, built directly
from the numeric output of `rnamelt.methods` — there is no intermediate
dict.

    SingleCurve.raw_tm()          -> RawTm
    SingleCurve.vant_hoff()        -> VanHoffFit
    SingleCurve.fit()             -> SingleFit
    MultiCurves.global_fit()      -> GlobalFit      (holds GlobalFitCurve)
    MultiCurves.conc_dependency() -> ConcDepResult  (holds ConcentrationCurve
                                                     and ConcentrationSeries)

Failure convention
------------------
A failed analysis is returned, not raised: `ok` is False, `error` holds
the reason, and every other field keeps its default. Check `ok` before
reading values. One failed step never aborts the others — a monomer has
a usable raw Tm alongside a van't Hoff that reports why it is undefined.

Every result, `GlobalFit` and `ConcDepResult` included, carries `ok`.

Units
-----
    dH         kcal/mol
    dS         kcal/(mol K)
    dG         kcal/mol at 37 C
    Tm         degrees C — ConcentrationCurve also exposes Tm_*_K in Kelvin
    c0         M, total strand concentration
    oligoC     uM, per-strand concentration
    baselines  (slope, intercept), signal units per degree C
"""

from __future__ import annotations
from collections.abc import Sequence

from typing import NamedTuple, Optional

import numpy as np


class RawTm(NamedTuple):
    ok:       bool                          = False
    error:    Optional[str]                 = None
    Tm:       Optional[float]               = None
    ym:       Optional[np.ndarray]          = None
    base_b:   Optional[tuple[float, float]] = None
    base_ub:  Optional[tuple[float, float]] = None
    base_med: Optional[tuple[float, float]] = None

class VanHoffFit(NamedTuple):
    """Linearised van't Hoff regression (ln K vs 1/T).

    `fit_params` is the `(slope, intercept)` tuple of the linear fit in
    the linearised x-axis (1000/(T-T0)) — multiply by `slope`·x + intercept
    to recover the fitted ln(K) values.
    """
    ok:         bool                          = False
    error:      Optional[str]                 = None
    dH:         Optional[float]               = None
    dS:         Optional[float]               = None
    dG:         Optional[float]               = None
    Tm:         Optional[float]               = None
    t1:         Optional[np.ndarray]          = None
    K:          Optional[np.ndarray]          = None
    xdata:      Optional[np.ndarray]          = None
    ydata:      Optional[np.ndarray]          = None
    fit_params: Optional[tuple[float, float]] = None

    def __repr__(self) -> str:
        if self.ok:
            return (f"VanHoffFit(dH={self.dH}, dS={self.dS}, "
                    f"dG={self.dG}, Tm={self.Tm})")
        return f"VanHoffFit(ok=False, error={self.error})"

class SingleFit(NamedTuple):
    ok:         bool                          = False
    error:      Optional[str]                 = None
    dH:         Optional[float]               = None
    dS:         Optional[float]               = None
    dG:         Optional[float]               = None
    Tm:         Optional[float]               = None
    ym:         Optional[float]               = None
    base_b:     Optional[tuple[float, float]] = None
    base_ub:    Optional[tuple[float, float]] = None
    base_med:   Optional[tuple[float, float]] = None
    curve_fit:  Optional[np.ndarray]          = None
    converged:  bool                          = True   # False: stopped at max_nfev
    # Starting values actually used: {"dH", "dS", "dH_from", "dS_from"},
    # *_from in {"user", "vh", "fallback", "multistart"}.
    seed:       Optional[dict]                = None

    def __repr__(self) -> str:
        if self.ok:
            return (f"SingleFit(dH={self.dH}, dS={self.dS}, "
                f"dG={self.dG}, Tm={self.Tm}"
                + ("" if self.converged else ", converged=False") + ")")
        return f"SingleFit(ok=False, error={self.error})"

class GlobalFitCurve(NamedTuple):
    """One curve inside a shared-ΔH/ΔS joint fit — baselines stay per-curve."""
    name:       str
    oligoC:     Optional[float]               = None
    c0:         Optional[float]               = None
    Tm:         Optional[float]               = None
    base_b:     Optional[tuple[float, float]] = None
    base_ub:    Optional[tuple[float, float]] = None
    base_med:   Optional[tuple[float, float]] = None
    fit:        Optional[np.ndarray]          = None
    T_used:     Optional[np.ndarray]          = None
    signal:     Optional[np.ndarray]          = None
    signal_all: Optional[np.ndarray]          = None

class GlobalFit(NamedTuple):
    """Return value of `MultiCurves.global_fit()` — one ΔH/ΔS for all curves."""
    ok:      bool                     = False
    dH:      Optional[float]          = None      # kcal/mol
    dS:      Optional[float]          = None      # kcal/(mol·K)
    dG:      Optional[float]          = None      # kcal/mol at 37 °C
    T_used:  Sequence[np.ndarray]     = ()      # one window per curve
    curves:  Sequence[GlobalFitCurve] = ()
    error:   Optional[str]            = None
    converged: bool                   = True   # False: stopped at max_nfev
    seed:      Optional[dict]         = None   # see SingleFit.seed

    def __repr__(self) -> str:
        if self.error:
            return f"GlobalFit(error={self.error})"
        return (f"GlobalFit(dH={self.dH}, dS={self.dS}, "
            f"dG={self.dG}, curves=GlobalFitCurve({len(self.curves)}, "
            f"oligoC={[c.oligoC for c in self.curves]}, "
            f"Tm_fit={[c.Tm for c in self.curves]}))")

class ConcentrationCurve(NamedTuple):
    """One curve in a concentration series — Tm by all three extractions.

    `*_K` fields are the same Tm in Kelvin, which is what the 1/Tm
    regression consumes.
    """
    name:       str
    oligoC:     Optional[float]               = None
    c0:         Optional[float]               = None
    Tm_raw:     Optional[float]               = None
    Tm_vH:      Optional[float]               = None
    Tm_fit:     Optional[float]               = None
    lnCT:       Optional[float]               = None
    T_used:     Optional[np.ndarray]          = None
    signal:     Optional[np.ndarray]          = None
    signal_all: Optional[np.ndarray]          = None
    base_b:     Optional[tuple[float, float]] = None
    base_ub:    Optional[tuple[float, float]] = None

    @staticmethod
    def _to_K(T_C: Optional[float]) -> Optional[float]:
        """°C → K, passing through None / non-finite as None."""
        from rnamelt import constants
        if T_C is None or not np.isfinite(T_C):
            return None
        return float(T_C - constants.T0)

    @property
    def Tm_raw_K(self) -> Optional[float]:
        """Baseline-intersection Tm in Kelvin."""
        return self._to_K(self.Tm_raw)

    @property
    def Tm_vH_K(self) -> Optional[float]:
        """van't Hoff Tm in Kelvin."""
        return self._to_K(self.Tm_vH)

    @property
    def Tm_fit_K(self) -> Optional[float]:
        """Full-fit Tm in Kelvin."""
        return self._to_K(self.Tm_fit)

    def Tm_K(self, method: str) -> Optional[float]:
        """Tm in Kelvin for `method` in {"raw", "vh", "fit"}."""
        return {
            "raw": self.Tm_raw_K,
            "vh":  self.Tm_vH_K,
            "fit": self.Tm_fit_K,
        }[method]

class ConcentrationSeries(NamedTuple):
    """1/Tm vs ln(C_T/f) linear regression for one Tm-extraction method."""
    method:      str
    ok:          bool                 = False
    error:       Optional[str]        = None
    n:           Optional[int]        = None
    dH:          Optional[float]      = None
    dS:          Optional[float]      = None
    dG:          Optional[float]      = None
    r_squared:   Optional[float]      = None
    slope:       Optional[float]      = None
    intercept:   Optional[float]      = None
    ln_ct:       Optional[np.ndarray] = None
    inv_tm:      Optional[np.ndarray] = None
    ln_ct_line:  Optional[np.ndarray] = None
    inv_tm_line: Optional[np.ndarray] = None

class ConcDepResult(NamedTuple):
    """Return value of `MultiCurves.conc_dependency()`. ``ok`` is True when
    at least two curves gave a usable Tm; each ``series_*`` has its own
    ``ok`` / ``error``."""
    ok:                 bool                         = False
    struct_type:        str                          = "heterodimer"
    self_complementary: bool                         = False
    per_curve:          Sequence[ConcentrationCurve] = ()
    skipped:            Sequence[dict]               = ()
    series_raw:         ConcentrationSeries          = ConcentrationSeries(method='raw')
    series_vh:          ConcentrationSeries          = ConcentrationSeries(method='vh')
    series_fit:         ConcentrationSeries          = ConcentrationSeries(method='fit')
    error:              Optional[str]                = None

    def __repr__(self) -> str:
        if self.error:
            return f"ConcDepResult(error={self.error})"
        return (f"ConcDepResult(struct_type={self.struct_type}, "
            f"self_complementary={self.self_complementary}, "
            f"per_curve=ConcentrationCurve({len(self.per_curve)}, "
            f"oligoC={[c.oligoC for c in self.per_curve]}, "
            f"Tm_raw={[c.Tm_raw for c in self.per_curve]}, "
            f"Tm_vH={[c.Tm_vH for c in self.per_curve]}, "
            f"Tm_fit={[c.Tm_fit for c in self.per_curve]}))")


