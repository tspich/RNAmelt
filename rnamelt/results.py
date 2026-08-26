"""Typed result objects layered on top of `analysis_melting.run`.

Each *Result class is a thin parsed view over the dict produced by the
orchestrator. The original dict is kept on `._raw` so `to_dict()` returns
byte-identical output — this is the contract with the Pyodide bridge,
which keeps consuming `analysis_melting.run` directly and is unaffected
by this module.

Failed sub-fits keep `ok=False` and `error` set; call `.require()` to
turn a failed sub-fit into a `FitFailed` exception.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd


class RawTm(NamedTuple):
    ok:         bool                 = False
    error:      Optional[str]        = None
    Tm:         Optional[float]      = np.nan
    signal:     Optional[np.ndarray] = None
    base_b_r:   Optional[tuple]      = None
    base_ub_r:  Optional[tuple]      = None
    base_med_r: Optional[tuple]      = None

class VanHoffFit(NamedTuple):
    """Linearised van't Hoff regression (ln K vs 1/T).

    `fit_params` is the `(slope, intercept)` tuple of the linear fit in
    the linearised x-axis (1000/(T-T0)) — multiply by `slope`·x + intercept
    to recover the fitted ln(K) values.
    """
    ok:         bool                 = False
    error:      Optional[str]        = None
    dH:         Optional[float]      = None
    dS:         Optional[float]      = None
    dG:         Optional[float]      = None
    Tm:         Optional[float]      = None
    t1:         Optional[np.ndarray] = None
    K:          Optional[np.ndarray] = None
    xdata:      Optional[np.ndarray] = None
    ydata:      Optional[np.ndarray] = None
    fit_params: Optional[tuple]      = None

    def __repr__(self) -> str:
        if self.ok:
            return (f"VanHoffFit(dH={self.dH}, dS={self.dS}, "
                    f"dG={self.dG}, Tm={self.Tm})")
        return f"VanHoffFit(ok=False, error={self.error})"

class SingleFit(NamedTuple):
    ok:         bool                 = False
    error:      Optional[str]        = None
    dH:         Optional[float]      = None
    dS:         Optional[float]      = None
    dG:         Optional[float]      = None
    Tm:         Optional[float]      = None
    base_b:     Optional[tuple]      = None
    base_ub:    Optional[tuple]      = None
    base_med:   Optional[tuple]      = None
    curve_fit:      Optional[np.ndarray] = None
    derivative: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        if self.ok:
            return (f"SingleFit(dH={self.dH}, dS={self.dS}, "
                f"dG={self.dG}, Tm={self.Tm})")
        return f"SingleFit(ok=False, error={self.error})"

class GlobalFitCurve(NamedTuple):
    """One curve inside a shared-ΔH/ΔS joint fit — baselines stay per-curve."""
    name:       str
    oligoC:     Optional[float]      = None
    c0:         Optional[float]      = None
    Tm_fit:     Optional[float]      = None
    base_b:     Optional[tuple]      = None
    base_ub:    Optional[tuple]      = None
    base_med:   Optional[tuple]      = None
    curve:      Optional[np.ndarray] = None
    T_used:     Optional[np.ndarray] = None
    signal:     Optional[np.ndarray] = None
    signal_all: Optional[np.ndarray] = None

class GlobalFit(NamedTuple):
    """Return value of `MeltSet.global_fit()` — one ΔH/ΔS for all curves."""
    dH:      Optional[float]      = None      # kcal/mol
    dS:      Optional[float]      = None      # kcal/(mol·K)
    dG:      Optional[float]      = None      # kcal/mol at 37 °C
    T_used:  Optional[list]       = None      # one window per curve
    curves:  list[GlobalFitCurve] = field(default_factory=list)
    error:   Optional[str]        = None

    def __repr__(self) -> str:
        if self.error:
            return f"GlobalFit(error={self.error})"
        return (f"GlobalFit(dH={self.dH}, dS={self.dS}, "
            f"dG={self.dG}, curves=GlobalFitCurve({len(self.curves)}, "
            f"oligoC={[c.oligoC for c in self.curves]}, "
            f"Tm_fit={[c.Tm_fit for c in self.curves]}))")

class ConcentrationCurve(NamedTuple):
    """One curve in a concentration series — Tm by all three extractions.

    `*_K` fields are the same Tm in Kelvin, which is what the 1/Tm
    regression consumes.
    """
    name:       str
    oligoC:     Optional[float]      = None
    c0:         Optional[float]      = None
    Tm_raw:     Optional[float]      = None
    Tm_vH:      Optional[float]      = None
    Tm_fit:     Optional[float]      = None
    lnCT:       Optional[float]      = None
    T_used:     Optional[np.ndarray] = None
    signal:     Optional[np.ndarray] = None
    signal_all: Optional[np.ndarray] = None
    base_b:     Optional[tuple]      = None
    base_ub:    Optional[tuple]      = None

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
    dG_37:       Optional[float]      = None
    r_squared:   Optional[float]      = None
    slope:       Optional[float]      = None
    intercept:   Optional[float]      = None
    ln_ct:       Optional[np.ndarray] = None
    inv_tm:      Optional[np.ndarray] = None
    ln_ct_line:  Optional[np.ndarray] = None
    inv_tm_line: Optional[np.ndarray] = None

class ConcDepResult(NamedTuple):
    struct_type:        str                       = "heterodimer"
    self_complementary: bool                      = False
    per_curve:          list[ConcentrationCurve]  = field(default_factory=list)
    skipped:            list[dict]                = field(default_factory=list)
    series_raw:         ConcentrationSeries       = field(
        default_factory=lambda: ConcentrationSeries(method="raw"))
    series_vh:          ConcentrationSeries       = field(
        default_factory=lambda: ConcentrationSeries(method="vh"))
    series_fit:         ConcentrationSeries       = field(
        default_factory=lambda: ConcentrationSeries(method="fit"))
    error:              Optional[str]             = None

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


