"""
Simplified version of the RNAmelt api. Keep only necessay stuff.

At first construction is only directly in python
The MeltAnalysis take only contain only one curve.

"""

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from rnamelt import analysis_melting
from rnamelt.cleaning import clean, get_signal_columns

# Re-export so callers can introspect / extend defaults:
#   from rnamelt import SOLVER_DEFAULTS, VH_DEFAULTS, FIT_INIT_DEFAULTS
from rnamelt.methods import SOLVER_DEFAULTS, VH_DEFAULTS, FIT_INIT_DEFAULTS  # noqa: F401
from rnamelt.results import (
    ConcentrationResult,
    MultiResult,
    SingleResult,
)


_CONFIG_KEYS = {
    "struct_type",
    "salt",
    "T_low",
    "T_high",
    "bl_lower_offset",
    "bl_upper_offset",
    "solver",
    "vh",
    "fit_init",
}


class MeltAnalysis:
    """
    Knobs:
        struct_type      "heterodimer" / "homodimer" / "monomer"
        T_low, T_high    transition window (°C); None → use full range.
        bl_lower_offset  folded-baseline offset above T_low (°C).
        bl_upper_offset  unfolded-baseline offset below T_high (°C).
        solver           dict of `scipy.optimize.least_squares` overrides
                         (see `rnamelt.SOLVER_DEFAULTS`).
        vh               dict of van't Hoff linearisation overrides
                         (see `rnamelt.VH_DEFAULTS`).
        fit_init         dict of full-fit initial guesses
                         (see `rnamelt.FIT_INIT_DEFAULTS`).

    Mode methods return objects from `rnamelt.results` and also cache
    them on `self.last_result`. Use `result.to_dict()` to get the legacy
    orchestrator dict (same shape `analysis_melting.run` produces).
    """

    def __init__(
        self,
        signal,
        temps,
        *,
        struct_type: str = "heterodimer",
        T_low: Optional[float] = None,
        T_high: Optional[float] = None,
        bl_lower_offset: float = 10.0,
        bl_upper_offset: float = 10.0,
        solver: Optional[Mapping[str, Any]] = None,
        vh: Optional[Mapping[str, Any]] = None,
        fit_init: Optional[Mapping[str, Any]] = None,
    ):
        self.signal          = signal
        self.temps           = temps
        self.struct_type     = struct_type
        self.T_low           = T_low
        self.T_high          = T_high
        self.bl_lower_offset = bl_lower_offset
        self.bl_upper_offset = bl_upper_offset
        self.solver          = dict(solver)   if solver   else None
        self.vh              = dict(vh)       if vh       else None
        self.fit_init        = dict(fit_init) if fit_init else None
        self.last_result: Union[SingleResult, MultiResult, ConcentrationResult, None] = None

    # ── construction helpers ───────────────────────────────────────────
    # NOTE: not implemented yet, one should construct the class directly in
    # python accordingly.

    # ── mode methods ───────────────────────────────────────────────────

    def single(self, column: str, *, oligo: float = 0.5) -> SingleResult:
        """Single-column two-state fit. Returns a `SingleResult`."""
        params = self._params()
        params["column"] = column
        params["oligo"]  = oligo
        result = SingleResult.from_dict(analysis_melting.run(self.signal, params))
        self.last_result = result
        return result

    def multi(self, oligo_multi) -> MultiResult:
        """Shared-ΔH/ΔS joint fit across the columns in `oligo_multi`."""
        params = self._params()
        params["column"]      = "__multi__"
        params["oligo_multi"] = oligo_multi
        result = MultiResult.from_dict(analysis_melting.run(self.signal, params))
        self.last_result = result
        return result

    def concentration(self, oligo_multi: Mapping[str, float]) -> ConcentrationResult:
        """1/Tm vs ln(C_T/f) van't Hoff regression across concentrations."""
        params = self._params()
        params["column"]      = "__concentration__"
        params["oligo_multi"] = dict(oligo_multi)
        result = ConcentrationResult.from_dict(analysis_melting.run(self.signal, params))
        self.last_result = result
        return result

    # ── plotting (lazy matplotlib import via rnamelt.plots) ───────────────

    def plot(
        self,
        column: Optional[str] = None,
        *,
        oligo: float = 0.5,
        axes=None,
        figsize=(15, 5),
    ):
        """Run `single(column)` and plot its 3-panel figure.

        When `column` is None, the first detected signal column is used.
        Returns `(fig, axes, SingleResult)`.
        """
        if column is None:
            sigs = self.signal
            if not sigs:
                raise ValueError("no signal columns")
            column = sigs[0]
        result = self.single(column, oligo=oligo)
        fig, axes = result.plot(axes=axes, figsize=figsize)
        return fig, axes, result

    # ── internals ──────────────────────────────────────────────────────

    def _params(self) -> dict:
        """Assemble the orchestrator params dict from the current config."""
        p: dict = {
            "struct_type":     self.struct_type,
            "bl_lower_offset": self.bl_lower_offset,
            "bl_upper_offset": self.bl_upper_offset,
            "salt":            self.salt,
        }
        if self.T_low    is not None: p["T_low"]    = self.T_low
        if self.T_high   is not None: p["T_high"]   = self.T_high
        if self.solver:               p["solver"]   = dict(self.solver)
        if self.vh:                   p["vh"]       = dict(self.vh)
        if self.fit_init:             p["fit_init"] = dict(self.fit_init)
        return p


__all__ = [
    "MeltAnalysis",
    "SOLVER_DEFAULTS",
    "VH_DEFAULTS",
    "FIT_INIT_DEFAULTS",
]
