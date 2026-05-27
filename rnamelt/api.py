"""Pythonic façade over `analysis_melting.run`.

Two surfaces, same machinery underneath:

* `MeltAnalysis` — a single class that holds the DataFrame and the
  configuration knobs (`struct_type`, `salt`, `T_low/T_high`, baseline
  offsets, `solver`, `vh`, `fit_init`) and exposes the three modes as
  methods returning typed result objects from `rnamelt.results`.

* `analyze_single` / `analyze_multi` / `analyze_concentration` /
  `analyze_csv` — thin functional wrappers that build a `MeltAnalysis`,
  call the appropriate mode, and return the legacy dict (`.to_dict()`).
  The Pyodide bridge does not go through this module — it calls
  `analysis_melting.run` directly — so the browser shape is unaffected.
"""

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import pandas as pd

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
    """Stateful façade over the orchestrator.

    The DataFrame and the configuration knobs live on the instance; the
    three mode methods (`single`, `multi`, `concentration`) reuse them.
    Configuration is mutable — set attributes directly, or call
    `configure(**overrides)` for a chainable form.

    Knobs:
        struct_type      "heterodimer" / "homodimer" / "monomer"
        salt             NaCl (mM) — emitted in the result; not in the model.
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
        df: pd.DataFrame,
        *,
        struct_type: str = "heterodimer",
        salt: float = 150.0,
        T_low: Optional[float] = None,
        T_high: Optional[float] = None,
        bl_lower_offset: float = 10.0,
        bl_upper_offset: float = 10.0,
        solver: Optional[Mapping[str, Any]] = None,
        vh: Optional[Mapping[str, Any]] = None,
        fit_init: Optional[Mapping[str, Any]] = None,
    ):
        self.df              = df
        self.struct_type     = struct_type
        self.salt            = salt
        self.T_low           = T_low
        self.T_high          = T_high
        self.bl_lower_offset = bl_lower_offset
        self.bl_upper_offset = bl_upper_offset
        self.solver          = dict(solver)   if solver   else None
        self.vh              = dict(vh)       if vh       else None
        self.fit_init        = dict(fit_init) if fit_init else None
        self.last_result: Union[SingleResult, MultiResult, ConcentrationResult, None] = None

    # ── construction helpers ───────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: Union[str, Path], **kwargs) -> "MeltAnalysis":
        """Read CSV, run `cleaning.clean`, return a fresh analyzer."""
        return cls(clean(pd.read_csv(path)), **kwargs)

    def configure(self, **overrides) -> "MeltAnalysis":
        """Set one or more configuration knobs in place; return self for chaining.

        Unknown keys raise `ValueError`. Dict-valued overrides (`solver`,
        `vh`, `fit_init`) replace the previous dict — they don't merge.
        """
        for k, v in overrides.items():
            if k not in _CONFIG_KEYS:
                raise ValueError(
                    f"unknown configuration key {k!r}; valid: {sorted(_CONFIG_KEYS)}"
                )
            setattr(self, k, v)
        return self

    def copy(self) -> "MeltAnalysis":
        """Return a new analyzer with the same DataFrame and config (config dicts copied)."""
        return MeltAnalysis(
            self.df,
            struct_type     = self.struct_type,
            salt            = self.salt,
            T_low           = self.T_low,
            T_high          = self.T_high,
            bl_lower_offset = self.bl_lower_offset,
            bl_upper_offset = self.bl_upper_offset,
            solver          = self.solver,
            vh              = self.vh,
            fit_init        = self.fit_init,
        )

    # ── introspection ──────────────────────────────────────────────────

    @property
    def signal_columns(self) -> list[str]:
        """Signal columns detected in the DataFrame (temperature excluded)."""
        return get_signal_columns(self.df)

    # ── mode methods ───────────────────────────────────────────────────

    def single(self, column: str, *, oligo: float = 0.5) -> SingleResult:
        """Single-column two-state fit. Returns a `SingleResult`."""
        params = self._params()
        params["column"] = column
        params["oligo"]  = oligo
        result = SingleResult.from_dict(analysis_melting.run(self.df, params))
        self.last_result = result
        return result

    def multi(self, oligo_multi: Mapping[str, float]) -> MultiResult:
        """Shared-ΔH/ΔS joint fit across the columns in `oligo_multi`."""
        params = self._params()
        params["column"]      = "__multi__"
        params["oligo_multi"] = dict(oligo_multi)
        result = MultiResult.from_dict(analysis_melting.run(self.df, params))
        self.last_result = result
        return result

    def concentration(self, oligo_multi: Mapping[str, float]) -> ConcentrationResult:
        """1/Tm vs ln(C_T/f) van't Hoff regression across concentrations."""
        params = self._params()
        params["column"]      = "__concentration__"
        params["oligo_multi"] = dict(oligo_multi)
        result = ConcentrationResult.from_dict(analysis_melting.run(self.df, params))
        self.last_result = result
        return result

    def single_all(self, *, oligo: float = 0.5) -> dict[str, SingleResult]:
        """Run `single(...)` on every detected signal column."""
        return {c: self.single(c, oligo=oligo) for c in self.signal_columns}

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


# ── functional façade — thin wrappers preserved for back-compat ─────────

def analyze_single(
    df: pd.DataFrame,
    column: str,
    *,
    struct_type: str = "heterodimer",
    oligo: float = 0.5,
    salt: float = 150.0,
    T_low: Optional[float] = None,
    T_high: Optional[float] = None,
    bl_lower_offset: float = 10.0,
    bl_upper_offset: float = 10.0,
    solver: Optional[Mapping[str, Any]] = None,
    vh: Optional[Mapping[str, Any]] = None,
    fit_init: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Single-column two-state fit. Returns the legacy dict.

    Equivalent to `MeltAnalysis(df, ...).single(column, oligo=...).to_dict()`.
    Prefer the class for new code — it exposes typed access to the same
    data via `SingleResult`. See `MeltAnalysis` for the meaning of
    `solver`, `vh`, `fit_init`.
    """
    return MeltAnalysis(
        df,
        struct_type=struct_type, salt=salt,
        T_low=T_low, T_high=T_high,
        bl_lower_offset=bl_lower_offset, bl_upper_offset=bl_upper_offset,
        solver=solver, vh=vh, fit_init=fit_init,
    ).single(column, oligo=oligo).to_dict()


def analyze_multi(
    df: pd.DataFrame,
    oligo_multi: Mapping[str, float],
    *,
    struct_type: str = "heterodimer",
    salt: float = 150.0,
    T_low: Optional[float] = None,
    T_high: Optional[float] = None,
    bl_lower_offset: float = 10.0,
    bl_upper_offset: float = 10.0,
    solver: Optional[Mapping[str, Any]] = None,
    vh: Optional[Mapping[str, Any]] = None,
    fit_init: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Shared-ΔH/ΔS joint fit. Returns the legacy dict.

    Equivalent to `MeltAnalysis(df, ...).multi(oligo_multi).to_dict()`.
    """
    return MeltAnalysis(
        df,
        struct_type=struct_type, salt=salt,
        T_low=T_low, T_high=T_high,
        bl_lower_offset=bl_lower_offset, bl_upper_offset=bl_upper_offset,
        solver=solver, vh=vh, fit_init=fit_init,
    ).multi(oligo_multi).to_dict()


def analyze_concentration(
    df: pd.DataFrame,
    oligo_multi: Mapping[str, float],
    *,
    struct_type: str = "heterodimer",
    salt: float = 150.0,
    T_low: Optional[float] = None,
    T_high: Optional[float] = None,
    bl_lower_offset: float = 10.0,
    bl_upper_offset: float = 10.0,
    solver: Optional[Mapping[str, Any]] = None,
    vh: Optional[Mapping[str, Any]] = None,
    fit_init: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Concentration-series van't Hoff. Returns the legacy dict.

    Equivalent to `MeltAnalysis(df, ...).concentration(oligo_multi).to_dict()`.
    """
    return MeltAnalysis(
        df,
        struct_type=struct_type, salt=salt,
        T_low=T_low, T_high=T_high,
        bl_lower_offset=bl_lower_offset, bl_upper_offset=bl_upper_offset,
        solver=solver, vh=vh, fit_init=fit_init,
    ).concentration(oligo_multi).to_dict()


def analyze_csv(
    path: Union[str, Path],
    *,
    mode: str = "single",
    column: Optional[str] = None,
    oligo: float = 0.5,
    oligo_multi: Optional[Mapping[str, float]] = None,
    **kwargs,
) -> dict:
    """Read CSV from `path`, clean, dispatch to the requested mode.

    `mode` is one of "single", "multi", "concentration". Single mode
    defaults to the first signal column when `column` is None. Multi and
    concentration modes require `oligo_multi`. Other keyword arguments
    are forwarded to `MeltAnalysis`.
    """
    m = MeltAnalysis.from_csv(path, **kwargs)

    if mode == "single":
        col = column
        if col is None:
            sig = m.signal_columns
            if not sig:
                raise ValueError("CSV has no signal columns")
            col = sig[0]
        return m.single(col, oligo=oligo).to_dict()

    if mode == "multi":
        if not oligo_multi:
            raise ValueError("multi mode requires oligo_multi")
        return m.multi(oligo_multi).to_dict()

    if mode == "concentration":
        if not oligo_multi:
            raise ValueError("concentration mode requires oligo_multi")
        return m.concentration(oligo_multi).to_dict()

    raise ValueError(
        f"unknown mode {mode!r}; expected 'single', 'multi', or 'concentration'"
    )


__all__ = [
    "MeltAnalysis",
    "analyze_single",
    "analyze_multi",
    "analyze_concentration",
    "analyze_csv",
    "SOLVER_DEFAULTS",
    "VH_DEFAULTS",
    "FIT_INIT_DEFAULTS",
]
