"""Pythonic façade over `analysis_melting.run`.

Three mode-specific entry points (single column, shared-ΔH multi fit,
concentration-series van't Hoff) plus a CSV one-liner. Each builds the
same params dict the browser bridge passes and forwards to
`analysis_melting.run`. Results are plain dicts — same shape returned to
the browser and printed by the CLI.
"""

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import pandas as pd

from rnamelt import analysis_melting
from rnamelt.cleaning import clean, get_signal_columns

# Re-export so callers can introspect / extend defaults:
#   from rnamelt import SOLVER_DEFAULTS, VH_DEFAULTS, FIT_INIT_DEFAULTS
from rnamelt.methods import SOLVER_DEFAULTS, VH_DEFAULTS, FIT_INIT_DEFAULTS  # noqa: F401


def _common_params(
    *,
    #signal_type: str,
    struct_type: str,
    T_low: Optional[float],
    T_high: Optional[float],
    bl_lower_offset: float,
    bl_upper_offset: float,
    salt: float,
    solver: Optional[Mapping[str, Any]],
    vh: Optional[Mapping[str, Any]],
    fit_init: Optional[Mapping[str, Any]],
) -> dict:
    p = {
        #"signal_type":     signal_type,
        "struct_type":     struct_type,
        "bl_lower_offset": bl_lower_offset,
        "bl_upper_offset": bl_upper_offset,
        "salt":            salt,
    }
    if T_low  is not None: p["T_low"]    = T_low
    if T_high is not None: p["T_high"]   = T_high
    if solver:             p["solver"]   = dict(solver)
    if vh:                 p["vh"]       = dict(vh)
    if fit_init:           p["fit_init"] = dict(fit_init)
    return p


def analyze_single(
    df: pd.DataFrame,
    column: str,
    *,
    struct_type: str = "heterodimer",
    #signal_type: str = "absorbance",
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
    """Single-column two-state fit.

    Returns a dict with `TmRaw`, `vantHoff` (linearised), `fit_result`
    (full nonlinear fit), plus the temperature/signal arrays used.

    `solver` is an optional dict of overrides forwarded to
    `scipy.optimize.least_squares` for the full-function fit. Valid keys
    are those in `rnamelt.methods.SOLVER_DEFAULTS`:
    `max_nfev`, `ftol`, `gtol`, `xtol`, `method`, `loss`, `f_scale`,
    `jac`, `verbose`, `residuals_method` (multi-fit only). Unspecified
    keys fall back to defaults tuned for in-browser use.

    `vh` is an optional dict of van't Hoff linearisation overrides — keys
    in `rnamelt.methods.VH_DEFAULTS`: `border` (fraction-folded cutoff,
    default 0.15), `t1_min` / `t1_max` (raw T_scale/(T-T0) clips,
    default -1 = no clip), `T_scale` (numerical conditioning factor,
    default 1000).

    `fit_init` is an optional dict of full-fit initial guesses — keys in
    `rnamelt.methods.FIT_INIT_DEFAULTS`: `dH_init`, `dS_init` (None →
    auto-seed from van't Hoff or fall back), `lin_init` (number of
    leading/trailing points averaged for the baseline-intercept seed;
    default 10), `b1_init`, `b2_init` (single-mode only — explicit
    (slope, intercept) tuples for the folded / unfolded baselines).
    """
    params = _common_params(
        #signal_type=signal_type,
        struct_type=struct_type,
        T_low=T_low, T_high=T_high,
        bl_lower_offset=bl_lower_offset, bl_upper_offset=bl_upper_offset,
        salt=salt, solver=solver, vh=vh, fit_init=fit_init,
    )
    params["column"] = column
    params["oligo"]  = oligo
    return analysis_melting.run(df, params)


def analyze_multi(
    df: pd.DataFrame,
    oligo_multi: Mapping[str, float],
    *,
    struct_type: str = "heterodimer",
    #signal_type: str = "absorbance",
    salt: float = 150.0,
    T_low: Optional[float] = None,
    T_high: Optional[float] = None,
    bl_lower_offset: float = 10.0,
    bl_upper_offset: float = 10.0,
    solver: Optional[Mapping[str, Any]] = None,
    vh: Optional[Mapping[str, Any]] = None,
    fit_init: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Joint shared-ΔH/ΔS fit across the columns named in `oligo_multi`.

    Each column carries its own concentration (µM) and independent
    baselines, but ΔH and ΔS are shared. See `analyze_single` for the
    shape of the `solver` argument.
    """
    params = _common_params(
        #signal_type=signal_type,
        struct_type=struct_type,
        T_low=T_low, T_high=T_high,
        bl_lower_offset=bl_lower_offset, bl_upper_offset=bl_upper_offset,
        salt=salt, solver=solver, vh=vh, fit_init=fit_init,
    )
    params["column"]      = "__multi__"
    params["oligo_multi"] = dict(oligo_multi)
    return analysis_melting.run(df, params)


def analyze_concentration(
    df: pd.DataFrame,
    oligo_multi: Mapping[str, float],
    *,
    struct_type: str = "heterodimer",
    #signal_type: str = "absorbance",
    salt: float = 150.0,
    T_low: Optional[float] = None,
    T_high: Optional[float] = None,
    bl_lower_offset: float = 10.0,
    bl_upper_offset: float = 10.0,
    solver: Optional[Mapping[str, Any]] = None,
    vh: Optional[Mapping[str, Any]] = None,
    fit_init: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Concentration-series van't Hoff: 1/Tm vs ln(C_T/f) regression.

    Extracts three Tm values per column (raw / van't Hoff / full fit) and
    returns a separate regression for each under `series.{raw,vh,fit}`.
    The `solver` dict is forwarded to the per-curve full-function fit;
    see `analyze_single`.
    """
    params = _common_params(
        #signal_type=signal_type,
        struct_type=struct_type,
        T_low=T_low, T_high=T_high,
        bl_lower_offset=bl_lower_offset, bl_upper_offset=bl_upper_offset,
        salt=salt, solver=solver, vh=vh, fit_init=fit_init,
    )
    params["column"]      = "__concentration__"
    params["oligo_multi"] = dict(oligo_multi)
    return analysis_melting.run(df, params)


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
    are forwarded to the mode function.
    """
    df = clean(pd.read_csv(path))

    if mode == "single":
        if column is None:
            sig = get_signal_columns(df)
            if not sig:
                raise ValueError("CSV has no signal columns")
            column = sig[0]
        return analyze_single(df, column, oligo=oligo, **kwargs)

    if mode == "multi":
        if not oligo_multi:
            raise ValueError("multi mode requires oligo_multi")
        return analyze_multi(df, oligo_multi, **kwargs)

    if mode == "concentration":
        if not oligo_multi:
            raise ValueError("concentration mode requires oligo_multi")
        return analyze_concentration(df, oligo_multi, **kwargs)

    raise ValueError(
        f"unknown mode {mode!r}; expected 'single', 'multi', or 'concentration'"
    )
