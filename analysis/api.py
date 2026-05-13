"""Pythonic façade over `analysis_melting.run`.

Three mode-specific entry points (single column, shared-ΔH multi fit,
concentration-series van't Hoff) plus a CSV one-liner. Each builds the
same params dict the browser bridge passes and forwards to
`analysis_melting.run`. Results are plain dicts — same shape returned to
the browser and printed by the CLI.
"""

from pathlib import Path
from typing import Mapping, Optional, Union

import pandas as pd

from analysis import analysis_melting
from analysis.cleaning import clean, get_signal_columns


def _common_params(
    *,
    signal_type: str,
    struct_type: str,
    T_low: Optional[float],
    T_high: Optional[float],
    bl_lower_offset: float,
    bl_upper_offset: float,
    salt: float,
) -> dict:
    p = {
        "signal_type":     signal_type,
        "struct_type":     struct_type,
        "bl_lower_offset": bl_lower_offset,
        "bl_upper_offset": bl_upper_offset,
        "salt":            salt,
    }
    if T_low  is not None: p["T_low"]  = T_low
    if T_high is not None: p["T_high"] = T_high
    return p


def analyze_single(
    df: pd.DataFrame,
    column: str,
    *,
    struct_type: str = "heterodimer",
    signal_type: str = "absorbance",
    oligo: float = 0.5,
    salt: float = 150.0,
    T_low: Optional[float] = None,
    T_high: Optional[float] = None,
    bl_lower_offset: float = 10.0,
    bl_upper_offset: float = 10.0,
) -> dict:
    """Single-column two-state fit.

    Returns a dict with `TmRaw`, `vantHoff` (linearised), `fit_result`
    (full nonlinear fit), plus the temperature/signal arrays used.
    """
    params = _common_params(
        signal_type=signal_type, struct_type=struct_type,
        T_low=T_low, T_high=T_high,
        bl_lower_offset=bl_lower_offset, bl_upper_offset=bl_upper_offset,
        salt=salt,
    )
    params["column"] = column
    params["oligo"]  = oligo
    return analysis_melting.run(df, params)


def analyze_multi(
    df: pd.DataFrame,
    oligo_multi: Mapping[str, float],
    *,
    struct_type: str = "heterodimer",
    signal_type: str = "absorbance",
    salt: float = 150.0,
    T_low: Optional[float] = None,
    T_high: Optional[float] = None,
    bl_lower_offset: float = 10.0,
    bl_upper_offset: float = 10.0,
) -> dict:
    """Joint shared-ΔH/ΔS fit across the columns named in `oligo_multi`.

    Each column carries its own concentration (µM) and independent
    baselines, but ΔH and ΔS are shared.
    """
    params = _common_params(
        signal_type=signal_type, struct_type=struct_type,
        T_low=T_low, T_high=T_high,
        bl_lower_offset=bl_lower_offset, bl_upper_offset=bl_upper_offset,
        salt=salt,
    )
    params["column"]      = "__multi__"
    params["oligo_multi"] = dict(oligo_multi)
    return analysis_melting.run(df, params)


def analyze_concentration(
    df: pd.DataFrame,
    oligo_multi: Mapping[str, float],
    *,
    struct_type: str = "heterodimer",
    signal_type: str = "absorbance",
    salt: float = 150.0,
    T_low: Optional[float] = None,
    T_high: Optional[float] = None,
    bl_lower_offset: float = 10.0,
    bl_upper_offset: float = 10.0,
) -> dict:
    """Concentration-series van't Hoff: 1/Tm vs ln(C_T/f) regression.

    Extracts three Tm values per column (raw / van't Hoff / full fit) and
    returns a separate regression for each under `series.{raw,vh,fit}`.
    """
    params = _common_params(
        signal_type=signal_type, struct_type=struct_type,
        T_low=T_low, T_high=T_high,
        bl_lower_offset=bl_lower_offset, bl_upper_offset=bl_upper_offset,
        salt=salt,
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
