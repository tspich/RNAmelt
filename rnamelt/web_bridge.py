"""JS ↔ Python bridge for the browser UI.

Only responsibility: translate the params dict the JS side sends into
``SingleCurve`` / ``MultiCurves`` calls, then translate the resulting
NamedTuples into the flat dict shape ``renderResults`` (and its
multi/concentration cousins) already reads. No analysis lives here.

Entry points:

    from rnamelt.web_bridge import run, auto_baseline_windows
    payload  = run(df, params)                    # JSON-safe dict
    detected = auto_baseline_windows(df, params)  # {ok, fold_low_T, fold_high_T,
                                                  #  unf_low_T, unf_high_T, ...}

Params dict (JS → Python):
    struct_type      : "heterodimer" | "homodimer" | "monomer"
    column           : signal column name, or "__multi__" / "__concentration__"
    oligo            : per-strand µM  (single-column mode)
    oligo_multi      : {column: µM, ...}  (multi / concentration modes)
    fold_low_T       : °C, outer edge of the folded baseline window
    fold_high_T      : °C, inner edge of the folded baseline window
    unf_low_T        : °C, inner edge of the unfolded baseline window
    unf_high_T       : °C, outer edge of the unfolded baseline window
    signal_type      : accepted, currently unused (the API is direction-agnostic)

Omit all four window temperatures to let SingleCurve detect them.

Analysis settings (the sidebar's Advanced panel; one set for every curve).
Missing or null values keep the API defaults:
    vh          : {border}                      → SingleCurve(vh=...)
    fit_init    : {dH_init, dS_init, lin_init}  → SingleCurve(fit_init=...),
                  dH in kcal/mol, dS in kcal/(mol·K)
    bounds_mode : "physical" | "loose" | "positive" | "unbounded"
    strategy    : "single" | "multistart"  (single-curve and concentration
                  fits; the global fit is always one fit)

Per-column overrides (multi / concentration modes):
    column_windows : {col: {fold_low_T, fold_high_T, unf_low_T, unf_high_T}}
        Full 4-window override for the given column. Every column in
        ``oligo_multi`` may be listed or omitted independently; columns
        without an entry use the top-level shared windows, or are
        auto-detected when those are absent too.

The single-column response always includes
``baseline_windows = [fold_low_T, fold_high_T, unf_low_T, unf_high_T]``
so the chart can shade the folded / unfolded fit windows.

``auto_baseline_windows`` detects the windows (as SingleCurve does when
built without them) on the selected column (or the first ``oligo_multi``
column in multi / concentration mode) and returns the four temperatures
ready to drop into the UI, with ``source`` "auto". When detection fails it
returns the edge-window fallback the analysis would use, with ``source``
"fallback" and the reason in ``warning``. Concentration is not needed.

Errors return ``{"error": <str>, "is_concentration": <bool>}`` from
``run`` and ``{"ok": False, "error": <str>}`` from
``auto_baseline_windows`` — never raise.
"""
from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd

# Import directly from the api submodule, not through the package façade.
# Reason: index.html reloads rnamelt.api on every run (dev hot-reload), which
# rebinds api.SingleCurve/api.MultiCurves to fresh class objects. The package
# namespace (rnamelt.SingleCurve) is set once by __init__.py and never
# re-bound, so `from rnamelt import ...` here would capture stale v1 classes
# while MultiCurves.__init__ (executing from the reloaded api) would resolve
# SingleCurve to the v2 class in api.__dict__. isinstance(v1, v2) is False,
# and the error surfaces as the surreal "expected SingleCurve, got SingleCurve".
from rnamelt.api import SingleCurve, MultiCurves
from rnamelt.methods import BaselineWarning
from rnamelt.utils import safe_json


_MULTI_SENTINEL = "__multi__"
_CONC_SENTINEL  = "__concentration__"

# Cap the nonlinear-fit iteration count for browser workloads. The API's
# SOLVER_DEFAULTS uses max_nfev=1e12 (effectively unlimited) which is fine
# on a workstation but locks Pyodide up on real UV curves — especially in
# concentration mode where fit_full_function runs once per curve. 500
# iterations is enough for a well-behaved two-state fit; a fit that needs
# more is diverging and the extra iterations don't help. Merged into any
# solver overrides the caller passes so we don't clobber tolerance settings.
_WEB_SOLVER_CAPS = {"max_nfev": 500}


def run(df: pd.DataFrame, params: Mapping[str, Any]) -> dict:
    """JS-facing entry point. Dispatches on ``params['column']``."""
    try:
        column = params.get("column")
        if column == _MULTI_SENTINEL:
            return _run_multi(df, params)
        if column == _CONC_SENTINEL:
            return _run_concentration(df, params)
        return _run_single(df, params, column)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "is_concentration": params.get("column") == _CONC_SENTINEL}


# ── single-column ────────────────────────────────────────────────────────────

def _run_single(df: pd.DataFrame, params: Mapping[str, Any],
                column: str) -> dict:
    T_all, signal_all = _extract_column(df, column)
    curve = _build_curve(T_all, signal_all, params, column,
                         oligoC=_get_oligo(params))

    raw = curve.raw_tm()
    vh  = curve.vant_hoff() if raw.ok else None
    fit = curve.fit(vh_fit=vh, strategy=_strategy(params)) if raw.ok else None

    T_used, signal_used = curve.windowed_data()

    # The four baseline boundaries — always emitted so the melt chart can
    # shade [fold_low_T, fold_high_T] (folded) and [unf_low_T, unf_high_T]
    # (unfolded), whether the values came from the sidebar handles or from
    # the Auto detect button.
    baseline_windows = [
        float(curve.fold_low_T),
        float(curve.fold_high_T),
        float(curve.unf_low_T),
        float(curve.unf_high_T),
    ]

    payload = {
        "is_multi":         False,
        "is_concentration": False,
        "column":           column,
        "c0":               curve.c0,
        "T_all":            T_all,
        "signal_all":       signal_all,
        "T_used":           T_used,
        "signal_used":      signal_used,
        "TmRaw":            raw.Tm if raw.ok else None,
        "base_b_r":         raw.base_b  if raw.ok else None,
        "base_ub_r":        raw.base_ub if raw.ok else None,
        "vantHoff":         _vh_payload(vh),
        "fit_result":       _fit_payload(fit, T_used, signal_used),
        "baseline_windows": baseline_windows,
    }
    return safe_json(payload)


def auto_baseline_windows(df: pd.DataFrame,
                          params: Mapping[str, Any]) -> dict:
    """Detect the baseline windows of the target column the way SingleCurve
    does when built without windows, and return the four temperatures for
    the UI's four handles.

    In single-column mode the target is ``params['column']``. In
    ``__multi__`` / ``__concentration__`` mode it is the first entry of
    ``params['oligo_multi']``.
    """
    try:
        column = params.get("column")
        if column in (_MULTI_SENTINEL, _CONC_SENTINEL):
            column = next(iter(_get_oligo_multi(params)))

        T_all, S = _extract_column(df, column)
        no_windows = {k: v for k, v in params.items()
                      if k not in _WINDOW_KEYS and k != "column_windows"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", BaselineWarning)
            # Detection does not depend on concentration: any valid oligoC
            # lets the curve be built before the user has entered C_T.
            curve = _build_curve(T_all, S, no_windows, name=column,
                                 oligoC=_DETECT_OLIGO)
        fl, fh, ul, uh = curve.window_temps
        out = {
            "ok":          True,
            "column":      column,
            "source":      curve.window_source,
            "fold_low_T":  float(fl),
            "fold_high_T": float(fh),
            "unf_low_T":   float(ul),
            "unf_high_T":  float(uh),
        }
        if curve.window_source != "auto":
            out["warning"] = next((str(w.message) for w in caught
                                   if issubclass(w.category, BaselineWarning)),
                                  "automatic baseline detection failed")
        return out
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _vh_payload(vh) -> dict | None:
    if vh is None:
        return None
    if not vh.ok:
        return {"success": False, "error": vh.error}
    return {
        "success":  True,
        "dH":       vh.dH,
        "dS":       vh.dS,
        "dG":       vh.dG,
        "T_m_vH":   vh.Tm,
        "t1":       vh.t1,
        "K":        vh.K,
        "xdata":    vh.xdata,
        "ydata":    vh.ydata,
        "fit_vh":   list(vh.fit_params) if vh.fit_params is not None else None,
    }


def _fit_payload(fit, T_used, signal_used) -> dict | None:
    if fit is None:
        return None
    if not fit.ok:
        return {"success": False, "error": fit.error}
    return {
        "success":    True,
        "dH":         fit.dH,
        "dS":         fit.dS,
        "dG":         fit.dG,
        "T_m_fit":    fit.Tm,
        "ym":         fit.ym,
        "base_b_f":   list(fit.base_b)   if fit.base_b   is not None else None,
        "base_ub_f":  list(fit.base_ub)  if fit.base_ub  is not None else None,
        "base_med_f": list(fit.base_med) if fit.base_med is not None else None,
        "fit":        fit.curve_fit,
        # derivative of the measured signal in the fit window (plotted with the fit)
        "derivative": -np.gradient(signal_used, T_used),
        "converged":  fit.converged,
        **_errors_payload(fit),
        "SE_Tm":      fit.SE_Tm,
    }


# ── multi (shared ΔH joint fit) ──────────────────────────────────────────────

def _run_multi(df: pd.DataFrame, params: Mapping[str, Any]) -> dict:
    curves, T_all = _build_curves(df, params)
    ms = MultiCurves(curves)
    gf = ms.global_fit()

    if not gf.ok:
        return {"is_multi": True, "error": gf.error}

    # Every curve has its own windowed T; the melt chart uses one shared
    # x-axis. Emit the first curve's T_used as the reference grid and
    # ship each curve's own T_used alongside its fit so the JS can still
    # draw the fit against the right x-values.
    T_used_ref = gf.curves[0].T_used

    columns = []
    for c in gf.curves:
        columns.append({
            "name":       c.name,
            "oligoC":     c.oligoC,
            "c0":         c.c0,
            "T_m_fit":    c.Tm,
            "base_b_f":   list(c.base_b)  if c.base_b  is not None else None,
            "base_ub_f":  list(c.base_ub) if c.base_ub is not None else None,
            "base_med_f": list(c.base_med) if c.base_med is not None else None,
            "fit":        c.fit,
            "T_used":     c.T_used,
            "signal":     c.signal,
            "signal_all": c.signal_all,
        })

    payload = {
        "is_multi":         True,
        "is_concentration": False,
        "struct_type":      ms.struct_type,
        "T_all":            T_all,
        "T_used":           T_used_ref,
        "dH":               gf.dH,
        "dS":               gf.dS,
        "dG":               gf.dG,
        "converged":        gf.converged,
        **_errors_payload(gf),
        "columns":          columns,
    }
    return safe_json(payload)


# ── concentration series ─────────────────────────────────────────────────────

def _run_concentration(df: pd.DataFrame, params: Mapping[str, Any]) -> dict:
    curves, T_all = _build_curves(df, params)
    ms = MultiCurves(curves)
    # If the caller (typically the browser) has cached per-curve Tms from
    # single-mode runs, pass them through so conc_dependency doesn't redo
    # the fit for those curves. Caller vouches that the cache matches the
    # current baseline windows.
    precomputed = params.get("precomputed_tms")
    cdr = ms.conc_dependency(precomputed_tms=precomputed,
                             strategy=_strategy(params))

    if not cdr.ok and not cdr.per_curve:
        return {"is_concentration": True, "error": cdr.error}

    # Reference T grid for the melt chart. per_curve may be a subset of
    # the constructed curves if some got skipped — use the first curve
    # that survived, falling back to the first input curve.
    T_used_ref = (cdr.per_curve[0].T_used if cdr.per_curve
                  else curves[0].windowed_data()[0])

    per_curve = []
    for c in cdr.per_curve:
        per_curve.append({
            "name":       c.name,
            "oligoC":     c.oligoC,
            "c0":         c.c0,
            "lnCT":       c.lnCT,
            "TmRaw":      c.Tm_raw,
            "TmvH":       c.Tm_vH,
            "Tmfit":      c.Tm_fit,
            "TmRaw_K":    c.Tm_raw_K,
            "TmvH_K":     c.Tm_vH_K,
            "Tmfit_K":    c.Tm_fit_K,
            "T_used":     c.T_used,
            "signal":     c.signal,
            "signal_all": c.signal_all,
            "base_b":     list(c.base_b)  if c.base_b  is not None else None,
            "base_ub":    list(c.base_ub) if c.base_ub is not None else None,
        })

    payload = {
        "is_multi":           False,
        "is_concentration":   True,
        "struct_type":        cdr.struct_type,
        "self_complementary": cdr.self_complementary,
        "T_all":              T_all,
        "T_used":             T_used_ref,
        "per_curve":          per_curve,
        "skipped":            list(cdr.skipped),
        "series":             {
            "raw": _series_payload(cdr.series_raw),
            "vh":  _series_payload(cdr.series_vh),
            "fit": _series_payload(cdr.series_fit),
        },
        "error":              cdr.error,
    }
    return safe_json(payload)


def _series_payload(s) -> dict | None:
    if s is None:
        return None
    if not s.ok:
        return {"success": False, "method": s.method, "error": s.error}
    return {
        "success":     True,
        "method":      s.method,
        "n":           s.n,
        "dH":          s.dH,
        "dS":          s.dS,
        "dG_37":       s.dG,       # JS reads dG_37 in concentration mode
        "r_squared":   s.r_squared,
        "slope":       s.slope,
        "intercept":   s.intercept,
        "ln_ct":       s.ln_ct,
        "inv_tm":      s.inv_tm,
        "ln_ct_line":  s.ln_ct_line,
        "inv_tm_line": s.inv_tm_line,
        **_errors_payload(s),
    }


def _errors_payload(res) -> dict:
    """Standard errors of ΔH / ΔS / ΔG (37 °C); None when undefined."""
    return {k: getattr(res, k) for k in
            ("SE_dH", "SE_dS", "SE_dG", "cov_dH_dS", "SE_note")}


# ── shared helpers ───────────────────────────────────────────────────────────

def _extract_column(df: pd.DataFrame, column: str) -> tuple[np.ndarray,
                                                             np.ndarray]:
    if column not in df.columns:
        raise ValueError(
            f"column {column!r} not in DataFrame "
            f"(have: {list(df.columns)})"
        )
    T_col = df.columns[0]
    T = np.asarray(df[T_col], dtype=float)
    S = np.asarray(df[column], dtype=float)
    return T, S


def _validate_oligo(name: str, v) -> float:
    """Coerce ``v`` to a float and require it to be finite and > 0.
    The UI removed the default C_T value on purpose (users must supply
    it), so a missing / non-positive / NaN value here is a real error
    that should be surfaced with a helpful message instead of silently
    propagating NaN through the fit."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive number; got {v!r}")
    if not np.isfinite(f) or f <= 0:
        raise ValueError(f"{name} must be a positive finite number; got {f}")
    return f


def _get_oligo(params: Mapping[str, Any]) -> float:
    v = params.get("oligo")
    if v is None:
        raise ValueError(
            "params['oligo'] is required in single-column mode "
            "(enter C_T µM in the sidebar)"
        )
    return _validate_oligo("oligo", v)


def _get_oligo_multi(params: Mapping[str, Any]) -> dict[str, float]:
    v = params.get("oligo_multi")
    if not isinstance(v, dict) or not v:
        raise ValueError(
            "params['oligo_multi'] is required and must be a "
            "{column: µM, ...} dict in multi / concentration mode"
        )
    out = {}
    for k, vv in v.items():
        out[str(k)] = _validate_oligo(f"oligo_multi[{k!r}]", vv)
    return out


_WINDOW_KEYS = ("fold_low_T", "fold_high_T", "unf_low_T", "unf_high_T")

# Placeholder concentration for baseline detection only (see
# auto_baseline_windows); never used for an analysis.
_DETECT_OLIGO = 1.0


def _set(d) -> dict:
    """A settings dict from the JS side with null / missing entries dropped,
    so they keep the API defaults."""
    if d is None:
        return {}
    if not isinstance(d, Mapping):
        raise ValueError(f"expected a settings object, got {d!r}")
    return {k: v for k, v in d.items() if v is not None}


def _settings_kw(params: Mapping[str, Any]) -> dict:
    """SingleCurve settings from the Advanced panel: vh, fit_init, bounds."""
    kw = {}
    vh = _set(params.get("vh"))
    if vh:
        kw["vh"] = vh
    fit_init = _set(params.get("fit_init"))
    if fit_init:
        kw["fit_init"] = fit_init
    if params.get("bounds_mode"):
        kw["bounds"] = {"mode": params["bounds_mode"]}
    return kw


def _strategy(params: Mapping[str, Any]) -> str:
    return params.get("strategy") or "single"


def _build_curve(T: np.ndarray, S: np.ndarray,
                 params: Mapping[str, Any], name: str,
                 oligoC: float) -> SingleCurve:
    """Build one SingleCurve from the JS param bag. Window temperatures that
    are present are passed through; with none, SingleCurve auto-detects."""
    # Merge web-side solver caps with any user-supplied overrides. Caps win
    # for max_nfev so a rogue caller can't uncap it, but other keys (ftol,
    # xtol, gtol, ...) still flow through.
    solver_overrides = dict(params.get("solver") or {})
    solver_overrides.update(_WEB_SOLVER_CAPS)
    kw = dict(
        name=name,
        oligoC=oligoC,
        struct_type=params.get("struct_type", "heterodimer"),
        solver=solver_overrides,
        **_settings_kw(params),
    )
    for k in _WINDOW_KEYS:
        if params.get(k) is not None:
            kw[k] = float(params[k])
    return SingleCurve(T, S, **kw)


def _build_curves(df: pd.DataFrame, params: Mapping[str, Any]
                  ) -> tuple[list[SingleCurve], np.ndarray]:
    """Build one SingleCurve per column in oligo_multi. Returns
    ``(curves, T_all)`` where ``T_all`` is the shared temperature array
    from the DataFrame.

    Per-column windows: ``params['column_windows'][col]`` if it has all four
    temperatures, else the top-level shared windows, else auto-detection.
    """
    oligo_multi = _get_oligo_multi(params)
    column_windows = params.get("column_windows") or {}
    T_col = df.columns[0]
    T_all = np.asarray(df[T_col], dtype=float)
    curves: list[SingleCurve] = []
    for col, oligo in oligo_multi.items():
        if col not in df.columns:
            raise ValueError(
                f"oligo_multi references unknown column {col!r} "
                f"(have: {list(df.columns)})"
            )
        S = np.asarray(df[col], dtype=float)
        override = column_windows.get(col)
        if override and all(override.get(k) is not None for k in _WINDOW_KEYS):
            per_col_params = {**params, **{k: float(override[k]) for k in _WINDOW_KEYS}}
        else:
            per_col_params = params
        curves.append(_build_curve(T_all, S, per_col_params, col, oligoC=oligo))
    if len(curves) < 2:
        raise ValueError(
            f"multi / concentration mode needs at least 2 curves; "
            f"got {len(curves)}"
        )
    return curves, T_all
