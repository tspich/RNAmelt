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
from typing import Any, Optional

import numpy as np
import pandas as pd


# ── CSV-writer helpers (browser-download format) ────────────────────────────

# Header shared by single / multi / batch per-column tables. Concentration
# mode writes its own multi-block layout.
_PER_COL_HEADER = [
    "column",
    "Tm_raw_C",
    "Tm_vH_C", "dH_vH_kcal_mol", "dS_vH_cal_molK", "dG37_vH_kcal_mol",
    "Tm_fit_C", "dH_fit_kcal_mol", "dS_fit_cal_molK", "dG37_fit_kcal_mol",
    "Tm_multi_C", "dH_multi_kcal_mol", "dS_multi_cal_molK", "dG37_multi_kcal_mol",
]


def _cell(v, scale=1.0):
    if v is None:
        return ""
    try:
        x = float(v) * scale
    except (TypeError, ValueError):
        return ""
    if x != x:  # NaN
        return ""
    return repr(x) if isinstance(x, float) else str(x)


def _inv_K(v):
    if v is None:
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return ""
    if x != x or x <= 0:
        return ""
    return repr(1.0 / x)


def _write_rows(path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)


class FitFailed(Exception):
    """Raised by SubFit.require() when the underlying fit did not succeed."""


# ── coercion helpers ────────────────────────────────────────────────────────

def _arr(x: Any) -> Optional[np.ndarray]:
    """Coerce list/tuple/ndarray/None to np.ndarray (None stays None).

    `safe_json` replaces NaN with None; convert those back so np.asarray
    doesn't choke on mixed dtypes.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray([np.nan if v is None else v for v in x], dtype=float)
    return np.asarray(x, dtype=float)


def _tup(x: Any) -> Optional[tuple]:
    if x is None:
        return None
    return tuple(float(v) for v in x)


# ── sub-fits ────────────────────────────────────────────────────────────────

@dataclass
class VanHoffFit:
    """Linearised van't Hoff regression (ln K vs 1/T)."""
    ok:       bool                  = False
    error:    Optional[str]         = None
    dH:       Optional[float]       = None
    dS:       Optional[float]       = None
    dG:       Optional[float]       = None
    Tm:       Optional[float]       = None
    t1:       Optional[np.ndarray]  = None
    K:        Optional[np.ndarray]  = None
    xdata:    Optional[np.ndarray]  = None
    ydata:    Optional[np.ndarray]  = None
    fit_line: Optional[np.ndarray]  = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "VanHoffFit":
        if not d:
            return cls(ok=False, error="missing")
        if not d.get("success", False):
            return cls(ok=False, error=d.get("error", "unknown"))
        return cls(
            ok       = True,
            dH       = d.get("dH"),
            dS       = d.get("dS"),
            dG       = d.get("dG"),
            Tm       = d.get("T_m_vH"),
            t1       = _arr(d.get("t1")),
            K        = _arr(d.get("K")),
            xdata    = _arr(d.get("xdata")),
            ydata    = _arr(d.get("ydata")),
            fit_line = _arr(d.get("fit_vh")),
        )

    def require(self) -> "VanHoffFit":
        if not self.ok:
            raise FitFailed(self.error or "van't Hoff fit failed")
        return self


@dataclass
class FullFit:
    """Non-linear least-squares fit of the full two-state sigmoid."""
    ok:         bool                 = False
    error:      Optional[str]        = None
    dH:         Optional[float]      = None
    dS:         Optional[float]      = None
    dG:         Optional[float]      = None
    Tm:         Optional[float]      = None
    base_b:     Optional[tuple]      = None
    base_ub:    Optional[tuple]      = None
    base_med:   Any                  = None
    curve:      Optional[np.ndarray] = None
    derivative: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "FullFit":
        if not d:
            return cls(ok=False, error="missing")
        if not d.get("success", False):
            return cls(ok=False, error=d.get("error", "unknown"))
        return cls(
            ok         = True,
            dH         = d.get("dH"),
            dS         = d.get("dS"),
            dG         = d.get("dG"),
            Tm         = d.get("T_m_fit"),
            base_b     = _tup(d.get("base_b_f")),
            base_ub    = _tup(d.get("base_ub_f")),
            base_med   = d.get("base_med_f"),
            curve      = _arr(d.get("fit")),
            derivative = _arr(d.get("derivative")),
        )

    def require(self) -> "FullFit":
        if not self.ok:
            raise FitFailed(self.error or "full-function fit failed")
        return self


# ── single-column result ────────────────────────────────────────────────────

@dataclass
class SingleResult:
    """Return value of `MeltAnalysis.single(...)`."""
    column:     str
    oligoC:     Optional[float]      = None
    c0:         Optional[float]      = None
    saltC:      Optional[float]      = None
    Tm_raw:     Optional[float]      = None
    T_used:     Optional[np.ndarray] = None
    T_all:      Optional[np.ndarray] = None
    signal:     Optional[np.ndarray] = None
    signal_all: Optional[np.ndarray] = None
    base_b_r:   Optional[tuple]      = None
    base_ub_r:  Optional[tuple]      = None
    vh:         VanHoffFit           = field(default_factory=VanHoffFit)
    fit:        FullFit              = field(default_factory=FullFit)
    error:      Optional[str]        = None
    _raw:       dict                 = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: dict) -> "SingleResult":
        err = d.get("error")
        if err:
            return cls(
                column = d.get("name", "?"),
                vh     = VanHoffFit(ok=False, error=err),
                fit    = FullFit(ok=False, error=err),
                error  = err,
                _raw   = d,
            )
        return cls(
            column     = d["name"],
            oligoC     = d.get("oligoC"),
            c0         = d.get("c0"),
            saltC      = d.get("saltC"),
            Tm_raw     = d.get("TmRaw"),
            T_used     = _arr(d.get("T_used")),
            T_all      = _arr(d.get("T_all")),
            signal     = _arr(d.get("signal")),
            signal_all = _arr(d.get("signal_all")),
            base_b_r   = _tup(d.get("base_b_r")),
            base_ub_r  = _tup(d.get("base_ub_r")),
            vh         = VanHoffFit.from_dict(d.get("vantHoff")),
            fit        = FullFit.from_dict(d.get("fit_result")),
            _raw       = d,
        )

    def to_dict(self) -> dict:
        """Original orchestrator dict — same shape the Pyodide bridge sees."""
        return self._raw

    def to_dataframe(self) -> pd.DataFrame:
        """One-row summary table (Python-friendly units: dS in kcal/(mol·K))."""
        return pd.DataFrame([{
            "column":    self.column,
            "oligoC_uM": self.oligoC,
            "saltC_mM":  self.saltC,
            "Tm_raw_C":  self.Tm_raw,
            "Tm_vH_C":   self.vh.Tm if self.vh.ok else None,
            "dH_vH":     self.vh.dH if self.vh.ok else None,
            "dS_vH":     self.vh.dS if self.vh.ok else None,
            "dG37_vH":   self.vh.dG if self.vh.ok else None,
            "Tm_fit_C":  self.fit.Tm if self.fit.ok else None,
            "dH_fit":    self.fit.dH if self.fit.ok else None,
            "dS_fit":    self.fit.dS if self.fit.ok else None,
            "dG37_fit":  self.fit.dG if self.fit.ok else None,
        }])

    def to_csv_row(self) -> list:
        """One CSV row in the browser-download per-column format."""
        vh  = self.vh  if self.vh.ok  else None
        fit = self.fit if self.fit.ok else None
        return [
            self.column,
            _cell(self.Tm_raw),
            _cell(vh.Tm  if vh  else None), _cell(vh.dH  if vh  else None),
            _cell(vh.dS  if vh  else None, 1000), _cell(vh.dG  if vh  else None),
            _cell(fit.Tm if fit else None), _cell(fit.dH if fit else None),
            _cell(fit.dS if fit else None, 1000), _cell(fit.dG if fit else None),
            "", "", "", "",   # multi columns empty in single mode
        ]

    def to_csv(self, path) -> None:
        """Browser-download-equivalent CSV (header + one row)."""
        _write_rows(path, [_PER_COL_HEADER, self.to_csv_row()])


# ── multi (shared-ΔH) result ────────────────────────────────────────────────

@dataclass
class MultiColumnResult:
    """One column inside a shared-ΔH joint fit."""
    name:       str
    oligoC:     Optional[float]      = None
    c0:         Optional[float]      = None
    Tm_fit:     Optional[float]      = None
    base_b:     Optional[tuple]      = None
    base_ub:    Optional[tuple]      = None
    base_med:   Any                  = None
    curve:      Optional[np.ndarray] = None
    signal:     Optional[np.ndarray] = None
    signal_all: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, d: dict) -> "MultiColumnResult":
        return cls(
            name       = d["name"],
            oligoC     = d.get("oligoC"),
            c0         = d.get("c0"),
            Tm_fit     = d.get("T_m_fit"),
            base_b     = _tup(d.get("base_b_f")),
            base_ub    = _tup(d.get("base_ub_f")),
            base_med   = d.get("base_med_f"),
            curve      = _arr(d.get("fit")),
            signal     = _arr(d.get("signal")),
            signal_all = _arr(d.get("signal_all")),
        )


@dataclass
class MultiResult:
    """Return value of `MeltAnalysis.multi(...)`."""
    saltC:   Optional[float]              = None
    dH:      Optional[float]              = None
    dS:      Optional[float]              = None
    dG:      Optional[float]              = None
    T_used:  Optional[np.ndarray]         = None
    T_all:   Optional[np.ndarray]         = None
    columns: list[MultiColumnResult]      = field(default_factory=list)
    error:   Optional[str]                = None
    _raw:    dict                         = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: dict) -> "MultiResult":
        err = d.get("error")
        if err:
            return cls(error=err, _raw=d)
        return cls(
            saltC   = d.get("saltC"),
            dH      = d.get("dH"),
            dS      = d.get("dS"),
            dG      = d.get("dG"),
            T_used  = _arr(d.get("T_used")),
            T_all   = _arr(d.get("T_all")),
            columns = [MultiColumnResult.from_dict(c) for c in d.get("columns", [])],
            _raw    = d,
        )

    def to_dict(self) -> dict:
        return self._raw

    def to_dataframe(self) -> pd.DataFrame:
        """One row per column; shared ΔH/ΔS/ΔG repeated."""
        return pd.DataFrame([{
            "column":    c.name,
            "oligoC_uM": c.oligoC,
            "c0_M":      c.c0,
            "Tm_fit_C":  c.Tm_fit,
            "dH":        self.dH,
            "dS":        self.dS,
            "dG37":      self.dG,
        } for c in self.columns])

    def to_csv_rows(self) -> list:
        """One CSV row per column — multi columns filled, raw/vh/fit blank."""
        rows = []
        for c in self.columns:
            rows.append([
                c.name,
                "",                       # Tm_raw_C
                "", "", "", "",           # vH block
                "", "", "", "",           # per-column fit block
                _cell(c.Tm_fit),
                _cell(self.dH),
                _cell(self.dS, 1000),
                _cell(self.dG),
            ])
        return rows

    def to_csv(self, path) -> None:
        """Browser-download-equivalent CSV (header + one row per column)."""
        _write_rows(path, [_PER_COL_HEADER, *self.to_csv_rows()])


# ── concentration-series result ─────────────────────────────────────────────

@dataclass
class ConcentrationCurve:
    """One curve in a concentration-series sweep — three Tm extractions."""
    name:       str
    oligoC:     Optional[float]      = None
    c0:         Optional[float]      = None
    Tm_raw:     Optional[float]      = None
    Tm_raw_K:   Optional[float]      = None
    Tm_vH:      Optional[float]      = None
    Tm_vH_K:    Optional[float]      = None
    Tm_fit:     Optional[float]      = None
    Tm_fit_K:   Optional[float]      = None
    lnCT:       Optional[float]      = None
    T_used:     Optional[np.ndarray] = None
    signal:     Optional[np.ndarray] = None
    signal_all: Optional[np.ndarray] = None
    base_b_r:   Optional[tuple]      = None
    base_ub_r:  Optional[tuple]      = None

    @classmethod
    def from_dict(cls, d: dict) -> "ConcentrationCurve":
        return cls(
            name       = d["name"],
            oligoC     = d.get("oligoC"),
            c0         = d.get("c0"),
            Tm_raw     = d.get("TmRaw"),
            Tm_raw_K   = d.get("TmRaw_K"),
            Tm_vH      = d.get("TmvH"),
            Tm_vH_K    = d.get("TmvH_K"),
            Tm_fit     = d.get("Tmfit"),
            Tm_fit_K   = d.get("Tmfit_K"),
            lnCT       = d.get("lnCT"),
            T_used     = _arr(d.get("T_used")),
            signal     = _arr(d.get("signal")),
            signal_all = _arr(d.get("signal_all")),
            base_b_r   = _tup(d.get("base_b_r")),
            base_ub_r  = _tup(d.get("base_ub_r")),
        )


@dataclass
class ConcentrationSeries:
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

    @classmethod
    def from_dict(cls, method: str, d: Optional[dict]) -> "ConcentrationSeries":
        if not d:
            return cls(method=method, ok=False, error="series missing or insufficient points")
        return cls(
            method      = method,
            ok          = bool(d.get("success", False)),
            n           = d.get("n"),
            dH          = d.get("dH"),
            dS          = d.get("dS"),
            dG_37       = d.get("dG_37"),
            r_squared   = d.get("r_squared"),
            slope       = d.get("slope"),
            intercept   = d.get("intercept"),
            ln_ct       = _arr(d.get("ln_ct")),
            inv_tm      = _arr(d.get("inv_tm")),
            ln_ct_line  = _arr(d.get("ln_ct_line")),
            inv_tm_line = _arr(d.get("inv_tm_line")),
        )

    def require(self) -> "ConcentrationSeries":
        if not self.ok:
            raise FitFailed(self.error or f"{self.method} series failed")
        return self


@dataclass
class ConcentrationResult:
    """Return value of `MeltAnalysis.concentration(...)`."""
    struct_type:        str                       = "?"
    self_complementary: bool                      = False
    saltC:              Optional[float]           = None
    T_all:              Optional[np.ndarray]      = None
    T_used:             Optional[np.ndarray]      = None
    per_curve:          list[ConcentrationCurve]  = field(default_factory=list)
    skipped:            list[dict]                = field(default_factory=list)
    series_raw:         ConcentrationSeries       = field(
        default_factory=lambda: ConcentrationSeries(method="raw"))
    series_vh:          ConcentrationSeries       = field(
        default_factory=lambda: ConcentrationSeries(method="vh"))
    series_fit:         ConcentrationSeries       = field(
        default_factory=lambda: ConcentrationSeries(method="fit"))
    error:              Optional[str]             = None
    _raw:               dict                      = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: dict) -> "ConcentrationResult":
        series = d.get("series") or {}
        return cls(
            struct_type        = d.get("struct_type", "?"),
            self_complementary = bool(d.get("self_complementary", False)),
            saltC              = d.get("saltC"),
            T_all              = _arr(d.get("T_all")),
            T_used             = _arr(d.get("T_used")),
            per_curve          = [ConcentrationCurve.from_dict(c) for c in d.get("per_curve", [])],
            skipped            = list(d.get("skipped", [])),
            series_raw         = ConcentrationSeries.from_dict("raw", series.get("raw")),
            series_vh          = ConcentrationSeries.from_dict("vh",  series.get("vh")),
            series_fit         = ConcentrationSeries.from_dict("fit", series.get("fit")),
            error              = d.get("error"),
            _raw               = d,
        )

    def to_dict(self) -> dict:
        return self._raw

    @property
    def series(self) -> dict[str, ConcentrationSeries]:
        """Indexed view: `{"raw": ..., "vh": ..., "fit": ...}`."""
        return {"raw": self.series_raw, "vh": self.series_vh, "fit": self.series_fit}

    def to_dataframe(self) -> pd.DataFrame:
        """Per-curve table — one row per concentration, three Tm columns."""
        return pd.DataFrame([{
            "column":    c.name,
            "oligoC_uM": c.oligoC,
            "c0_M":      c.c0,
            "lnCT":      c.lnCT,
            "Tm_raw_C":  c.Tm_raw,
            "Tm_vH_C":   c.Tm_vH,
            "Tm_fit_C":  c.Tm_fit,
        } for c in self.per_curve])

    def series_dataframe(self) -> pd.DataFrame:
        """One row per regression method (raw / vh / fit)."""
        return pd.DataFrame([{
            "method":    s.method,
            "ok":        s.ok,
            "n":         s.n,
            "slope":     s.slope,
            "intercept": s.intercept,
            "dH":        s.dH,
            "dS":        s.dS,
            "dG_37":     s.dG_37,
            "r_squared": s.r_squared,
        } for s in (self.series_raw, self.series_vh, self.series_fit)])

    def to_csv(self, path) -> None:
        """Browser-download-equivalent multi-block CSV.

        Layout: header line → per-curve table → blank → series table →
        blank → skipped-columns table (only if any were skipped).
        """
        f_factor = 1 if self.self_complementary else 4
        lines = [
            [f"# concentration series (struct={self.struct_type}, "
             f"self_complementary={self.self_complementary}, f={f_factor})"],
            ["column", "CT_uM", "CT_M", "ln_CT_over_f",
             "Tm_raw_C", "inv_Tm_raw_K",
             "Tm_vH_C",  "inv_Tm_vH_K",
             "Tm_fit_C", "inv_Tm_fit_K"],
        ]
        for c in self.per_curve:
            lines.append([
                c.name,
                _cell(c.oligoC),
                _cell(c.c0),
                _cell(c.lnCT),
                _cell(c.Tm_raw), _inv_K(c.Tm_raw_K),
                _cell(c.Tm_vH),  _inv_K(c.Tm_vH_K),
                _cell(c.Tm_fit), _inv_K(c.Tm_fit_K),
            ])

        lines.append([])
        lines.append([
            "method", "n", "slope_K^-1", "intercept_K^-1",
            "dH_kcal_mol", "dS_cal_molK", "dG37_kcal_mol", "r_squared",
        ])
        for s, label in (
            (self.series_raw, "Tm_raw"),
            (self.series_vh,  "Tm_vH"),
            (self.series_fit, "Tm_fit"),
        ):
            if not s.ok:
                lines.append([label, "0", "", "", "", "", "", ""])
                continue
            lines.append([
                label,
                str(s.n if s.n is not None else ""),
                _cell(s.slope),
                _cell(s.intercept),
                _cell(s.dH),
                _cell(s.dS, 1000),
                _cell(s.dG_37),
                _cell(s.r_squared),
            ])

        if self.skipped:
            lines.append([])
            lines.append(["skipped_column", "reason"])
            for entry in self.skipped:
                lines.append([entry.get("name", ""), entry.get("reason", "")])

        _write_rows(path, lines)


# ── batch result (CLI: every signal column at the same oligo) ───────────────

@dataclass
class BatchResult:
    """Thin wrapper around `dict[str, SingleResult]` for serialisation.

    Produced by the CLI when invoked without `--column`; consumed for
    JSON / CSV emission. The Python class API just returns a plain dict
    from `MeltAnalysis.single_all`.
    """
    columns: list[str]
    results: dict[str, SingleResult] = field(default_factory=dict)
    error:   Optional[str]           = None

    def to_dict(self) -> dict:
        return {
            "is_batch": True,
            "columns":  list(self.columns),
            "results":  {k: v.to_dict() for k, v in self.results.items()},
        }

    def to_dataframe(self) -> pd.DataFrame:
        """One row per column (concat of each SingleResult's to_dataframe)."""
        if not self.results:
            return pd.DataFrame()
        return pd.concat([r.to_dataframe() for r in self.results.values()],
                         ignore_index=True)

    def to_csv(self, path) -> None:
        """Browser-download-equivalent CSV (header + one row per column)."""
        rows = [_PER_COL_HEADER]
        for r in self.results.values():
            rows.append(r.to_csv_row())
        _write_rows(path, rows)


__all__ = [
    "FitFailed",
    "VanHoffFit",
    "FullFit",
    "SingleResult",
    "MultiColumnResult",
    "MultiResult",
    "ConcentrationCurve",
    "ConcentrationSeries",
    "ConcentrationResult",
    "BatchResult",
]
