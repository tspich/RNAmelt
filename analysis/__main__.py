"""CLI entry point for the RNA melting analysis pipeline.

Usage
-----
    python -m analysis FILE.csv [options]

The CLI loads the CSV, runs `analysis.cleaning.clean`, and dispatches to
the appropriate `analysis.api` function. The full result is printed as
JSON on stdout; pass `--csv-out PATH` to also write a results CSV in the
format produced by the browser download.
"""

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path

import pandas as pd

from analysis import analyze_concentration, analyze_multi, analyze_single
from analysis.cleaning import clean, get_signal_columns


def _parse_oligo_multi(values):
    """Parse repeated --oligo NAME=VAL into {NAME: float µM}."""
    out = {}
    for v in values or []:
        if "=" not in v:
            raise argparse.ArgumentTypeError(
                f"--oligo expects NAME=VALUE in multi/concentration mode, got {v!r}"
            )
        name, val = v.split("=", 1)
        try:
            out[name.strip()] = float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(f"non-numeric concentration: {val!r}")
    return out


def _build_argparser():
    p = argparse.ArgumentParser(
        prog="python -m analysis",
        description="RNA melting curve analysis — single column, multi (shared ΔH), "
                    "or concentration-series van't Hoff.",
    )
    p.add_argument("csv", type=Path, help="CSV: temperature column + one or more signal columns")
    p.add_argument(
        "--column", default=None,
        help="Signal column to fit. Use '__multi__' for shared-ΔH fit across all columns "
             "or '__concentration__' for the 1/Tm vs ln(C_T/f) series. "
             "Default: first signal column.",
    )
    p.add_argument("--struct-type", default="heterodimer",
                   choices=["heterodimer", "homodimer", "monomer"])
    p.add_argument("--signal-type", default="absorbance",
                   choices=["absorbance", "fluorescence"])
    p.add_argument("--T-low",  type=float, default=None, help="lower transition-window edge (°C)")
    p.add_argument("--T-high", type=float, default=None, help="upper transition-window edge (°C)")
    p.add_argument("--bl-lower", type=float, default=10.0,
                   help="folded-baseline offset above T_low (°C). Default: 10")
    p.add_argument("--bl-upper", type=float, default=10.0,
                   help="unfolded-baseline offset below T_high (°C). Default: 10")
    p.add_argument("--salt", type=float, default=150.0, help="NaCl concentration (mM). Default: 150")
    p.add_argument("--oligo", type=float, default=0.5,
                   help="single-column strand concentration in µM. Default: 0.5")
    p.add_argument(
        "--oligo-multi", action="append", default=None, metavar="NAME=VAL",
        help="per-column concentration (µM) for __multi__ / __concentration__ modes, "
             "e.g. --oligo-multi c1=0.5. Repeat once per column.",
    )
    p.add_argument("--csv-out", type=Path, default=None,
                   help="write a results CSV (same format as the browser download)")
    p.add_argument("--indent", type=int, default=2,
                   help="JSON indent for stdout (0 = compact). Default: 2")
    return p


def _common_kwargs(args):
    """Kwargs shared by analyze_single / analyze_multi / analyze_concentration."""
    return {
        "struct_type":     args.struct_type,
        "signal_type":     args.signal_type,
        "salt":            args.salt,
        "T_low":           args.T_low,
        "T_high":          args.T_high,
        "bl_lower_offset": args.bl_lower,
        "bl_upper_offset": args.bl_upper,
    }


# ── CSV writer mirroring the browser download ─────────────────────────────

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


def _per_column_rows(result):
    """Return (column_name -> {TmRaw, vh, fit, multi}) for a single run."""
    rows = {}
    if result.get("is_multi"):
        for c in result.get("columns", []):
            rows[c["name"]] = {
                "TmRaw": None, "vh": None, "fit": None,
                "multi": {
                    "T_m_fit": c.get("T_m_fit"),
                    "dH": result.get("dH"),
                    "dS": result.get("dS"),
                    "dG": result.get("dG"),
                },
            }
    elif not result.get("is_concentration"):
        # single-column run
        name = result.get("name")
        if name:
            vh  = result.get("vantHoff")   or {}
            fit = result.get("fit_result") or {}
            rows[name] = {
                "TmRaw": result.get("TmRaw"),
                "vh":    vh  if vh.get("success")  else None,
                "fit":   fit if fit.get("success") else None,
                "multi": None,
            }
    return rows


def _write_csv(result, path: Path):
    """Mirror the JS download: per-column table + optional concentration block."""
    lines = []

    rows = _per_column_rows(result)
    if rows:
        lines.append([
            "column",
            "Tm_raw_C",
            "Tm_vH_C", "dH_vH_kcal_mol", "dS_vH_cal_molK", "dG37_vH_kcal_mol",
            "Tm_fit_C", "dH_fit_kcal_mol", "dS_fit_cal_molK", "dG37_fit_kcal_mol",
            "Tm_multi_C", "dH_multi_kcal_mol", "dS_multi_cal_molK", "dG37_multi_kcal_mol",
        ])
        for name, e in rows.items():
            vh = e.get("vh") or {}
            fit = e.get("fit") or {}
            multi = e.get("multi") or {}
            lines.append([
                name,
                _cell(e.get("TmRaw")),
                _cell(vh.get("T_m_vH")),    _cell(vh.get("dH")),    _cell(vh.get("dS"), 1000),    _cell(vh.get("dG")),
                _cell(fit.get("T_m_fit")),  _cell(fit.get("dH")),   _cell(fit.get("dS"), 1000),   _cell(fit.get("dG")),
                _cell(multi.get("T_m_fit")),_cell(multi.get("dH")), _cell(multi.get("dS"), 1000), _cell(multi.get("dG")),
            ])

    if result.get("is_concentration"):
        series = result.get("series") or {"raw": result.get("vantHoff"), "vh": None, "fit": None}
        f = 1 if result.get("self_complementary") else 4
        if lines:
            lines.append([])
        lines.append([
            f"# concentration series (struct={result.get('struct_type')}, "
            f"self_complementary={result.get('self_complementary')}, f={f})"
        ])
        lines.append([
            "column", "CT_uM", "CT_M", "ln_CT_over_f",
            "Tm_raw_C", "inv_Tm_raw_K",
            "Tm_vH_C",  "inv_Tm_vH_K",
            "Tm_fit_C", "inv_Tm_fit_K",
        ])
        for c in result.get("per_curve", []):
            lines.append([
                c.get("name"),
                _cell(c.get("oligoC")),
                _cell(c.get("c0")),
                _cell(c.get("lnCT")),
                _cell(c.get("TmRaw")),    _inv_K(c.get("TmRaw_K") or c.get("TmKelvin")),
                _cell(c.get("TmvH")),     _inv_K(c.get("TmvH_K")),
                _cell(c.get("Tmfit")),    _inv_K(c.get("Tmfit_K")),
            ])
        lines.append([])
        lines.append([
            "method", "n", "slope_K^-1", "intercept_K^-1",
            "dH_kcal_mol", "dS_cal_molK", "dG37_kcal_mol", "r_squared",
        ])
        for key, label in (("raw", "Tm_raw"), ("vh", "Tm_vH"), ("fit", "Tm_fit")):
            s = series.get(key)
            if not s:
                lines.append([label, "0", "", "", "", "", "", ""])
                continue
            lines.append([
                label,
                str(s.get("n", "")),
                _cell(s.get("slope")),
                _cell(s.get("intercept")),
                _cell(s.get("dH")),
                _cell(s.get("dS"), 1000),
                _cell(s.get("dG_37")),
                _cell(s.get("r_squared")),
            ])
        skipped = result.get("skipped") or []
        if skipped:
            lines.append([])
            lines.append(["skipped_column", "reason"])
            for s in skipped:
                lines.append([s.get("name", ""), s.get("reason", "")])

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        for row in lines:
            w.writerow(row)


# ── Entry point ───────────────────────────────────────────────────────────

def main(argv=None):
    args = _build_argparser().parse_args(argv)

    if not args.csv.is_file():
        print(f"error: CSV not found: {args.csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)
    df = clean(df)
    oligo_multi = _parse_oligo_multi(args.oligo_multi)
    common = _common_kwargs(args)

    # The orchestrator emits debug prints; route them to stderr so stdout stays JSON.
    with contextlib.redirect_stdout(sys.stderr):
        if args.column == "__multi__":
            if not oligo_multi:
                print("error: --column __multi__ needs at least one --oligo-multi NAME=VAL", file=sys.stderr)
                return 2
            result = analyze_multi(df, oligo_multi, **common)
        elif args.column == "__concentration__":
            if not oligo_multi:
                print("error: --column __concentration__ needs at least one --oligo-multi NAME=VAL", file=sys.stderr)
                return 2
            result = analyze_concentration(df, oligo_multi, **common)
        else:
            column = args.column
            if column is None:
                sigs = get_signal_columns(df)
                if not sigs:
                    print("error: CSV has no signal columns", file=sys.stderr)
                    return 2
                column = sigs[0]
            result = analyze_single(df, column, oligo=args.oligo, **common)

    indent = args.indent if args.indent and args.indent > 0 else None
    json.dump(result, sys.stdout, indent=indent, default=str)
    sys.stdout.write("\n")

    if isinstance(result, dict) and result.get("error"):
        print(f"error: {result['error']}", file=sys.stderr)
        return 1

    if args.csv_out:
        _write_csv(result, args.csv_out)
        print(f"wrote {args.csv_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
