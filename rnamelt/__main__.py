"""CLI entry point for the RNA melting analysis pipeline.

Usage
-----
    rnamelt FILE.csv [options]              # installed via pip
    python -m rnamelt FILE.csv [options]    # equivalent module form

The CLI loads the CSV, runs `rnamelt.cleaning.clean`, and dispatches to
the appropriate `MeltAnalysis` mode. The full result is printed as JSON
on stdout; pass `--csv-out PATH` to also write a results CSV in the
format produced by the browser download.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

import pandas as pd

from rnamelt import BatchResult, MeltAnalysis
from rnamelt.cleaning import clean


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
        prog="rnamelt",
        description="RNA melting curve analysis — single column, multi (shared ΔH), "
                    "or concentration-series van't Hoff.",
    )
    p.add_argument("csv", type=Path, help="CSV: temperature column + one or more signal columns")
    p.add_argument(
        "--column", default=None,
        help="Signal column to fit. Omit to run single-mode analysis "
             "(van't Hoff + full fit) on every signal column in the CSV. "
             "Pass '__multi__' for the shared-ΔH joint fit (defaults to all "
             "columns at --oligo if no --oligo-multi is given) or "
             "'__concentration__' for the 1/Tm vs ln(C_T/f) series "
             "(requires --oligo-multi).",
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

    # ── scipy.optimize.least_squares overrides (full-function fit) ─────
    s = p.add_argument_group(
        "solver options",
        "Overrides forwarded to scipy.optimize.least_squares for the full "
        "two-state nonlinear fit. Unset flags fall back to defaults tuned "
        "for in-browser use (see rnamelt.methods.SOLVER_DEFAULTS).",
    )
    s.add_argument("--max-nfev", type=int, default=None,
                   help="maximum function evaluations. Default: 10000 (browser-safe)")
    s.add_argument("--ftol", type=float, default=None,
                   help="cost-function tolerance. Default: 1e-8")
    s.add_argument("--gtol", type=float, default=None,
                   help="gradient tolerance. Default: 1e-8")
    s.add_argument("--xtol", type=float, default=None,
                   help="step-size tolerance. Default: 1e-8")
    s.add_argument("--method", default=None, choices=["trf", "dogbox", "lm"],
                   help="trust-region algorithm. Default: trf. "
                        "Note: 'lm' does not support bounds and will error here.")
    s.add_argument("--loss", default=None,
                   choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
                   help="residual loss function. Default: linear (least squares). "
                        "Use soft_l1/huber for outlier-robust fits.")
    s.add_argument("--f-scale", type=float, default=None,
                   help="soft-margin transition value for non-linear losses. Default: 1.0")
    s.add_argument("--jac", default=None, choices=["2-point", "3-point", "cs"],
                   help="Jacobian estimation scheme. Default: 2-point")
    s.add_argument("--solver-verbose", type=int, default=None, choices=[0, 1, 2],
                   help="0 silent, 1 termination report, 2 per-iteration. Default: 0")
    s.add_argument("--residuals-method", default=None,
                   choices=["linear", "square", "cubic", "quadratic"],
                   help="multi-fit only: how per-curve residuals are aggregated "
                        "into the scalar passed to least_squares. Default: square. "
                        "Ignored by single-curve and concentration-series fits.")

    # ── van't Hoff linearisation overrides ────────────────────────────
    v = p.add_argument_group(
        "van't Hoff options",
        "Overrides for the linear-fit step of the van't Hoff analysis "
        "(see rnamelt.methods.VH_DEFAULTS). Ignored by the multi-fit mode.",
    )
    v.add_argument("--vh-border", type=float, default=None,
                   help="fraction-folded cutoff θ; points with θ outside "
                        "[border, 1-border] are excluded from the regression. "
                        "Default: 0.15")
    v.add_argument("--vh-t1-min", type=float, default=None,
                   help="lower clip on the linearised x-axis "
                        "(T_scale/(T_C-T0) units; -1 = no clip). Default: -1")
    v.add_argument("--vh-t1-max", type=float, default=None,
                   help="upper clip on the linearised x-axis "
                        "(T_scale/(T_C-T0) units; -1 = no clip). Default: -1")
    v.add_argument("--vh-T-scale", type=float, default=None,
                   help="numerical conditioning factor for 1/T. Default: 1000")

    # ── Full-fit initial guesses ──────────────────────────────────────
    fi = p.add_argument_group(
        "fit-init options",
        "Initial guesses for the full two-state nonlinear fit "
        "(see rnamelt.methods.FIT_INIT_DEFAULTS). When --dH-init / --dS-init "
        "are not set, ΔH / ΔS are auto-seeded from the van't Hoff result "
        "(or fall back to -80 / -0.2 if van't Hoff fails). "
        "Setting them here overrides that auto-seeding.",
    )
    fi.add_argument("--dH-init", type=float, default=None,
                    help="ΔH initial guess in kcal/mol")
    fi.add_argument("--dS-init", type=float, default=None,
                    help="ΔS initial guess in kcal/(mol·K)")
    fi.add_argument("--lin-init", type=int, default=None,
                    help="number of leading/trailing data points averaged "
                         "for the baseline-intercept seed. Default: 10")
    return p


def _build_solver(args):
    raw = {
        "max_nfev":         args.max_nfev,
        "ftol":             args.ftol,
        "gtol":             args.gtol,
        "xtol":             args.xtol,
        "method":           args.method,
        "loss":             args.loss,
        "f_scale":          args.f_scale,
        "jac":              args.jac,
        "verbose":          args.solver_verbose,
        "residuals_method": args.residuals_method,
    }
    return {k: v for k, v in raw.items() if v is not None}


def _build_vh(args):
    raw = {
        "border":  args.vh_border,
        "t1_min":  args.vh_t1_min,
        "t1_max":  args.vh_t1_max,
        "T_scale": args.vh_T_scale,
    }
    return {k: v for k, v in raw.items() if v is not None}


def _build_fit_init(args):
    raw = {
        "dH_init":  args.dH_init,
        "dS_init":  args.dS_init,
        "lin_init": args.lin_init,
        "b1_init":  args.lin_init,
        "b2_init":  args.lin_init,
    }
    return {k: v for k, v in raw.items() if v is not None}


def _build_analyzer(df, args) -> MeltAnalysis:
    return MeltAnalysis(
        df,
        struct_type     = args.struct_type,
        salt            = args.salt,
        T_low           = args.T_low,
        T_high          = args.T_high,
        bl_lower_offset = args.bl_lower,
        bl_upper_offset = args.bl_upper,
        solver   = _build_solver(args)   or None,
        vh       = _build_vh(args)       or None,
        fit_init = _build_fit_init(args) or None,
    )


# ── Entry point ───────────────────────────────────────────────────────────

def main(argv=None):
    args = _build_argparser().parse_args(argv)

    if not args.csv.is_file():
        print(f"error: CSV not found: {args.csv}", file=sys.stderr)
        return 2

    df = clean(pd.read_csv(args.csv))
    oligo_multi = _parse_oligo_multi(args.oligo_multi)

    m = _build_analyzer(df, args)
    sigs = m.signal_columns

    # The orchestrator emits debug prints; route them to stderr so stdout stays JSON.
    with contextlib.redirect_stdout(sys.stderr):
        if args.column == "__multi__":
            if not oligo_multi:
                if not sigs:
                    print("error: CSV has no signal columns", file=sys.stderr)
                    return 2
                oligo_multi = {c: args.oligo for c in sigs}
            result = m.multi(oligo_multi)
        elif args.column == "__concentration__":
            if not oligo_multi:
                print("error: --column __concentration__ needs at least one "
                      "--oligo-multi NAME=VAL (concentration must vary across columns)",
                      file=sys.stderr)
                return 2
            result = m.concentration(oligo_multi)
        elif args.column is not None:
            result = m.single(args.column, oligo=args.oligo)
        else:
            if not sigs:
                print("error: CSV has no signal columns", file=sys.stderr)
                return 2
            per_col = m.single_all(oligo=args.oligo)
            result = BatchResult(columns=sigs, results=per_col)

    indent = args.indent if args.indent and args.indent > 0 else None
    json.dump(result.to_dict(), sys.stdout, indent=indent, default=str)
    sys.stdout.write("\n")

    top_error = getattr(result, "error", None)
    if top_error:
        print(f"error: {top_error}", file=sys.stderr)
        return 1

    if isinstance(result, BatchResult):
        failures = [n for n, r in result.results.items() if r.error]
        if failures and len(failures) == len(result.results):
            print(f"error: all {len(failures)} columns failed", file=sys.stderr)
            return 1
        if failures:
            print(f"warning: {len(failures)}/{len(result.results)} columns errored: "
                  f"{', '.join(failures)}", file=sys.stderr)

    if args.csv_out:
        result.to_csv(args.csv_out)
        print(f"wrote {args.csv_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
