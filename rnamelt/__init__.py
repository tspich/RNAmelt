"""rnamelt — RNA / DNA melting-curve thermodynamic analysis.

Public API:
    SingleCurve  — one melting curve; raw Tm, van't Hoff and full fit
    MultiCurves  — a sequence of curves; global fit, concentration series

    from rnamelt import SingleCurve, MultiCurves

    c = SingleCurve(temps, signal, oligo=0.5)
    c.raw_tm()                         # → RawTm
    c.vantHoff()                       # → VanHoffFit
    c.fit()                            # → SingleFit

    m = MultiCurves([c1, c2, c3])
    m.global_fit()                     # → GlobalFit
    m.conc_dependency()                # → ConcDepResult

Curves are independent: each carries its own concentration, transition
window and baseline offsets, so a MultiCurves may hold curves of
different lengths.

Solver, van't Hoff and initial-guess defaults are exposed as
SOLVER_DEFAULTS, VH_DEFAULTS and FIT_INIT_DEFAULTS; override them per
curve through the matching `solver=`, `vh=` and `fit_init=` arguments.
"""

from rnamelt.api import (
    SingleCurve,
    MultiCurves,
)
from rnamelt.methods import FIT_INIT_DEFAULTS, SOLVER_DEFAULTS, VH_DEFAULTS
from rnamelt.results import (
    RawTm,
    VanHoffFit,
    SingleFit,
    GlobalFit,
    GlobalFitCurve,
    ConcDepResult,
    ConcentrationCurve,
    ConcentrationSeries,
)

#ANALYSES = [
#    "rnamelt.analysis_melting",
#]

__all__ = [
    # analysis entry points
    "SingleCurve",
    "MultiCurves",
    # per-curve results
    "RawTm",
    "VanHoffFit",
    "SingleFit",
    # multi-curve results
    "GlobalFit",
    "GlobalFitCurve",
    "ConcDepResult",
    "ConcentrationCurve",
    "ConcentrationSeries",
    # tunable defaults
    "SOLVER_DEFAULTS",
    "VH_DEFAULTS",
    "FIT_INIT_DEFAULTS",
]
