"""rnamelt — RNA / DNA melting-curve thermodynamic analysis.

Public API:
    SingleCurve  — one melting curve; raw Tm, van't Hoff and full fit
    MultiCurves  — a sequence of curves; global fit, concentration series

    from rnamelt import SingleCurve, MultiCurves

    c = SingleCurve(temps, signal, oligoC=0.5)
    c.raw_tm()                         # → RawTm
    c.vant_hoff()                       # → VanHoffFit
    c.fit()                            # → SingleFit

    m = MultiCurves([c1, c2, c3])
    m.global_fit()                     # → GlobalFit
    m.conc_dependency()                # → ConcDepResult

Curves are independent: each carries its own concentration and baseline
windows (given, or auto-detected when omitted), so a MultiCurves may hold
curves of different lengths.

Every tunable default is a field of one settings class: WindowSettings,
SolverSettings, VanHoffSettings, FitInitSettings, BoundsSettings,
MultistartSettings and RawSettings. Override them per curve with the
matching `windows=`, `solver=`, `vh=`, `fit_init=`, `bounds=`,
`multistart=` and `raw=` arguments of SingleCurve (an instance or a dict of
overrides), or per call on its analysis methods.
"""

from rnamelt.api import (
    SingleCurve,
    MultiCurves,
)
from rnamelt.methods import (
    BOUNDS_MODES,
    SEED_POLICIES,
    BoundsSettings,
    FitInitSettings,
    MultistartSettings,
    RawSettings,
    SolverSettings,
    VanHoffSettings,
    WindowSettings,
    BaselineWarning,
    ConvergenceWarning,
)
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
    # settings
    "SolverSettings",
    "VanHoffSettings",
    "FitInitSettings",
    "BoundsSettings",
    "MultistartSettings",
    "RawSettings",
    "WindowSettings",
    "BOUNDS_MODES",
    "SEED_POLICIES",
    "BaselineWarning",
    "ConvergenceWarning",
]
