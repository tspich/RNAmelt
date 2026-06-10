"""rnamelt — RNA / DNA melting-curve thermodynamic analysis.

Public API:
    MeltAnalysis                                          — class façade
    analyze_single, analyze_multi, analyze_concentration  — DataFrame in
    analyze_csv                                           — path in
"""

from rnamelt.api import (
    MeltAnalysis,
    analyze_concentration,
    analyze_csv,
    analyze_multi,
    analyze_single,
)
from rnamelt.methods import FIT_INIT_DEFAULTS, SOLVER_DEFAULTS, VH_DEFAULTS
from rnamelt.results import (
    BatchResult,
    ConcentrationCurve,
    ConcentrationResult,
    ConcentrationSeries,
    FitFailed,
    FullFit,
    MultiColumnResult,
    MultiResult,
    SingleResult,
    VanHoffFit,
)

ANALYSES = [
    "rnamelt.analysis_melting",
]

__all__ = [
    "MeltAnalysis",
    "analyze_single",
    "analyze_multi",
    "analyze_concentration",
    "analyze_csv",
    "SingleResult",
    "MultiResult",
    "MultiColumnResult",
    "ConcentrationResult",
    "ConcentrationCurve",
    "ConcentrationSeries",
    "BatchResult",
    "VanHoffFit",
    "FullFit",
    "FitFailed",
    "SOLVER_DEFAULTS",
    "VH_DEFAULTS",
    "FIT_INIT_DEFAULTS",
    "ANALYSES",
]
