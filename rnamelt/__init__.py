"""rnamelt — RNA / DNA melting-curve thermodynamic analysis.

Public API:
    MeltAnalysis                                          — class façade
    analyze_single, analyze_multi, analyze_concentration  — DataFrame in
    analyze_csv                                           — path in
"""

from rnamelt.api import (
    SingleCurve,
    MultiCurves,
)
from rnamelt.methods import FIT_INIT_DEFAULTS, SOLVER_DEFAULTS, VH_DEFAULTS
from rnamelt.results import (
    VanHoffFit,
    RawTm,
)

#ANALYSES = [
#    "rnamelt.analysis_melting",
#]

__all__ = [
    "RawTm",
    "VanHoffFit",
    "SingleCurve",
    "MultiCurves",
    "SOLVER_DEFAULTS",
    "VH_DEFAULTS",
    "FIT_INIT_DEFAULTS",
]
