"""End-to-end tests for analysis.analysis_melting.run — concentration-series mode."""
import os
import sys
import unittest

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import analysis_melting, functions, constants


def _expected_tm_kelvin(dH: float, dS: float, ct: float, f: float) -> float:
    """Invert 1/Tm = (R/dH)·ln(C_T/f) + dS/dH for a known (dH, dS, C_T, f)."""
    return 1.0 / ((constants.R / dH) * np.log(ct / f) + (dS / dH))


def _synthesize_curve(T_celsius, dH: float, dS: float, c0: float,
                      structType: str = "heterodimer",
                      m_bound=0.0, b_bound=0.0, m_unbound=0.0, b_unbound=1.0):
    """Two-state sigmoid via functions.full_function. Defaults to an increasing
    curve (b_bound=0 at low T → b_unbound=1 at high T) since the existing
    intersect_lin only handles upward-crossing data."""
    # functions.full_function signature is (T, dH, dS, m1, b1, m2, b2, c0=...)
    # where (m1, b1) is the bound (low-T) baseline and (m2, b2) is the
    # unbound (high-T) baseline.
    return np.array([
        functions.full_function(t, dH, dS, m_bound, b_bound, m_unbound, b_unbound,
                                c0=c0, structType=structType)
        for t in T_celsius
    ])


def _build_df(temps, curves):
    df = pd.DataFrame({"temperature": temps})
    for name, y in curves.items():
        df[name] = y
    return df


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed in this environment")
class TestConcentrationSeriesOrchestrator(unittest.TestCase):

    def setUp(self):
        # 0.5 °C grid from 5 °C to 95 °C — wide enough for clean baselines
        # at all concentrations in the series.
        self.T = np.arange(5.0, 95.01, 0.5)
        self.dH = -80.0
        self.dS = -0.22
        # User enters per-strand concentration in µM; code multiplies by 2 to
        # produce c0 (total strand, M). Pick µM values that span 3 decades.
        self.oligo_uM = {"c1": 0.5, "c2": 5.0, "c3": 50.0, "c4": 500.0}

    def _build_concentration_series(self, structType: str = "heterodimer"):
        curves = {}
        for name, oligo in self.oligo_uM.items():
            c0 = 1e-6 * oligo * 2  # match orchestrator convention
            curves[name] = _synthesize_curve(
                self.T, self.dH, self.dS, c0, structType=structType,
            )
        return _build_df(self.T, curves)

    def test_recovers_dH_and_dS_from_synthetic_series(self):
        df = self._build_concentration_series()
        params = {
            "column":          "concentration",
            "struct_type":     "heterodimer",
            "T_low":           self.T[0],
            "T_high":          self.T[-1],
            "bl_lower_offset": 15.0,
            "bl_upper_offset": 15.0,
            "oligo_multi":     self.oligo_uM,
            "salt":            150,
        }
        result = analysis_melting.run(df, params)

        self.assertNotIn("error", result, msg=result.get("error"))
        self.assertTrue(result["is_concentration"])
        self.assertEqual(len(result["per_curve"]), 4)
        self.assertEqual(result["skipped"], [])

        vh = result["vantHoff"]
        self.assertTrue(vh["success"])
        # Tm extraction by baseline intersection has small residual error;
        # 1 kcal/mol on ΔH, 0.005 kcal/mol/K on ΔS is a fair tolerance.
        self.assertAlmostEqual(vh["dH"], self.dH, delta=1.0)
        self.assertAlmostEqual(vh["dS"], self.dS, delta=5e-3)
        self.assertGreater(vh["r_squared"], 0.999)

        # Per-curve Tm should match the analytic expectation in Celsius.
        for entry in result["per_curve"]:
            ct = entry["c0"]
            expected_tm_C = _expected_tm_kelvin(self.dH, self.dS, ct, f=4.0) + constants.T0
            self.assertAlmostEqual(entry["TmRaw"], expected_tm_C, delta=0.5)

    def test_monomer_is_rejected(self):
        df = self._build_concentration_series()
        params = {
            "column":      "concentration",
            "struct_type": "monomer",
            "oligo_multi": self.oligo_uM,
        }
        result = analysis_melting.run(df, params)
        self.assertIn("error", result)
        self.assertIn("monomer", result["error"].lower())

    def test_too_few_valid_points(self):
        df = self._build_concentration_series()
        params = {
            "column":          "concentration",
            "struct_type":     "heterodimer",
            "T_low":           self.T[0],
            "T_high":          self.T[-1],
            "bl_lower_offset": 15.0,
            "bl_upper_offset": 15.0,
            # only one column has a concentration → only one (Tm, C_T) point
            "oligo_multi":     {"c1": 0.5},
        }
        result = analysis_melting.run(df, params)
        self.assertIn("error", result)
        self.assertEqual(len(result["skipped"]), 3)

    def test_homodimer_uses_self_complementary_branch(self):
        df = self._build_concentration_series(structType="homodimer")
        params = {
            "column":          "concentration",
            "struct_type":     "homodimer",
            "T_low":           self.T[0],
            "T_high":          self.T[-1],
            "bl_lower_offset": 15.0,
            "bl_upper_offset": 15.0,
            "oligo_multi":     self.oligo_uM,
        }
        result = analysis_melting.run(df, params)
        self.assertNotIn("error", result, msg=result.get("error"))
        self.assertTrue(result["self_complementary"])
        self.assertAlmostEqual(result["vantHoff"]["dH"], self.dH, delta=3.0)


if __name__ == "__main__":
    unittest.main()
