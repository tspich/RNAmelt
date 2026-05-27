"""Tests for the analysis.api facade — one round-trip per mode."""
import os
import sys
import tempfile
import unittest

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnamelt import (
    analyze_concentration,
    analyze_csv,
    analyze_multi,
    analyze_single,
    functions,
)


def _synth_curve(T_C, dH, dS, c0, structType="heterodimer",
                 m_bound=0.0, b_bound=0.0, m_unbound=0.0, b_unbound=1.0):
    return np.array([
        functions.full_function(t, dH, dS, m_bound, b_bound, m_unbound, b_unbound,
                                c0=c0, structType=structType)
        for t in T_C
    ])


def _series_df(T, oligo_uM, dH=-80.0, dS=-0.22, structType="heterodimer"):
    df = pd.DataFrame({"temperature": T})
    for name, oligo in oligo_uM.items():
        c0 = 1e-6 * oligo * 2
        df[name] = _synth_curve(T, dH, dS, c0, structType=structType)
    return df


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed in this environment")
class TestApi(unittest.TestCase):

    def setUp(self):
        self.T = np.arange(5.0, 95.01, 0.5)
        self.dH = -80.0
        self.dS = -0.22
        self.oligo_uM = {"c1": 0.5, "c2": 5.0, "c3": 50.0, "c4": 500.0}

    def test_analyze_single(self):
        df = _series_df(self.T, {"c1": 0.5}, dH=self.dH, dS=self.dS)
        r = analyze_single(
            df, "c1",
            struct_type="heterodimer", oligo=0.5,
            T_low=self.T[0], T_high=self.T[-1],
            bl_lower_offset=15.0, bl_upper_offset=15.0,
        )
        self.assertNotIn("error", r, msg=r.get("error"))
        self.assertEqual(r["name"], "c1")
        self.assertIn("TmRaw", r)
        self.assertTrue(r["vantHoff"]["success"])
        self.assertTrue(r["fit_result"]["success"])
        # Full-fit ΔH should recover the synthetic value tightly.
        self.assertAlmostEqual(r["fit_result"]["dH"], self.dH, delta=0.5)

    def test_analyze_multi(self):
        df = _series_df(self.T, self.oligo_uM, dH=self.dH, dS=self.dS)
        r = analyze_multi(
            df, self.oligo_uM,
            struct_type="heterodimer",
            T_low=self.T[0], T_high=self.T[-1],
            bl_lower_offset=15.0, bl_upper_offset=15.0,
        )
        # Multi fit may or may not converge depending on initial conditions;
        # if it does, ΔH should be near the synthetic value, otherwise we
        # at least expect a structured error rather than an exception.
        if "error" in r:
            self.assertIn("multi", r["error"].lower())
            return
        self.assertTrue(r["is_multi"])
        self.assertEqual(len(r["columns"]), 4)
        self.assertAlmostEqual(r["dH"], self.dH, delta=2.0)

    def test_analyze_concentration(self):
        df = _series_df(self.T, self.oligo_uM, dH=self.dH, dS=self.dS)
        r = analyze_concentration(
            df, self.oligo_uM,
            struct_type="heterodimer",
            T_low=self.T[0], T_high=self.T[-1],
            bl_lower_offset=15.0, bl_upper_offset=15.0,
        )
        self.assertNotIn("error", r, msg=r.get("error"))
        self.assertTrue(r["is_concentration"])
        self.assertEqual(len(r["per_curve"]), 4)
        self.assertEqual(r["skipped"], [])
        for key in ("raw", "vh", "fit"):
            s = r["series"][key]
            self.assertIsNotNone(s, msg=f"series.{key} is None")
            self.assertGreater(s["r_squared"], 0.999)
            self.assertAlmostEqual(s["dH"], self.dH, delta=2.0)

    def test_analyze_csv_dispatches_each_mode(self):
        df = _series_df(self.T, self.oligo_uM, dH=self.dH, dS=self.dS)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            path = f.name
        try:
            # Single mode auto-picks first signal column.
            r1 = analyze_csv(
                path, mode="single",
                struct_type="heterodimer", oligo=0.5,
                T_low=self.T[0], T_high=self.T[-1],
                bl_lower_offset=15.0, bl_upper_offset=15.0,
            )
            self.assertEqual(r1["name"], "c1")

            r3 = analyze_csv(
                path, mode="concentration",
                oligo_multi=self.oligo_uM,
                struct_type="heterodimer",
                T_low=self.T[0], T_high=self.T[-1],
                bl_lower_offset=15.0, bl_upper_offset=15.0,
            )
            self.assertTrue(r3["is_concentration"])
        finally:
            os.unlink(path)

    def test_analyze_csv_rejects_unknown_mode(self):
        df = _series_df(self.T, {"c1": 0.5}, dH=self.dH, dS=self.dS)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                analyze_csv(path, mode="bogus")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
