"""Tests for analysis.methods — pytest-compatible, runnable via unittest too."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnamelt import methods, constants


def _synthesize_tm(dH: float, dS: float, concentrations: np.ndarray, f: float) -> np.ndarray:
    """Inverse of the van't Hoff regression: given (ΔH, ΔS), return Tm[K] for each C_T."""
    inv_tm = (constants.R / dH) * np.log(concentrations / f) + (dS / dH)
    return 1.0 / inv_tm


class TestVantHoffConcentration(unittest.TestCase):

    def test_recovers_known_thermo_non_self_complementary(self):
        dH_true, dS_true = -80.0, -0.22
        cs = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4])
        tm = _synthesize_tm(dH_true, dS_true, cs, f=4.0)

        res = methods.vant_hoff_concentration(tm, cs, self_complementary=False)

        self.assertAlmostEqual(res["dH"], dH_true, places=6)
        self.assertAlmostEqual(res["dS"], dS_true, places=6)
        self.assertAlmostEqual(
            res["dG_37"], dH_true - (37.0 - constants.T0) * dS_true, places=6
        )
        self.assertGreater(res["r_squared"], 1 - 1e-12)

    def test_self_complementary_uses_f_equals_one(self):
        dH_true, dS_true = -65.0, -0.18
        cs = np.array([1e-7, 1e-6, 1e-5, 1e-4])
        tm_self = _synthesize_tm(dH_true, dS_true, cs, f=1.0)

        res = methods.vant_hoff_concentration(tm_self, cs, self_complementary=True)
        self.assertAlmostEqual(res["dH"], dH_true, places=6)
        self.assertAlmostEqual(res["dS"], dS_true, places=6)

        # Re-fitting the same Tm with f=4 instead of f=1 leaves ΔH unchanged
        # (slope of inv_tm vs ln(C_T) is invariant to a constant shift in x)
        # but biases ΔS by +R·ln(4). This pins down the f-factor handling.
        res_wrong = methods.vant_hoff_concentration(
            tm_self, cs, self_complementary=False
        )
        self.assertAlmostEqual(res_wrong["dH"], dH_true, places=6)
        self.assertAlmostEqual(
            res_wrong["dS"] - dS_true, constants.R * np.log(4.0), places=8
        )

    def test_returns_intermediate_arrays(self):
        cs = np.array([1e-6, 1e-5, 1e-4])
        tm = _synthesize_tm(-70.0, -0.2, cs, f=4.0)
        res = methods.vant_hoff_concentration(tm, cs)

        np.testing.assert_allclose(res["inv_tm"], 1.0 / tm)
        np.testing.assert_allclose(res["ln_ct"], np.log(cs / 4.0))
        self.assertFalse(res["self_complementary"])

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            methods.vant_hoff_concentration(
                np.array([300.0, 310.0]), np.array([1e-6, 1e-5, 1e-4])
            )

    def test_rejects_too_few_points(self):
        with self.assertRaises(ValueError):
            methods.vant_hoff_concentration(np.array([310.0]), np.array([1e-6]))

    def test_rejects_non_positive_tm(self):
        with self.assertRaises(ValueError):
            methods.vant_hoff_concentration(
                np.array([310.0, 0.0]), np.array([1e-6, 1e-5])
            )

    def test_rejects_non_positive_concentration(self):
        with self.assertRaises(ValueError):
            methods.vant_hoff_concentration(
                np.array([310.0, 320.0]), np.array([1e-6, 0.0])
            )

    def test_rejects_constant_concentration(self):
        with self.assertRaises(ValueError):
            methods.vant_hoff_concentration(
                np.array([310.0, 320.0, 330.0]),
                np.array([1e-6, 1e-6, 1e-6]),
            )

    def test_accepts_python_lists(self):
        cs = [1e-6, 1e-5, 1e-4]
        tm_arr = _synthesize_tm(-75.0, -0.21, np.array(cs), f=4.0)
        res = methods.vant_hoff_concentration(list(tm_arr), cs)
        self.assertAlmostEqual(res["dH"], -75.0, places=6)


if __name__ == "__main__":
    unittest.main()
