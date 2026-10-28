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


class TestTmDsRawOffGridBoundaries(unittest.TestCase):
    """Regression: T_m_ds_raw used to have `if boundary not in T` fallback
    that silently fitted the folded baseline over the *entire scan* when
    the boundary didn't sit on a grid point. Any hand-typed offset value
    triggered it. This locks in the corrected behavior."""

    def _linear_bl_curve(self):
        # A slightly-sloped folded baseline (m=1e-4) plus a sharp transition
        # so on-grid vs off-grid boundaries should give near-identical slopes
        # when both windows fully cover the folded plateau.
        T = np.arange(10.0, 90.01, 0.5)
        y_fold = 0.0 + 1e-4 * T
        y_unf  = 1.0 + 1e-4 * T
        theta = 1.0 / (1.0 + np.exp(-(T - 55.0) / 1.5))
        S = (1.0 - theta) * y_fold + theta * y_unf
        return T, S

    def test_off_grid_boundary_matches_on_grid(self):
        T, S = self._linear_bl_curve()
        # 40.0 is on the 0.5 °C grid; 40.3 and 39.7 are off-grid.
        _, _, on_b,   _, _ = methods.T_m_ds_raw(T, S, baseline_bound_maxT=40.0,
                                                 baseline_unbound_minT=70.0)
        _, _, off_hi, _, _ = methods.T_m_ds_raw(T, S, baseline_bound_maxT=40.3,
                                                 baseline_unbound_minT=70.0)
        _, _, off_lo, _, _ = methods.T_m_ds_raw(T, S, baseline_bound_maxT=39.7,
                                                 baseline_unbound_minT=70.0)
        # Off-grid boundaries must produce near-identical folded baselines,
        # not fall back to a whole-scan fit.
        self.assertAlmostEqual(on_b[0], off_hi[0], places=5)
        self.assertAlmostEqual(on_b[0], off_lo[0], places=5)
        self.assertAlmostEqual(on_b[1], off_hi[1], places=4)
        self.assertAlmostEqual(on_b[1], off_lo[1], places=4)

    def test_off_grid_boundary_stays_near_true_slope(self):
        T, S = self._linear_bl_curve()
        _, _, base_b, _, _ = methods.T_m_ds_raw(T, S, baseline_bound_maxT=40.3,
                                                 baseline_unbound_minT=70.0)
        # True folded slope is 1e-4; whole-scan fit would give ~1e-2.
        self.assertLess(abs(base_b[0] - 1e-4), 5e-5)

    def test_boundary_just_above_scan_max_is_clamped(self):
        """Regression: JS toFixed(1) rounds T[-1]=89.6999969 up to 89.7 —
        the UI then sends baseline_unbound_maxT=89.7 which used to raise
        StopIteration inside fit_linear (empty error string) and kill the
        whole pipeline. Clamp gracefully to the last index instead."""
        T = np.arange(10.0, 90.0, 0.5, dtype=float)   # T[-1] = 89.5
        # Sigmoid so both folded and unfolded windows have real data.
        theta = 1.0 / (1.0 + np.exp(-(T - 60.0) / 2.0))
        S = 1.0 * (1.0 - theta) + 2.0 * theta
        # baseline_unbound_maxT overshoots T[-1] by 0.1 °C — clamp, don't crash.
        Tm, _, base_b, base_ub, _ = methods.T_m_ds_raw(
            T, S,
            baseline_bound_maxT=40.0,
            baseline_unbound_minT=70.0,
            baseline_unbound_maxT=89.6,   # T[-1] is 89.5, overshoot by 0.1
        )
        self.assertTrue(np.isfinite(Tm))
        self.assertTrue(np.isfinite(base_ub[0]) and np.isfinite(base_ub[1]))


if __name__ == "__main__":
    unittest.main()
