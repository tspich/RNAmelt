"""Tests for the MeltAnalysis class and rnamelt.results dataclasses."""
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
    FitFailed,
    FullFit,
    MeltAnalysis,
    SingleResult,
    VanHoffFit,
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
class TestMeltAnalysisConstruction(unittest.TestCase):
    """Construction, configure, copy, signal_columns — no fitting."""

    def setUp(self):
        self.T = np.arange(5.0, 95.01, 0.5)
        self.df = _series_df(self.T, {"c1": 0.5, "c2": 5.0})

    def test_defaults(self):
        m = MeltAnalysis(self.df)
        self.assertEqual(m.struct_type, "heterodimer")
        self.assertEqual(m.salt, 150.0)
        self.assertIsNone(m.T_low)
        self.assertIsNone(m.T_high)
        self.assertEqual(m.bl_lower_offset, 10.0)
        self.assertEqual(m.bl_upper_offset, 10.0)
        self.assertIsNone(m.solver)
        self.assertIsNone(m.vh)
        self.assertIsNone(m.fit_init)
        self.assertIsNone(m.last_result)

    def test_constructor_copies_dict_args(self):
        """solver/vh/fit_init dicts must be defensively copied at __init__."""
        sv = {"max_nfev": 99}
        m = MeltAnalysis(self.df, solver=sv)
        sv["max_nfev"] = 1
        self.assertEqual(m.solver["max_nfev"], 99)

    def test_signal_columns(self):
        m = MeltAnalysis(self.df)
        self.assertEqual(m.signal_columns, ["c1", "c2"])

    def test_configure_chains_and_mutates(self):
        m = MeltAnalysis(self.df)
        ret = m.configure(struct_type="homodimer", T_low=10.0, T_high=80.0)
        self.assertIs(ret, m)
        self.assertEqual(m.struct_type, "homodimer")
        self.assertEqual(m.T_low, 10.0)
        self.assertEqual(m.T_high, 80.0)

    def test_configure_rejects_unknown_key(self):
        m = MeltAnalysis(self.df)
        with self.assertRaises(ValueError) as ctx:
            m.configure(bogus=1)
        self.assertIn("bogus", str(ctx.exception))

    def test_configure_replaces_dict_knobs(self):
        """Dict overrides replace — not merge — by design."""
        m = MeltAnalysis(self.df, solver={"max_nfev": 100, "ftol": 1e-9})
        m.configure(solver={"max_nfev": 999})
        self.assertEqual(m.solver, {"max_nfev": 999})

    def test_copy_isolates_config_dicts(self):
        m1 = MeltAnalysis(self.df, solver={"max_nfev": 100})
        m2 = m1.copy()
        m2.struct_type = "homodimer"
        m2.solver["max_nfev"] = 999
        self.assertEqual(m1.struct_type, "heterodimer")
        self.assertEqual(m1.solver["max_nfev"], 100)
        self.assertIs(m1.df, m2.df)  # DataFrame is shared, not deep-copied

    def test_params_dict_omits_none(self):
        """_params() must skip T_low/T_high/solver/vh/fit_init when unset."""
        m = MeltAnalysis(self.df)
        p = m._params()
        self.assertNotIn("T_low", p)
        self.assertNotIn("T_high", p)
        self.assertNotIn("solver", p)
        self.assertNotIn("vh", p)
        self.assertNotIn("fit_init", p)
        self.assertEqual(p["struct_type"], "heterodimer")
        self.assertEqual(p["salt"], 150.0)


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed in this environment")
class TestMeltAnalysisSingle(unittest.TestCase):
    """single() — typed-view access, last_result, to_dict parity."""

    def setUp(self):
        self.T = np.arange(5.0, 95.01, 0.5)
        self.dH = -80.0
        self.dS = -0.22
        self.df = _series_df(self.T, {"c1": 0.5}, dH=self.dH, dS=self.dS)
        self.m = MeltAnalysis(
            self.df,
            T_low=self.T[0], T_high=self.T[-1],
            bl_lower_offset=15.0, bl_upper_offset=15.0,
        )

    def test_returns_typed_result(self):
        r = self.m.single("c1", oligo=0.5)
        self.assertIsInstance(r, SingleResult)
        self.assertEqual(r.column, "c1")

    def test_full_fit_recovers_synthetic_dH(self):
        r = self.m.single("c1", oligo=0.5)
        self.assertTrue(r.fit.ok)
        self.assertAlmostEqual(r.fit.dH, self.dH, delta=0.5)

    def test_vh_fit_recovers_synthetic_dH(self):
        r = self.m.single("c1", oligo=0.5)
        self.assertTrue(r.vh.ok)
        self.assertAlmostEqual(r.vh.dH, self.dH, delta=2.0)

    def test_last_result_cache(self):
        r = self.m.single("c1", oligo=0.5)
        self.assertIs(self.m.last_result, r)

    def test_to_dict_is_underlying_raw(self):
        """to_dict() returns the same dict the orchestrator emitted."""
        r = self.m.single("c1", oligo=0.5)
        d = r.to_dict()
        self.assertIs(d, r._raw)
        # legacy keys still present
        for key in ("name", "TmRaw", "vantHoff", "fit_result"):
            self.assertIn(key, d)

    def test_typed_view_is_numpy_raw_is_list(self):
        """Typed fields are restored to ndarray; _raw keeps lists from safe_json."""
        r = self.m.single("c1", oligo=0.5)
        self.assertIsInstance(r.T_used, np.ndarray)
        self.assertIsInstance(r.fit.curve, np.ndarray)
        self.assertIsInstance(r._raw["T_used"], list)
        self.assertIsInstance(r._raw["fit_result"]["fit"], list)

    def test_to_dataframe_shape(self):
        r = self.m.single("c1", oligo=0.5)
        df = r.to_dataframe()
        self.assertEqual(df.shape, (1, 12))
        self.assertEqual(df.loc[0, "column"], "c1")


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed in this environment")
class TestMeltAnalysisMulti(unittest.TestCase):
    """multi() — error-tolerant like test_api.py."""

    def setUp(self):
        self.T = np.arange(5.0, 95.01, 0.5)
        self.dH = -80.0
        self.dS = -0.22
        self.oligo_uM = {"c1": 0.5, "c2": 5.0, "c3": 50.0, "c4": 500.0}
        self.df = _series_df(self.T, self.oligo_uM, dH=self.dH, dS=self.dS)
        self.m = MeltAnalysis(
            self.df,
            T_low=self.T[0], T_high=self.T[-1],
            bl_lower_offset=15.0, bl_upper_offset=15.0,
        )

    def test_multi_either_converges_or_reports_error(self):
        r = self.m.multi(self.oligo_uM)
        if r.error:
            self.assertIn("multi", r.error.lower())
            return
        self.assertEqual(len(r.columns), 4)
        self.assertAlmostEqual(r.dH, self.dH, delta=2.0)
        self.assertEqual(r.columns[0].name, "c1")

    def test_to_dataframe_shape(self):
        r = self.m.multi(self.oligo_uM)
        if r.error:
            self.skipTest("multi fit did not converge")
        df = r.to_dataframe()
        self.assertEqual(df.shape, (4, 7))


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed in this environment")
class TestMeltAnalysisConcentration(unittest.TestCase):

    def setUp(self):
        self.T = np.arange(5.0, 95.01, 0.5)
        self.dH = -80.0
        self.dS = -0.22
        self.oligo_uM = {"c1": 0.5, "c2": 5.0, "c3": 50.0, "c4": 500.0}
        self.df = _series_df(self.T, self.oligo_uM, dH=self.dH, dS=self.dS)
        self.m = MeltAnalysis(
            self.df,
            T_low=self.T[0], T_high=self.T[-1],
            bl_lower_offset=15.0, bl_upper_offset=15.0,
        )

    def test_per_curve_count(self):
        r = self.m.concentration(self.oligo_uM)
        self.assertIsNone(r.error)
        self.assertEqual(len(r.per_curve), 4)
        self.assertEqual(r.skipped, [])

    def test_three_series_recover_dH(self):
        r = self.m.concentration(self.oligo_uM)
        for s in (r.series_raw, r.series_vh, r.series_fit):
            self.assertTrue(s.ok, msg=f"series.{s.method} not ok: {s.error}")
            self.assertGreater(s.r_squared, 0.999)
            self.assertAlmostEqual(s.dH, self.dH, delta=2.0)

    def test_series_indexed_view(self):
        r = self.m.concentration(self.oligo_uM)
        self.assertIs(r.series["raw"], r.series_raw)
        self.assertIs(r.series["vh"],  r.series_vh)
        self.assertIs(r.series["fit"], r.series_fit)

    def test_dataframe_shapes(self):
        r = self.m.concentration(self.oligo_uM)
        self.assertEqual(r.to_dataframe().shape, (4, 7))
        self.assertEqual(r.series_dataframe().shape, (3, 9))


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed in this environment")
class TestMeltAnalysisBatch(unittest.TestCase):

    def setUp(self):
        self.T = np.arange(5.0, 95.01, 0.5)
        self.dH = -80.0
        self.dS = -0.22
        self.oligo_uM = {"c1": 0.5, "c2": 5.0}
        self.df = _series_df(self.T, self.oligo_uM, dH=self.dH, dS=self.dS)
        self.m = MeltAnalysis(
            self.df,
            T_low=self.T[0], T_high=self.T[-1],
            bl_lower_offset=15.0, bl_upper_offset=15.0,
        )

    def test_single_all_runs_every_column(self):
        results = self.m.single_all(oligo=0.5)
        self.assertEqual(set(results.keys()), {"c1", "c2"})
        for col, r in results.items():
            self.assertIsInstance(r, SingleResult)
            self.assertEqual(r.column, col)
            self.assertTrue(r.fit.ok)


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed in this environment")
class TestMeltAnalysisFromCsv(unittest.TestCase):

    def test_from_csv_round_trip(self):
        T = np.arange(5.0, 95.01, 0.5)
        df = _series_df(T, {"c1": 0.5}, dH=-80.0, dS=-0.22)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            path = f.name
        try:
            m = MeltAnalysis.from_csv(
                path,
                T_low=T[0], T_high=T[-1],
                bl_lower_offset=15.0, bl_upper_offset=15.0,
            )
            self.assertEqual(m.signal_columns, ["c1"])
            r = m.single("c1", oligo=0.5)
            self.assertTrue(r.fit.ok)
            self.assertAlmostEqual(r.fit.dH, -80.0, delta=0.5)
        finally:
            os.unlink(path)


class TestSubFitFailureSemantics(unittest.TestCase):
    """No DataFrame needed — exercises the dataclass behavior directly."""

    def test_failed_subfit_attribute_access_returns_none(self):
        bad = VanHoffFit(ok=False, error="did not converge")
        self.assertIsNone(bad.dH)
        self.assertIsNone(bad.Tm)
        self.assertFalse(bad.ok)
        self.assertEqual(bad.error, "did not converge")

    def test_require_raises_FitFailed(self):
        bad = FullFit(ok=False, error="bounds violated")
        with self.assertRaises(FitFailed) as ctx:
            bad.require()
        self.assertIn("bounds violated", str(ctx.exception))

    def test_require_passes_through_when_ok(self):
        good = VanHoffFit(ok=True, dH=-80.0, dS=-0.22, dG=-13.2, Tm=46.6)
        self.assertIs(good.require(), good)
        self.assertEqual(good.dH, -80.0)


class TestVanHoffFitFromDict(unittest.TestCase):

    def test_missing_dict(self):
        r = VanHoffFit.from_dict(None)
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "missing")

    def test_failed_dict(self):
        r = VanHoffFit.from_dict({"success": False, "error": "no convergence"})
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "no convergence")

    def test_arrays_restored_as_ndarray_with_nan(self):
        """safe_json turns NaN into None inside lists; from_dict must restore."""
        r = VanHoffFit.from_dict({
            "success": True,
            "dH": -80.0, "dS": -0.22, "dG": -13.2, "T_m_vH": 46.6,
            "t1":    [1.0, None, 3.0],
            "K":     [0.5, 1.0, 2.0],
            "xdata": [10.0, 20.0],
            "ydata": [None, 1.0],
            "fit_vh": [0.1, 0.2, 0.3],
        })
        self.assertTrue(r.ok)
        self.assertIsInstance(r.t1, np.ndarray)
        self.assertTrue(np.isnan(r.t1[1]))
        self.assertEqual(r.K.tolist(), [0.5, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
