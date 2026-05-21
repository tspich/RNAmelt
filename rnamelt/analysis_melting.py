"""
analysis_melting.py — two-state melting curve fit + van't Hoff analysis.

Two-state model
───────────────
The observed signal is a linear combination of folded (F) and unfolded (U)
baselines, weighted by the fraction unfolded α(T):

    A(T) = [mL·T + bL] · (1 - α) + [mU·T + bU] · α

where the fraction unfolded for a two-state equilibrium is:

    K(T)  = exp(-ΔH/RT + ΔS/R)
    α(T)  = K / (1 + K)          [monomolecular / intramolecular]

Adjustable parameters passed in via `params` dict
──────────────────────────────────────────────────
    mL, bL          : lower baseline slope & intercept  (°C, signal units)
    mU, bU          : upper baseline slope & intercept
    T_low, T_high   : transition window boundaries (°C) — rows outside this
                      range are excluded from the fit
    fix_baselines   : bool — if True, baselines are held fixed at the supplied
                      values and only ΔH, Tm are fitted
    CT              : total strand concentration (M), used for 1/Tm vs ln[CT]
                      van't Hoff plot (set to None / 0 to skip)
"""

import numpy as np
from rnamelt.utils import safe_json
from rnamelt import methods, functions, constants

# ─── Main entry point ────────────────────────────────────────────────────────

def run(df, params: dict):# -> dict:
    """
    params keys (all optional — sensible defaults applied):
        T_low         (float, °C)   lower boundary of transition window
        T_high        (float, °C)   upper boundary
        mL, bL        (float)       lower baseline slope & intercept
        mU, bU        (float)       upper baseline slope & intercept
        fix_baselines (bool)        hold baselines fixed during fit
        CT            (float, M)    strand concentration for 1/Tm vs ln[CT]
        column        (str)         which signal column to fit
    """
    from rnamelt.cleaning import get_signal_columns

    T_all      = df["temperature"].values.astype(float)
    sig_cols   = get_signal_columns(df)

    if not sig_cols:
        return {"name": "Melting Analysis", "error": "No signal columns found."}

    col = params.get("column", sig_cols[0])
    if col in ('multi', '__multi__'):
        return _run_multi(df, params, T_all, sig_cols)
    elif col in ('concentration', '__concentration__'):
        return _run_concentration(df, params, T_all, sig_cols)
    else:
        if col not in df.columns:
            col = sig_cols[0]

        signal_all = df[col].values.astype(float)

        # Remove NaNs
        valid = ~np.isnan(signal_all)
        T_all      = T_all[valid]
        signal_all = signal_all[valid]

        #if len(T_all) < 10:
        #    return {"name": "Melting Analysis", "error": "Not enough data points."}

        T_min, T_max = float(T_all.min()), float(T_all.max())

        #print(params)

        T_low  = float(params.get("T_low",  T_min))
        T_high = float(params.get("T_high", T_max))

        pos_start = np.where(T_all == T_low)[0][0]
        pos_end   = np.where(T_all == T_high)[0][0]+1

        salt_concentration = float(params.get("salt", 150))
        oligo_concentration = float(params.get("oligo", 0.5))

        TT = T_all[pos_start:pos_end]
        used_data = signal_all[pos_start:pos_end]

        ##signal_type = params.get("signal_type", "absorbance")
        struct_type = params.get("struct_type", "heterodimer")
        c0 = 1e-6 * oligo_concentration * constants.STRAND_STOICHIOMETRY[struct_type]
        base_b_offset = params.get("bl_lower_offset", 10)
        base_ub_offset = params.get("bl_upper_offset", 10)
        solver = params.get("solver")
        vh_kwargs = methods._vh_kwargs(params.get("vh"))
        fit_init = methods._fit_init_kwargs(params.get("fit_init"))

        r_base_b_maxT=T_low+base_b_offset
        r_base_ub_minT=T_high-base_ub_offset

        T_m_raw, y_r, base_b_r, base_ub_r, base_med_r = methods.T_m_ds_raw(
            TT,
            used_data,
            baseline_bound_maxT=r_base_b_maxT,
            baseline_unbound_minT=r_base_ub_minT,
            #debug=True
        )

        try:
            T_m_vH, dG_37_vH, dH_vH, dS_vH, t1, K, xdata, ydata, fit_vh = methods.vantHoff(
                TT,
                used_data,
                *base_b_r,
                *base_ub_r,
                c0,
                structType = struct_type,
                **vh_kwargs,
            )

            vantHoff = {
                "success": True,
                "dG":      dG_37_vH,
                "dH":      dH_vH,
                "dS":      dS_vH,
                "T_m_vH":  T_m_vH,
                "t1":      t1,
                "K":       K,
                "xdata":   xdata,
                "ydata":   ydata,
                "fit_vh":  fit_vh,
            }

        except Exception as e:
            vantHoff = {
                "success": False,
                "error":   str(e),
            }


        # User-supplied fit_init values override the vH-seeded / fallback
        # auto-seeding below.
        if fit_init["dH_init"] is not None:
            dH_init = fit_init["dH_init"]
        elif vantHoff['success'] and -150 < vantHoff["dH"] < 0:
            dH_init = vantHoff["dH"]
        else:
            dH_init = -80

        if fit_init["dS_init"] is not None:
            dS_init = fit_init["dS_init"]
        elif vantHoff['success'] and -5 < vantHoff["dS"] < 0:
            dS_init = vantHoff["dS"]
        else:
            dS_init = -0.2

        try:
            dG_37_f, dH_f, dS_f, T_m_f, y_f, base_b_f, base_ub_f, base_med_f = methods.fit_full_function(
                TT,
                used_data,
                c0=c0,
                dH_init  = dH_init,
                dS_init  = dS_init,
                lin_init = fit_init["lin_init"],
                b1_init  = fit_init["b1_init"],
                b2_init  = fit_init["b2_init"],
                solver   = solver,
            )

            fit = np.array([ functions.full_function(
                tt, dH_f, dS_f, *base_b_f, *base_ub_f, c0=c0
            ) for tt in TT ])

            # NOTE: May need to check that, not general enough?
            derivative = np.gradient(used_data, TT)/-1.

            fit_result = {
                "success":    True,
                "dG":         dG_37_f,
                "dH":         dH_f,
                "dS":         dS_f,
                "T_m_fit":    T_m_f,
                "base_b_f":   base_b_f,
                "base_ub_f":  base_ub_f,
                "base_med_f": base_med_f,
                "fit":        fit,
                "derivative": derivative,
            }

        except Exception as e:
            fit_result = {
                "success":    False,
                "error":      str(e),
            }


        thermo_properties = {
            "name":             col,
            'oligoC':           oligo_concentration,
            'c0':               c0,
            'saltC':            salt_concentration,
            'TmRaw':            T_m_raw,
            'T_used':           TT,
            'T_all':            T_all,
            'signal':           used_data,
            'signal_all':       signal_all,
            'base_b_r':         base_b_r,
            'base_ub_r':        base_ub_r,
            'vantHoff':         vantHoff,
            'fit_result':       fit_result,
        }


        return safe_json(thermo_properties)


def _run_multi(df, params: dict, T_all, sig_cols):
    """Simultaneous fit across all signal columns sharing dH, dS; independent baselines."""
    oligo_multi = params.get("oligo_multi") or {}
    salt_c      = float(params.get("salt", 150))
    struct_type = params.get("struct_type", "heterodimer")
    stoich      = constants.STRAND_STOICHIOMETRY[struct_type]
    solver      = params.get("solver")
    fit_init    = methods._fit_init_kwargs(params.get("fit_init"))

    T_min, T_max = float(T_all.min()), float(T_all.max())
    T_low  = float(params.get("T_low",  T_min))
    T_high = float(params.get("T_high", T_max))

    pos_start = int(np.where(T_all == T_low)[0][0])
    pos_end   = int(np.where(T_all == T_high)[0][0]) + 1
    TT        = T_all[pos_start:pos_end]

    ds, cs, oligos, col_names, signals_all = [], [], [], [], []
    for c in sig_cols:
        sig = df[c].values.astype(float)
        if np.isnan(sig).any():
            continue
        oligo = float(oligo_multi.get(c, 0.5))
        ds.append(sig[pos_start:pos_end])
        cs.append(1e-6 * oligo * stoich)
        oligos.append(oligo)
        col_names.append(c)
        signals_all.append(sig)

    if len(ds) < 2:
        return {"name": "Melting Analysis", "error": "multi fit needs at least 2 valid columns"}

    try:
        # Multi mode has no per-curve vH seed available, so fall back to
        # -80 / -0.2 when the user hasn't overridden via fit_init.
        # b1_init / b2_init are per-curve in multi mode and intentionally
        # not exposed through the bundle — fit_full_function_multi
        # auto-computes them.
        dH_init = fit_init["dH_init"] if fit_init["dH_init"] is not None else -80
        dS_init = fit_init["dS_init"] if fit_init["dS_init"] is not None else -0.2
        dG_37, dH, dS, T_ms, y_dats, bl_lo, bl_hi, bl_me = methods.fit_full_function_multi(
            TT, ds, cs=cs,
            dH_init    = dH_init,
            dS_init    = dS_init,
            lin_init   = fit_init["lin_init"],
            structType = struct_type,
            solver     = solver,
        )
    except Exception as e:
        return {"name": "Melting Analysis", "error": f"multi fit failed: {e}"}

    columns = []
    for i, name in enumerate(col_names):
        fit = np.array([
            functions.full_function(tt, dH, dS, *bl_lo[i], *bl_hi[i], c0=cs[i])
            for tt in TT
        ])
        columns.append({
            "name":       name,
            "oligoC":     oligos[i],
            "c0":         cs[i],
            "T_m_fit":    T_ms[i],
            "base_b_f":   bl_lo[i],
            "base_ub_f":  bl_hi[i],
            "base_med_f": bl_me[i],
            "fit":        fit,
            "signal":     ds[i],
            "signal_all": signals_all[i],
        })

    return safe_json({
        "name":       "multi",
        "is_multi":   True,
        "saltC":      salt_c,
        "T_used":     TT,
        "T_all":      T_all,
        "dG":         dG_37,
        "dH":         dH,
        "dS":         dS,
        "columns":    columns,
    })


def _run_concentration(df, params: dict, T_all, sig_cols):
    """Concentration-series van't Hoff: extract Tm per column, regress 1/Tm vs ln(C_T/f)."""
    struct_type = params.get("struct_type", "heterodimer")
    if struct_type == "monomer":
        return {
            "name": "Concentration Series",
            "error": "Concentration-series van't Hoff is not applicable to monomers — Tm is concentration-independent for intramolecular folding.",
        }
    self_comp = (struct_type == "homodimer")

    oligo_multi = params.get("oligo_multi") or {}
    salt_c      = float(params.get("salt", 150))
    solver      = params.get("solver")
    vh_kwargs   = methods._vh_kwargs(params.get("vh"))
    fit_init    = methods._fit_init_kwargs(params.get("fit_init"))

    T_min, T_max = float(T_all.min()), float(T_all.max())
    T_low  = float(params.get("T_low",  T_min))
    T_high = float(params.get("T_high", T_max))

    pos_start = int(np.where(T_all == T_low)[0][0])
    pos_end   = int(np.where(T_all == T_high)[0][0]) + 1
    TT        = T_all[pos_start:pos_end]

    base_b_offset  = float(params.get("bl_lower_offset", 10))
    base_ub_offset = float(params.get("bl_upper_offset", 10))
    r_base_b_maxT  = T_low  + base_b_offset
    r_base_ub_minT = T_high - base_ub_offset

    per_curve = []
    skipped   = []
    f_factor  = 1.0 if self_comp else 4.0

    for col in sig_cols:
        sig = df[col].values.astype(float)
        if np.isnan(sig).any():
            skipped.append({"name": col, "reason": "contains NaN"})
            continue
        if col not in oligo_multi:
            skipped.append({"name": col, "reason": "no concentration provided"})
            continue
        try:
            oligo_uM = float(oligo_multi[col])
        except (TypeError, ValueError):
            skipped.append({"name": col, "reason": "non-numeric concentration"})
            continue
        if not np.isfinite(oligo_uM) or oligo_uM <= 0:
            skipped.append({"name": col, "reason": f"non-positive concentration {oligo_uM}"})
            continue

        used_data = sig[pos_start:pos_end]
        c_M = 1e-6 * oligo_uM * constants.STRAND_STOICHIOMETRY[struct_type]

        # ── raw Tm via baseline intersection (mandatory) ──────────────────
        try:
            T_m_raw_C, _, base_b, base_ub, _ = methods.T_m_ds_raw(
                TT, used_data,
                baseline_bound_maxT=r_base_b_maxT,
                baseline_unbound_minT=r_base_ub_minT,
            )
        except Exception as e:
            skipped.append({"name": col, "reason": f"raw Tm extraction failed: {e}"})
            continue
        if T_m_raw_C is None or not np.isfinite(T_m_raw_C) or not (T_low <= T_m_raw_C <= T_high):
            skipped.append({"name": col, "reason": f"raw Tm out of window ({T_m_raw_C})"})
            continue

        # ── van't Hoff Tm (best effort) ───────────────────────────────────
        T_m_vH_C  = None
        dH_vh_seed = None
        dS_vh_seed = None
        try:
            T_m_vH_C, _, dH_vh_seed, dS_vh_seed, *_ = methods.vantHoff(
                TT, used_data, *base_b, *base_ub, c_M,
                structType=struct_type, **vh_kwargs,
            )
        except Exception:
            T_m_vH_C = None

        # ── full curve fit Tm (best effort) ───────────────────────────────
        # User-supplied fit_init values override the vH-seeded / fallback chain.
        if fit_init["dH_init"] is not None:
            dH_init = fit_init["dH_init"]
        elif dH_vh_seed is not None and -150 < dH_vh_seed < 0:
            dH_init = dH_vh_seed
        else:
            dH_init = -80.0

        if fit_init["dS_init"] is not None:
            dS_init = fit_init["dS_init"]
        elif dS_vh_seed is not None and -5 < dS_vh_seed < 0:
            dS_init = dS_vh_seed
        else:
            dS_init = -0.2

        T_m_fit_C = None
        try:
            _, _, _, T_m_fit_C, *_ = methods.fit_full_function(
                TT, used_data, c0=c_M,
                dH_init  = dH_init,
                dS_init  = dS_init,
                lin_init = fit_init["lin_init"],
                solver   = solver,
            )
        except Exception:
            T_m_fit_C = None

        def _to_K(T_C):
            if T_C is None or not np.isfinite(T_C):
                return None
            return float(T_C - constants.T0)

        per_curve.append({
            "name":        col,
            "oligoC":      oligo_uM,
            "c0":          c_M,
            "TmRaw":       float(T_m_raw_C),
            "TmRaw_K":     float(T_m_raw_C - constants.T0),
            "TmKelvin":    float(T_m_raw_C - constants.T0),  # back-compat alias
            "TmvH":        None if T_m_vH_C is None or not np.isfinite(T_m_vH_C) else float(T_m_vH_C),
            "TmvH_K":      _to_K(T_m_vH_C),
            "Tmfit":       None if T_m_fit_C is None or not np.isfinite(T_m_fit_C) else float(T_m_fit_C),
            "Tmfit_K":     _to_K(T_m_fit_C),
            "lnCT":        float(np.log(c_M / f_factor)),
            "T_used":      TT,
            "signal":      used_data,
            "signal_all":  sig,
            "base_b_r":    base_b,
            "base_ub_r":   base_ub,
        })

    if len(per_curve) < 2:
        return safe_json({
            "name": "Concentration Series",
            "error": f"Need at least 2 valid (Tm, C_T) points; got {len(per_curve)}.",
            "per_curve": per_curve,
            "skipped":   skipped,
        })

    def _series_for(key):
        tm_K = []
        ct_M = []
        for p in per_curve:
            v = p[key]
            if v is not None and np.isfinite(v) and v > 0:
                tm_K.append(v)
                ct_M.append(p["c0"])
        if len(tm_K) < 2:
            return None
        try:
            vh = methods.vant_hoff_concentration(
                np.array(tm_K), np.array(ct_M),
                self_complementary=self_comp,
            )
        except ValueError:
            return None
        ln_ct_min = float(vh["ln_ct"].min())
        ln_ct_max = float(vh["ln_ct"].max())
        ln_ct_line  = np.linspace(ln_ct_min, ln_ct_max, 50)
        inv_tm_line = vh["slope"] * ln_ct_line + vh["intercept"]
        return {
            "success":     True,
            "n":           len(tm_K),
            "dH":          vh["dH"],
            "dS":          vh["dS"],
            "dG_37":       vh["dG_37"],
            "r_squared":   vh["r_squared"],
            "slope":       vh["slope"],
            "intercept":   vh["intercept"],
            "ln_ct":       vh["ln_ct"],
            "inv_tm":      vh["inv_tm"],
            "ln_ct_line":  ln_ct_line,
            "inv_tm_line": inv_tm_line,
        }

    series = {
        "raw": _series_for("TmRaw_K"),
        "vh":  _series_for("TmvH_K"),
        "fit": _series_for("Tmfit_K"),
    }

    if series["raw"] is None:
        return safe_json({
            "name": "Concentration Series",
            "error": "Could not regress raw-Tm series — too few valid points.",
            "per_curve": per_curve,
            "skipped":   skipped,
            "series":    series,
        })

    return safe_json({
        "name":               "concentration",
        "is_concentration":   True,
        "struct_type":        struct_type,
        "self_complementary": self_comp,
        "saltC":              salt_c,
        "T_all":              T_all,
        "T_used":             TT,
        "per_curve":          per_curve,
        "skipped":            skipped,
        "series":             series,
        # back-compat: top-level vantHoff = raw series
        "vantHoff":           series["raw"],
    })
