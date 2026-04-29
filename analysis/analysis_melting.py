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
#from scipy.optimize import curve_fit
#from scipy.signal import savgol_filter
from analysis.utils import safe_json
#from analysis.util import analyze
from analysis import methods, functions

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
    from analysis.cleaning import get_signal_columns

    T_all      = df["temperature"].values.astype(float)
    sig_cols   = get_signal_columns(df)

    if not sig_cols:
        return {"name": "Melting Analysis", "error": "No signal columns found."}

    col = params.get("column", sig_cols[0])
    if col in ('multi', '__multi__'):
        return _run_multi(df, params, T_all, sig_cols)
    else:
        if col not in df.columns:
            col = sig_cols[0]

        signal_all = df[col].values.astype(float)

        # Remove NaNs
        valid = ~np.isnan(signal_all)
        T_all      = T_all[valid]
        signal_all = signal_all[valid]

        if len(T_all) < 10:
            return {"name": "Melting Analysis", "error": "Not enough data points."}

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

        c0 = 1e-6 * oligo_concentration*2

        #signal_type = params.get("signal_type", "absorbance")
        struct_type = params.get("struct_type", "heterodimer")
        base_b_offset = params.get("bl_lower_offset")
        base_ub_offset = params.get("bl_upper_offset")


        #if signal_type == "absorbance":
            # May implement automatic calculation of the oligo conc.
            # through the extinction coefficient at some point.

        # if signal_type == "fluorescence":
        r_base_b_maxT=T_low+base_b_offset
        r_base_ub_minT=T_high-base_ub_offset

        T_m_raw, y_r, base_b_r, base_ub_r, base_med_r = methods.T_m_ds_raw(
            TT,
            used_data,
            baseline_bound_maxT=r_base_b_maxT,
            baseline_unbound_minT=r_base_ub_minT,
            #debug=True
        )
        print(T_m_raw, base_b_r, base_ub_r, base_med_r)

        #try:
        print(len(TT), len(used_data), len(T_all), len(signal_all))
        print('c0', c0)
        print('struct_type', struct_type)

        T_m_vH, dG_37_vH, dH_vH, dS_vH, t1, K, xdata, ydata, fit_vh = methods.vantHoff(
            TT,
            used_data,
            *base_b_r,
            *base_ub_r,
            c0,
            border = 0.15,
            #t1_min = t1_min,
            #t1_max = t1_max
            structType = struct_type,
        )

        print('vantHoff', T_m_vH, dG_37_vH, dH_vH, dS_vH)

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

        #except Exception as e:
        #    vantHoff = {
        #        "success": False,
        #        "error":   str(e),
        #    }
        #    print('vantHoff', vantHoff)


        if vantHoff['success']:
            print(vantHoff["dH"])
            if -150 < vantHoff["dH"] < 0:
                dH_init = vantHoff["dH"]
            else:
                dH_init = -80
            if -5 < vantHoff["dS"] < 0:
                dS_init = vantHoff["dS"]
            else:
                dS_init = -0.2
        else:
            dH_init = -80
            dS_init = -0.2

        #try:
        #print(len(TT), TT[0], TT[-1])
        #print(len(used_data), min(used_data), max(used_data))
        #print('c0', c0)
        #print('dH_init', dH_init)
        #print('dS_init', dS_init)

        dG_37_f, dH_f, dS_f, T_m_f, y_f, base_b_f, base_ub_f, base_med_f = methods.fit_full_function(
            TT,
            used_data,
            c0=c0,
            dH_init = dH_init,
            dS_init = dS_init,
        )

        #print(dG_37_f, dH_f, dS_f, T_m_f)

        fit = np.array([ functions.full_function(
            tt, dH_f, dS_f, *base_b_f, *base_ub_f, c0=c0
        ) for tt in TT ])

        # NOTE: May need to check that, not general enough?
        derivative = np.gradient(used_data, 0.5)/-1.

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

        #except Exception as e:
        #    fit_result = {
        #        "success":    False,
        #        "error":      str(e),
        #    }


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
        cs.append(1e-6 * oligo * 2)
        oligos.append(oligo)
        col_names.append(c)
        signals_all.append(sig)

    if len(ds) < 2:
        return {"name": "Melting Analysis", "error": "multi fit needs at least 2 valid columns"}

    try:
        dG_37, dH, dS, T_ms, y_dats, bl_lo, bl_hi, bl_me = methods.fit_full_function_multi(
            TT, ds, cs=cs, dH_init=-80, dS_init=-0.2,
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
