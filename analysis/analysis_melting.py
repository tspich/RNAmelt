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

    signal_type = params.get("signal_type", "absorbance")
    struct_type = params.get("struct_type", "heterodimer")
    base_b_offset = params.get("bl_lower_offset")
    base_ub_offset = params.get("bl_upper_offset")


    
    if signal_type == "fluorescence":
        if struct_type == "heterodimer":
            r_base_b_maxT=T_all[0]+base_b_offset
            r_base_ub_minT=T_all[-1]-base_ub_offset

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
                    border = 0.15,
                    #t1_min = t1_min,
                    #t1_max = t1_max
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
                    "error":   e
                }

            if vantHoff['success']:
                if 0 < vantHoff["dH"] > -150:
                    dH_init = vantHoff["dH"]
                else:
                    dH_init = -80
                if 0 < vantHoff["dS"] > -5:
                    dS_init = vantHoff["dS"]
                else:
                    dS_init = -0.2

                try:
                    dG_37_f, dH_f, dS_f, T_m_f, y_f, base_b_f, base_ub_f, base_med_f = methods.fit_full_function(
                        TT,
                        used_data,
                        c0=c0,
                        dH_init = dH_init,
                        dS_init = dS_init,
                    )

                    fit = np.array([ functions.full_function(tt, dH_f, dS_f, *base_b_f, *base_ub_f, c0=c0) for tt in TT ])
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

                except Exception as e:
                    fit_result = {
                        "success":    False,
                        "error":      e,
                    }
                    

            thermo_properties = {
                "name":             col,
                'oligoC':           oligo_concentration,
                'saltC':            salt_concentration,
                'TmRaw':            T_m_raw,
                'vantHoff':         vantHoff,
                'fit_result':       fit_result,
            }


            print(type(thermo_properties))
            print(thermo_properties['base_b_offset'])

            return safe_json(thermo_properties)

        elif struct_type == "monomer":
            return {"name": "Melting Analysis", "error": "Not implemented yet."}
        elif struct_type == "homodimer":
            return {"name": "Melting Analysis", "error": "Not implemented yet."}

    elif signal_type == "absorbance":
        return {"name": "Melting Analysis", "error": "Not implemented yet."}

    else:
        return {"name": "Melting Analysis", "error": "Signal type is missing"}


