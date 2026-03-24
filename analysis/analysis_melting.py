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
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from analysis.utils import celsius_to_kelvin, R, safe_json
from analysis.util import analyze


# ─── Two-state model ──────────────────────────────────────────────────────────

def fraction_unfolded(T_C, dH, Tm_C):
    """
    α(T) for a two-state intramolecular transition.
    T_C  : temperature in °C
    dH   : enthalpy in J/mol  (positive for unfolding)
    Tm_C : melting temperature in °C
    """
    T  = celsius_to_kelvin(T_C)
    Tm = celsius_to_kelvin(Tm_C)
    K  = np.exp((dH / R) * (1.0 / Tm - 1.0 / T))
    return K / (1.0 + K)


def two_state_signal(T_C, dH, Tm_C, mL, bL, mU, bU):
    """Full two-state model: sloped lower + upper baselines."""
    alpha = fraction_unfolded(T_C, dH, Tm_C)
    lower = mL * T_C + bL
    upper = mU * T_C + bU
    return lower * (1 - alpha) + upper * alpha


def two_state_signal_fixed_baselines(T_C, dH, Tm_C, mL, bL, mU, bU):
    """Same model but called with externally fixed mL, bL, mU, bU."""
    return two_state_signal(T_C, dH, Tm_C, mL, bL, mU, bU)


# ─── Derivative (for Tm estimation) ──────────────────────────────────────────

def numerical_derivative(T, signal):
    """Smoothed first derivative; peak position ≈ Tm."""
    if len(T) < 7:
        return T, np.gradient(signal, T)
    smoothed = savgol_filter(signal, window_length=min(11, len(signal) // 2 * 2 - 1), polyorder=3)
    deriv    = np.gradient(smoothed, T)
    return T, deriv


# ─── Auto-baseline estimation ─────────────────────────────────────────────────

def estimate_baselines(T, signal, T_low, T_high):
    """
    Fit linear baselines to the pre- and post-transition regions.
    Returns (mL, bL, mU, bU) as initial guesses.
    """
    pre  = (T < T_low)
    post = (T > T_high)

    def fit_line(mask):
        if mask.sum() < 2:
            return 0.0, float(signal[mask].mean()) if mask.sum() else float(signal.mean())
        p = np.polyfit(T[mask], signal[mask], 1)
        return float(p[0]), float(p[1])

    mL, bL = fit_line(pre)
    mU, bU = fit_line(post)
    return mL, bL, mU, bU


# ─── Baseline-corrected fraction unfolded ────────────────────────────────────

def correct_baselines(T, signal, mL, bL, mU, bU):
    lower = mL * T + bL
    upper = mU * T + bU
    denom = upper - lower
    # avoid division by near-zero
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.where(np.abs(denom) > 1e-12, (signal - lower) / denom, np.nan)
    return alpha


# ─── Main entry point ────────────────────────────────────────────────────────

def run(df, params: dict) -> dict:
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

    T_low  = float(params.get("T_low",  T_min))
    T_high = float(params.get("T_high", T_max))
    salt_concentration = float(params.get("salt", 150))
    oligo_concentration = float(params.get("oligo", 0.5))

    signal_type = params.get("signal_type", "absorbance")
    struct_type = params.get("struct_type", "heterodimer")
    
    print(T_min, T_max)
    print(T_low, T_high)
    print(salt_concentration)
    print(oligo_concentration)
    print(signal_type)
    print(struct_type)

    if signal_type == "fluorescence":
        if struct_type == "heterodimer":
            thermo_properties = []

            analyze(
                col,
                salt_concentration,
                oligo_concentration,
                T_low,
                T_high,
                signal_all,
                store_data=thermo_properties,
                T=T_all,
            )

            print(type(thermo_properties))
            for t in thermo_properties[0]:
                print(t,type(thermo_properties[0][t]))

            return safe_json(thermo_properties[0])

        elif struct_type == "monomer":
            return {"name": "Melting Analysis", "error": "Not implemented yet."}
        elif struct_type == "homodimer":
            return {"name": "Melting Analysis", "error": "Not implemented yet."}

    elif signal_type == "absorbance":
        return {"name": "Melting Analysis", "error": "Not implemented yet."}

    else:
        return {"name": "Melting Analysis", "error": "Signal type is missing"}


    ## Auto-estimate baselines if not supplied
    #mL_est, bL_est, mU_est, bU_est = estimate_baselines(T_all, signal_all, T_low, T_high)
    #mL = float(params.get("mL", mL_est))
    #bL = float(params.get("bL", bL_est))
    #mU = float(params.get("mU", mU_est))
    #bU = float(params.get("bU", bU_est))

    #fix_baselines = bool(params.get("fix_baselines", False))

    ## Transition window mask for fitting
    #mask = (T_all >= T_low) & (T_all <= T_high)
    #T_fit  = T_all[mask]
    #S_fit  = signal_all[mask]

    #if len(T_fit) < 5:
    #    return {"name": "Melting Analysis", "error": "Transition window too narrow — too few points for fitting."}

    ## ── Fit ──────────────────────────────────────────────────────────────────
    ## Initial guess: Tm from derivative peak, ΔH ~200 kJ/mol
    #_, deriv_all = numerical_derivative(T_all, signal_all)
    #Tm_guess = float(T_all[np.argmax(deriv_all)])
    #dH_guess = 200_000.0   # J/mol

    #fit_result = {}
    #try:
    #    if fix_baselines:
    #        def model(T_C, dH, Tm_C):
    #            return two_state_signal(T_C, dH, Tm_C, mL, bL, mU, bU)
    #        p0     = [dH_guess, Tm_guess]
    #        bounds = ([0, T_min], [2_000_000, T_max])
    #        popt, pcov = curve_fit(model, T_fit, S_fit, p0=p0, bounds=bounds, maxfev=20000)
    #        dH_fit, Tm_fit = popt
    #        mL_fit, bL_fit, mU_fit, bU_fit = mL, bL, mU, bU
    #        perr = np.sqrt(np.diag(pcov))
    #        dH_err, Tm_err = perr[0], perr[1]
    #    else:
    #        def model(T_C, dH, Tm_C, mL_, bL_, mU_, bU_):
    #            return two_state_signal(T_C, dH, Tm_C, mL_, bL_, mU_, bU_)
    #        p0     = [dH_guess, Tm_guess, mL, bL, mU, bU]
    #        bounds = ([0,       T_min,   -np.inf, -np.inf, -np.inf, -np.inf],
    #                  [2e6,     T_max,    np.inf,  np.inf,  np.inf,  np.inf])
    #        popt, pcov = curve_fit(model, T_fit, S_fit, p0=p0, bounds=bounds, maxfev=20000)
    #        dH_fit, Tm_fit, mL_fit, bL_fit, mU_fit, bU_fit = popt
    #        perr = np.sqrt(np.diag(pcov))
    #        dH_err, Tm_err = perr[0], perr[1]

    #    dS_fit = dH_fit / celsius_to_kelvin(Tm_fit)          # J/(mol·K)
    #    dG_fit = dH_fit - 298.15 * dS_fit                    # J/mol at 25°C

    #    # Fitted curve over full T range
    #    T_curve  = np.linspace(T_min, T_max, 300)
    #    S_curve  = two_state_signal(T_curve, dH_fit, Tm_fit, mL_fit, bL_fit, mU_fit, bU_fit)

    #    # Baseline lines
    #    lower_bl = mL_fit * T_all + bL_fit
    #    upper_bl = mU_fit * T_all + bU_fit

    #    # Baseline-corrected fraction unfolded
    #    alpha_obs = correct_baselines(T_all, signal_all, mL_fit, bL_fit, mU_fit, bU_fit)
    #    alpha_fit = fraction_unfolded(T_curve, dH_fit, Tm_fit)

    #    # Residuals
    #    S_fit_vals = two_state_signal(T_fit, dH_fit, Tm_fit, mL_fit, bL_fit, mU_fit, bU_fit)
    #    residuals  = (S_fit - S_fit_vals).tolist()
    #    rmse       = float(np.sqrt(np.mean(np.array(residuals)**2)))

    #    fit_result = safe_json({
    #        "success":   True,
    #        "dH":        dH_fit,           # J/mol
    #        "dH_err":    dH_err,
    #        "Tm":        Tm_fit,           # °C
    #        "Tm_err":    Tm_err,
    #        "dS":        dS_fit,           # J/(mol·K)
    #        "dG_25":     dG_fit,           # J/mol
    #        "mL":        mL_fit,
    #        "bL":        bL_fit,
    #        "mU":        mU_fit,
    #        "bU":        bU_fit,
    #        "rmse":      rmse,
    #        "T_curve":   T_curve.tolist(),
    #        "S_curve":   S_curve.tolist(),
    #        "lower_bl":  lower_bl.tolist(),
    #        "upper_bl":  upper_bl.tolist(),
    #        "alpha_obs_T": T_all.tolist(),
    #        "alpha_obs":   alpha_obs.tolist(),
    #        "alpha_fit_T": T_curve.tolist(),
    #        "alpha_fit":   alpha_fit.tolist(),
    #        "residual_T":  T_fit.tolist(),
    #        "residuals":   residuals,
    #    })

    #except Exception as e:
    #    fit_result = {"success": False, "error": str(e)}

    ## ── Derivative ───────────────────────────────────────────────────────────
    #_, deriv = numerical_derivative(T_all, signal_all)
    #deriv_result = safe_json({
    #    "T":     T_all.tolist(),
    #    "deriv": deriv.tolist(),
    #    "Tm_deriv": float(T_all[np.argmax(deriv)]),
    #})

    ## ── Van't Hoff: 1/Tm vs ln[CT] ──────────────────────────────────────────
    ## Stored for multi-concentration assembly in the browser;
    ## a single run just contributes one point.
    #CT = params.get("CT", None)
    #vantHoff_point = None
    #if CT and float(CT) > 0 and fit_result.get("success"):
    #    Tm_K = celsius_to_kelvin(fit_result["Tm"])
    #    vantHoff_point = safe_json({
    #        "CT":     float(CT),
    #        "inv_Tm": float(1.0 / Tm_K),
    #        "ln_CT":  float(np.log(float(CT))),
    #    })

    #return safe_json({
    #    "name":           "Melting Analysis",
    #    "column":         col,
    #    "signal_type":    signal_type,
    #    "T_raw":          T_all.tolist(),
    #    "S_raw":          signal_all.tolist(),
    #    "T_low":          T_low,
    #    "T_high":         T_high,
    #    "fit":            fit_result,
    #    "derivative":     deriv_result,
    #    "vantHoff_point": vantHoff_point,
    #    "sig_cols":       sig_cols,
    #    "params_used": safe_json({
    #        "mL": mL, "bL": bL, "mU": mU, "bU": bU,
    #        "T_low": T_low, "T_high": T_high,
    #        "fix_baselines": fix_baselines,
    #    }),
    #})
