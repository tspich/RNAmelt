import matplotlib.pyplot as plt
import numpy as np
from rnamelt import functions, constants

palette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#999999", "#CC79A7"]
col_raw = palette[1]
col_fit = palette[5]
col_median = palette[0]

dH_string = r'$\Delta H = {:4.2f}~\frac{{kcal}}{{mol}}$'
dS_string = r'$\Delta S = {:4.2f}~\frac{{kcal}}{{mol \cdot K}}$'
dG_string = r'$\Delta G_{{37}} = {:4.2f}~\frac{{kcal}}{{mol}}$'

def plot_raw_data(axs,
                  T,
                  ydata,
                  label_raw = "Raw Data",
                  derivative = False,
                  base_bound = False,
                  base_bound_Tmax = -1,
                  base_unbound = False,
                  base_unbound_Tmin = -1,
                  base_med = False,
                  base_med_Tmin = -1,
                  base_med_Tmax = -1,
                  melting_point = False):
    T     = np.array(T)
    ydata = np.array(ydata)

    # plot actual raw data F
    axs.plot(T, ydata, lw = 4, color=col_raw, alpha = 0.66, label = label_raw)

    if base_bound:
        if base_bound_Tmax in T:
            T_stop = next(i for i,v in enumerate(T) if v > base_bound_Tmax)
        else:
            T_stop = int(len(T) / 4)

        axs.plot(
            T[:T_stop],
            functions.linear(T[:T_stop], *base_bound),
            "--",
            linewidth=0.66,
            color="black",
            label=None,
        )
        axs.text(
            T[T_stop],
            functions.linear(T[T_stop], *base_bound),
            "$L_2$",
            verticalalignment="center",
            horizontalalignment="left",
        )

    if base_unbound:
        if base_unbound_Tmin in T[:-10]:
            T_start = next(i for i, v in enumerate(T) if v > base_unbound_Tmin)
        else:
            T_start = int(3 * len(T) / 4)

        axs.plot(
            T[T_start:],
            functions.linear(T[T_start:], *base_unbound),
            "--",
            linewidth=0.66,
            color="black",
            label=None,
        )
        axs.text(
            T[T_start],
            functions.linear(T[T_start], *base_unbound),
            "$L_1$",
            verticalalignment="center",
            horizontalalignment="right",
        )

    if base_med:
        if base_med_Tmin in T:
            T_start = next(i for i, v in enumerate(T) if v > base_med_Tmin)
        elif melting_point:
            T_start = next(i for i, v in enumerate(T) if v > melting_point[0]-10)
        else:
            T_start = int(len(T) / 4)

        if base_med_Tmax in T:
            T_stop = next(i for i, v in enumerate(T) if v > base_med_Tmax)
        elif melting_point:
            T_stop = next(i for i, v in enumerate(T) if v > melting_point[0]+10)
        else:
            T_stop = int(3 * len(T) / 4)

        axs.plot(
            T[T_start:T_stop],
            functions.linear(T[T_start:T_stop], *base_med),
            "--",
            linewidth=0.66,
            color="black",
            label=None,
        )
        axs.text(
            T[T_stop],
            functions.linear(T[T_stop], *base_med),
            r"$\frac{1}{2} (L_1 + L_2)$",
            verticalalignment="center",
            horizontalalignment="left",
        )

    if melting_point:
        axs.vlines(
            melting_point[0],
            min(ydata),
            melting_point[1],
            linestyles="dotted",
            color="red",
            label=r"$T_m = %5.2f ^\degree C$" % melting_point[0],
        )

    axs.set_xlabel(r"Temperature [$^\degree C$]")
    axs.set_ylabel("Relative Flourescence")

    if derivative:
        ax2 = axs.twinx() # second axis that shares x-axis

        # plot derivative d(F)/d(T) of raw data
        d = np.gradient(ydata, 0.5) / -1.
        ax2.plot(T, d, '--', lw = 2, color=col_raw, label = None)
        ax2.set_ylabel("-d(F) / d(T)")


def plot_vantHoff(axs,
                  t1,
                  K,
                  c0,
                  xdata,
                  ydata,
                  fit,
                  T_m,
                  T_scale = 1000.,
                  K_min = 8,
                  K_max = 23
                  ):
    t_m = T_scale / (T_m - constants.T0)

    mask = ~np.isnan(ydata)
    xmin, xmax = xdata[mask].min(), xdata[mask].max()

    axs.plot(
        t1, K, "-", lw=4,
         color=col_raw, alpha=0.66, label="Raw Data (normalized)"
    )
    axs.plot(
        xdata,
        functions.linear(xdata, *fit),
        "--",
        lw=2,
        color=col_fit,
        label="Linear Fit",
    )

    if len(xdata) > 0 or len(xdata) > 0:
        axs.hlines(np.log(4 / c0), xmin, xmax,
                   linestyles="dotted", color="black", label = None)
        axs.vlines(t_m, min(ydata), np.log(4 / c0), linestyles="dotted",
                   color = "red", label = r"$T_m = %5.2f ^\degree C$" % T_m)
    axs.text(xmin-0.01, np.log(4 / c0), r'$K_a = \frac{4}{C_o}$',
             verticalalignment='center', horizontalalignment='right')
    #axs.set_xlim([t1[-1], t1[0]])
    #axs.set_xlim(2.8, 3.5)
    #axs.set_ylim([K_min, K_max])
    axs.set_xlabel("{:d} / T [{:d} $K^{{-1}}$]".format(int(T_scale), int(T_scale)), fontsize=12)
    axs.set_ylabel(r"$\ln(K)$", fontsize=12)
    axs.set_title("van't Hoff", fontsize=14)
    #axs.legend(fontsize="small", loc="best")
    axs.legend(loc="best")


def plot_function_fit(axs,
                      T,
                      TT,
                      ydata,
                      derivative = True,
                      raw_data = None,
                      base_bound = None,
                      base_bound_Tmax = -1,
                      base_unbound = None,
                      base_unbound_Tmin = -1,
                      raw_base_bound = None,
                      raw_base_bound_Tmax = -1,
                      raw_base_unbound = None,
                      raw_base_unbound_Tmin = -1,
                      melting_point = None,
                      fit_label = "Function fit",
                      raw_label = "Raw Data",
                      #min_v = None,
                      #max_v =None,
                      ):

    T = np.array(T)

    ydata = np.array(ydata)

    # plot raw data, if available
    if raw_data is not None and len(raw_data) > 0:
        axs.plot(TT[:len(raw_data)], raw_data, lw = 4, color=col_raw, alpha = 0.66, label = raw_label)

        if raw_base_bound:
            if raw_base_bound_Tmax in T:
                T_stop = next(i for i,v in enumerate(T) if v > raw_base_bound_Tmax)
            else:
                T_stop = int(len(T) / 4)

            axs.plot(T[:T_stop],
                     functions.linear(T[:T_stop], *raw_base_bound),
                     '--',
                     linewidth = 0.66,
                     color = "black",
                     alpha = 0.66,
                     label = None)
            axs.text(T[0],
                     functions.linear(T[0], *raw_base_bound),
                     '$L_2$',
                     alpha = 0.66,
                     verticalalignment='bottom',
                     horizontalalignment='right')

        if raw_base_unbound:
            if raw_base_unbound_Tmin in T:
                T_start = next(i for i, v in enumerate(T) if v > raw_base_unbound_Tmin)
            else:
                T_start = int(3 * len(T) / 4)

            axs.plot(T[T_start:],
                     functions.linear(T[T_start:], *raw_base_unbound),
                     '--',
                     linewidth = 0.66,
                     color = "black",
                     alpha = 0.66,
                     label = None)
            axs.text(T[-1],
                     functions.linear(T[-1], *raw_base_unbound),
                     '$L_1$',
                     alpha = 0.66,
                     verticalalignment='top',
                     horizontalalignment='left')

    # plot the function fit
    axs.plot(T, ydata, lw = 2, color=col_fit, label = fit_label)

    if base_bound:
        if base_bound_Tmax in T:
            T_stop = next(i for i,v in enumerate(T) if v > base_bound_Tmax)
        else:
            T_stop = int(len(T) / 4)

        axs.plot(T[:T_stop],
                 functions.linear(T[:T_stop], *base_bound),
                 '--',
                 linewidth = 0.66,
                 color="black",
                 label = None)
        axs.text(T[T_stop],
                 functions.linear(T[T_stop], *base_bound),
                 '$L_2$',
                 verticalalignment='bottom',
                 horizontalalignment='left')

    if base_unbound:
        if base_unbound_Tmin in T:
            T_start = next(i for i, v in enumerate(T) if v > base_unbound_Tmin)
        else:
            T_start = int(3 * len(T) / 4)

        axs.plot(T[T_start:],
                 functions.linear(T[T_start:], *base_unbound),
                 '--',
                 linewidth = 0.66,
                 color="black",
                 label = None)
        axs.text(T[T_start],
                 functions.linear(T[T_start], *base_unbound),
                 '$L_1$',
                 verticalalignment='top',
                 horizontalalignment='right')

    if melting_point:
        axs.vlines(melting_point[0],
                   min(ydata),
                   melting_point[1],
                   linestyles="dotted",
                   color = "red",
                   label = r"$T_m = {:5.2f} ^\degree C$".format(melting_point[0]))

    axs.set_xlabel(r"Temperature [$^\degree C$]", fontsize=12)
    axs.set_ylabel("Relative Fluorescence", fontsize=12)
    axs.set_title("Full fit", fontsize=14)

    if derivative:
        ax2 = axs.twinx() # second axis that shares x-axis

        # plot derivative d(F)/d(T) of raw data
        d = np.gradient(ydata, 0.5) / -1.
        ax2.plot(T, d, '--', lw = 2, color=col_raw, label = None)
        ax2.set_ylabel("-d(F) / d(T)")

    #axs.legend(fontsize="small", loc="lower right")
    axs.legend(loc="upper left")



def energy_params(axs, x, y, dG_37, dH, dS):
    axs.text(
        x, y,
        dH_string.format(dH) + "\n" + dS_string.format(dS) + "\n" + dG_string.format(dG_37),
        fontsize=10
    )


# ── result-object adapters ─────────────────────────────────────────────────
#
# Higher-level entry points that take a typed result from
# `rnamelt.results` and render it via the building blocks above. Result
# classes call these via lazy imports so matplotlib stays an opt-in
# dependency (Pyodide never has to pull it).


def plot_single(curve, *, raw=None, vh=None, fit=None, axes=None, figsize=(15, 5)):
    """Render a `SingleResult` as a 3-panel figure: raw / van't Hoff / fit.

    Pass `axes` (length-3 sequence of Axes) to draw into existing axes;
    otherwise a new figure is created. Returns `(fig, axes)`.
    """
    raw = raw if raw is not None else curve.raw_tm()
    vh  = vh  if vh  is not None else curve.vantHoff(raw=raw)
    fit = fit if fit is not None else curve.fit(vh=vh)
    T_used, signal = curve.windowed_data()

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig = axes[0].figure

    # ── panel 1: raw + baselines + Tm marker ─────────────────────────────
    mp = False
    if raw.Tm is not None:
        mp = (raw.Tm, raw.ym)


    plot_raw_data(
        axes[0], curve.temps, curve.signal,
        derivative=False,
        base_bound   = tuple(raw.base_b)  if raw.base_b  else False,
        base_bound_Tmax=curve.T_low+30, #curve.bl_lower_offset,
        base_unbound = tuple(raw.base_ub) if raw.base_ub else False,
        base_unbound_Tmin=curve.T_high-curve.bl_upper_offset,
        base_med= tuple(raw.base_med) if raw.base_med else False,
        melting_point=mp,
    )
    axes[0].set_title(f"{curve.name}: raw + baselines")
    axes[0].legend(fontsize="small", loc="best")

    # ── panel 2: van't Hoff (custom — bypasses plot_vantHoff's hard limits)
    if not vh.ok:
        axes[1].text(0.5, 0.5, f"van't Hoff failed:\n{vh.error}",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("van't Hoff")
    else:
        axes[1].plot(vh.t1, vh.K, "-", lw=4, color=col_raw, alpha=0.66,
                     label="Raw (normalized)")
        axes[1].plot(vh.xdata, functions.linear(vh.xdata, *vh.fit_params),
                     "--", lw=2, color=col_fit, label="Linear fit")
        if vh.Tm is not None and not np.isnan(vh.Tm):
            t_m = 1000.0 / (vh.Tm - constants.T0)
            axes[1].axvline(t_m, color="red", ls=":",
                            label=f"Tm = {vh.Tm:.2f} °C")
        if curve.c0:
            axes[1].axhline(np.log(4 / curve.c0), color="black", ls=":", lw=0.66,
                            label=r"$\ln(4/C_0)$")
        axes[1].set_xlabel(r"1000 / T [K$^{-1}$]")
        axes[1].set_ylabel(r"$\ln(K)$")
        axes[1].set_title(f"van't Hoff (ΔH = {vh.dH:.2f} kcal/mol)")
        axes[1].legend(fontsize="small", loc="best")

    # ── panel 3: full non-linear fit ─────────────────────────────────────
    if not fit.ok:
        axes[2].text(0.5, 0.5, f"full fit failed:\n{fit.error}",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("full fit")
    else:
        mp_fit = False
        mp_fit = (fit.Tm, fit.ym)
        plot_function_fit(
            axes[2], T_used, curve.temps, fit.curve_fit,
            derivative=False,
            raw_data=curve.signal,
            base_bound   = tuple(fit.base_b)  if fit.base_b  else None,
            base_unbound = tuple(fit.base_ub) if fit.base_ub else None,
            melting_point=mp_fit,
            fit_label="Function fit",
            raw_label="Raw (windowed)",
        )
        axes[2].set_title(f"Full fit (ΔH = {fit.dH:.2f} kcal/mol)")
        axes[2].legend(fontsize="small", loc="best")

    fig.suptitle(curve.name)
    fig.tight_layout()
    return fig, axes


def plot_multi(result, *, ax=None, figsize=(6, 4)):
    """Render a `MultiResult` (shared-ΔH joint fit) onto one panel.

    Each column gets a raw line + dashed fit in the same colour.
    Returns `(fig, ax)`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if result.error:
        ax.text(0.5, 0.5, f"multi fit failed:\n{result.error}",
                ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    ocs = list(set([c.oligoC for c in result.curves]))
    handles = []
    check_oc = []
    for i, cur in enumerate(result.curves):
        color = palette[ocs.index(cur.oligoC) % len(palette)]
        if cur.signal is not None:
            line = ax.plot(cur.T_used, cur.signal, lw=2, color=color,
                           alpha=0.5,
                    label=f"{cur.oligoC} µM")
            if line[0] not in handles and cur.oligoC not in check_oc:
                handles.append(line[0])
                check_oc.append(cur.oligoC)
        if cur.fit is not None:
            ax.plot(cur.T_used, cur.fit, "--", lw=2, color=color)#,
                    #label=f"{cur.name} fit (Tm={cur.Tm:.2f} °C)")
    ax.set_xlabel(r"Temperature [$^\circ$C]", fontsize=12)
    ax.set_ylabel("Signal", fontsize=12)
    ax.set_title(
        f"Global fit",
        fontsize=14,
    )
    #ax.legend(handles=handles, fontsize="small", loc="best")
    ax.legend(handles=handles, loc="best")
    #fig.tight_layout()
    return fig, ax


def plot_concentration(result, *, axes=None, figsize=(13, 5), legend=True):
    """Render a `ConcentrationResult` as 2 panels: curves + 1/Tm vs ln(C_T/f).

    Panel 1 overlays all per-curve raw signals with their fit-Tm markers.
    Panel 2 plots the three van't Hoff regressions (raw / vH / fit).
    Returns `(fig, axes)`.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = axes[0].figure
    if result.error:
        axes[0].text(0.5, 0.5, f"concentration analysis failed:\n{result.error}",
                     ha="center", va="center", transform=axes[0].transAxes)
        return fig, axes

    # ── panel 1: per-curve overlay ──────────────────────────────────────
    for i, c in enumerate(result.per_curve):
        color = palette[i % len(palette)]
        if c.signal is not None and c.T_used is not None:
            axes[0].plot(c.T_used, c.signal, lw=2, color=color, alpha=0.75,
                         label=f"{c.name} ({c.oligoC} µM)")
        if c.Tm_fit is not None and not np.isnan(c.Tm_fit):
            axes[0].axvline(c.Tm_fit, color=color, ls=":", lw=0.66)
    axes[0].set_xlabel(r"Temperature [$^\circ$C]")
    axes[0].set_ylabel("Signal")
    axes[0].set_title("Per-curve signals (dotted lines: fit Tm)")
    if legend:
        axes[0].legend(fontsize="small", loc="best")

    dH_string = r'$\Delta H = {:4.2f}~\frac{{kcal}}{{mol}}$'
    dS_string = r'$\Delta S = {:4.2f}~\frac{{kcal}}{{mol \cdot K}}$'
    dG_string = r'$\Delta G_{{37}} = {:4.2f}~\frac{{kcal}}{{mol}}$'

    # ── panel 2: 1/Tm vs ln(C_T / f) regression for each method ─────────
    method_colors = {"raw": col_raw, "vh": col_median, "fit": col_fit}
    for series in (result.series_raw, result.series_vh, result.series_fit):
        if series is None or not series.ok:
            continue
        c = method_colors.get(series.method, "black")
        axes[1].plot(series.ln_ct, series.inv_tm, "o", color=c,
                     label=f"{series.method} (n={series.n})")
        axes[1].plot(series.ln_ct_line, series.inv_tm_line, "--", color=c, lw=1.5,
                     label=f"{series.method}: ΔG={series.dG:.2f}, ΔH={series.dH:.2f},  ΔS={series.dS:.2f}, r²={series.r_squared:.4f}")
    axes[1].set_xlabel(r"$\ln(C_T / f)$")
    axes[1].set_ylabel(r"$1 / T_m$ [K$^{-1}$]")
    axes[1].set_title("Van't Hoff regression")
    axes[1].legend(fontsize="small", loc="best")
    fig.tight_layout()
    return fig, axes


def plot_conc(result, *, ax=None, figsize=(13, 5), legend=True):
    """Render a `ConcentrationResult` as 1 panel: 1/Tm vs ln(C_T/f).

    Returns `(fig, axes)`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if result.error:
        ax.text(0.5, 0.5, f"concentration analysis failed:\n{result.error}",
                     ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    # ── panel 2: 1/Tm vs ln(C_T / f) regression for each method ─────────
    method_colors = {"raw": col_raw, "vh": col_median, "fit": col_fit}
    for series in (result.series_raw, result.series_vh, result.series_fit):
        if series is None or not series.ok:
            continue
        c = method_colors.get(series.method, "black")
        ax.plot(series.ln_ct, series.inv_tm, "o", color=c,
                #label=f"{series.method} (n={series.n})"
                )
        ax.plot(
            series.ln_ct_line, series.inv_tm_line, "--", color=c, lw=1.5,
            #label="{:3s}: ΔG={:6.2f}, ΔH={:7.2f}, ΔS={:5.2f}, r²={:5.3f}".format(
            #    series.method, series.dG, series.dH,
            #    series.dS, series.r_squared
            #)
            label=f"{series.method}: ΔG={series.dG:.2f}, ΔH={series.dH:.2f},  ΔS={series.dS:.2f}, r²={series.r_squared:.4f}"
        )
    ax.set_xlabel(r"$\ln(C_T / f)$", fontsize=12)
    ax.set_ylabel(r"$1 / T_m$ [K$^{-1}$]", fontsize=12)
    ax.set_title("Van't Hoff regression", fontsize=14)
    ax.legend(loc="lower left", fontsize="small")
    #fig.tight_layout()
    return fig, ax

