import matplotlib.pyplot as plt
import numpy as np
from . import functions, constants

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

        axs.plot(T[:T_stop], functions.linear(T[:T_stop], *base_bound), '--', linewidth = 0.66, color="black", label = None)
        axs.text(T[T_stop], functions.linear(T[T_stop], *base_bound),'$L_2$', verticalalignment='center', horizontalalignment='left')

    if base_unbound:
        if base_unbound_Tmin in T:
            T_start = next(i for i, v in enumerate(T) if v > base_unbound_Tmin)
        else:
            T_start = int(3 * len(T) / 4)

        axs.plot(T[T_start:], functions.linear(T[T_start:], *base_unbound), '--', linewidth = 0.66, color="black", label = None)
        axs.text(T[T_start], functions.linear(T[T_start], *base_unbound),'$L_1$', verticalalignment='center', horizontalalignment='right')

    if base_med:
        if base_med_Tmin in T:
            T_start = next(i for i, v in enumerate(T) if v > base_med_Tmin)
        else:
            T_start = int(len(T) / 4)

        if base_med_Tmax in T:
            T_stop = next(i for i, v in enumerate(T) if v > base_med_Tmax)
        else:
            T_stop = int(3 * len(T) / 4)

        axs.plot(T[T_start:T_stop], functions.linear(T[T_start:T_stop], *base_med), '--', linewidth = 0.66, color="black", label = None)
        axs.text(T[T_stop], functions.linear(T[T_stop], *base_med),r'$\frac{1}{2} (L_1 + L_2)$', verticalalignment='center', horizontalalignment='left')

    if melting_point:
        axs.vlines(melting_point[0], min(ydata), melting_point[1], linestyles="dotted", color = "red", label = "$T_m = %5.2f ^\degree C$" % melting_point[0])

    axs.set_xlabel("Temperature [$^\degree C$]")
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
    axs.plot(t1, K, '-', lw = 4, color=col_raw, alpha = 0.66, label="Raw Data (normalized)")
    axs.plot(xdata, functions.linear(xdata, *fit), '--', lw = 2, color=col_fit, label='Linear Fit')

    if len(xdata) > 0 or len(xdata) > 0:
        axs.hlines(np.log(4 / c0), t1[-1], max(xdata), linestyles="dotted", color="black", label = None)
        axs.vlines(t_m, min(ydata), np.log(4 / c0), linestyles="dotted", color = "red", label = "$T_m = %5.2f ^\degree C$" % T_m)
    axs.text(t1[-1] - 0.1, np.log(4 / c0), r'$K_a = \frac{4}{C_o}$', verticalalignment='center', horizontalalignment='right')
    #axs.set_xlim([t1[-1], t1[0]])
    axs.set_xlim(2.8, 3.5)
    axs.set_ylim([K_min, K_max])
    axs.set_xlabel("{:d} / T [{:d} $K^{{-1}}$]".format(int(T_scale), int(T_scale)))
    axs.set_ylabel("$\ln(K_a)$")


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
    if raw_data:
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
                   label = "$T_m = {:5.2f} ^\degree C$".format(melting_point[0]))

    axs.set_xlabel("Temperature [$^\degree C$]")
    axs.set_ylabel("Relative Fluorescence")

    if derivative:
        ax2 = axs.twinx() # second axis that shares x-axis

        # plot derivative d(F)/d(T) of raw data
        d = np.gradient(ydata, 0.5) / -1.
        ax2.plot(T, d, '--', lw = 2, color=col_raw, label = None)
        ax2.set_ylabel("-d(F) / d(T)")


def energy_params(axs,
                  x,
                  y,
                  dG_37,
                  dH,
                  dS):
    axs.text(x, y, dH_string.format(dH) + "\n" + dS_string.format(dS) + "\n" + dG_string.format(dG_37))
