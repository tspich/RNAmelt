import os
from analysis import methods, functions
import numpy as np
from io import StringIO
#from matplotlib.figure import Figure
#from matplotlib.backends import backend_svg
from statistics import mean


#from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider
#from mpl_toolkits.axes_grid1 import Divider, Size

#default_baseline_fit = [20, 70]

#r_base_b_maxT = 20
#r_base_ub_minT = 70

default_vH = [-1,-1]

raw_vH = {}
fit_dH = {}

border_vH = {}


raw_cut_fit = {}

def store_results(strands,
                  oligo_c,
                  salt_c,
                  T_m_raw = None,
                  T_m_vH = None,
                  T_m_fit = None,
                  dG_37_vH = None,
                  dH_vH = None,
                  dS_vH = None,
                  dG_37_fit = None,
                  dH_fit = None,
                  dS_fit = None,
                  T = None,
                  T_low = None,
                  T_high = None,
                  derivative = None,
                  base_b_offset=10,
                  base_ub_offset=10,
                  fit=None,
                  raw_data = None,
                  data = None
                 ):


    #if data.exists(strands, oligo_c, salt_c):
    data.append({
            'name':             strands,
            'oligoC':           oligo_c,
            'saltC':            salt_c,
            'TmRaw':            T_m_raw,
            'TmVH':             T_m_vH,
            'TmFit':            T_m_fit,
            'dGVH':             dG_37_vH,
            'dHVH':             dH_vH,
            'dSVH':             dS_vH,
            'dGFit':            dG_37_fit,
            'dHFit':            dH_fit,
            'dSFit':            dS_fit,
            'T_raw':            list(T),
            'T_low':            T_low,
            'T_high':           T_low,
            'derivative':       derivative,
            'base_b_offset':    base_b_offset,
            'base_ub_offset':   base_ub_offset,
            'fit':              fit,
            'raw_data':         raw_data})

def analyze(
    strands,
    salt_c,
    oligo_c,
    T_low,
    T_high,
    input,
    store_cb = store_results,
    store_data = None,
    T=np.arange(5,85.5, 0.5),
    base_b_offset=10,
    base_ub_offset=10,
    dh_init=None,
    ds_init=None,
):
    TT = np.array(T)
    data = input
    # NOTE: oligo c is given per strand, therefore *2
    c0 = 1e-6 * oligo_c*2

    if T_low is None:
        T_low = TT[0]
    if T_high is None:
        T_high = TT[-1]

    pos_start = np.where(TT == T_low)[0][0]
    pos_end   = np.where(TT == T_high)[0][0]+1

    TT = TT[pos_start:pos_end]
    data = input[pos_start:pos_end]

    r_base_b_maxT=T[0]+base_b_offset
    r_base_ub_minT=T[-1]-base_ub_offset


    # 1. process raw RFU data
    # 1.1 intersection of median with base lines
    T_m_r, y_r, base_b_r, base_ub_r, base_med_r = methods.T_m_ds_raw(TT,
                                                                     data,
                                                                     baseline_bound_maxT=r_base_b_maxT,
                                                                     baseline_unbound_minT=r_base_ub_minT,
                                                                     #debug=True
                                                                    )

    # 1.2 van't Hoff analysis with base lines
    T_m, dG_37, dH, dS, t1, K, xdata, ydata, fit_vh = methods.vantHoff(TT,
                                                                       data,
                                                                       *base_b_r,
                                                                       *base_ub_r,
                                                                       c0,
                                                                       #border = border,
                                                                       #t1_min = t1_min,
                                                                       #t1_max = t1_max
                                                                       )

    # 2. fit the flourescence data to a full function with dH and dS
    #dH_init = -80
    if dh_init is None:
        dH_init = dH #-80
    else:
        dH_init = dh_init
    if ds_init is None:
        dS_init = dS
    else:
        dS_init = ds_init


    dG_37_f, dH_f, dS_f, T_m_f, y_f, base_b_f, base_ub_f, base_med_f = methods.fit_full_function(TT,
                                                                                                 data,
                                                                                                 c0=c0,
                                                                                                 dH_init = dH_init, #dH from the vH analysis
                                                                                                 dS_init = dS_init,
                                                                                                 )

    # compute list of predicted flourescence data points according to function fit
    fit = np.array([ functions.full_function(tt, dH_f, dS_f, *base_b_f, *base_ub_f, c0=c0) for tt in TT ])
    # NOTE: May need to check that, not general enough?
    derivative = np.gradient(data, 0.5)/-1.

    # 3. Store results
    if store_cb:
        store_cb(strands = strands,
                 oligo_c = oligo_c,
                 salt_c = salt_c,
                 T_m_raw = T_m_r,
                 T_m_vH = T_m,
                 T_m_fit = T_m_f,
                 dG_37_vH = dG_37,
                 dH_vH = dH,
                 dS_vH = dS,
                 dG_37_fit = dG_37_f,
                 dH_fit = dH_f,
                 dS_fit = dS_f,

                 ##Raw data
                 T = T,
                 T_low = T_low,
                 T_high = T_high,
                 derivative = derivative,
                 fit = fit,
                 raw_data = input,

                 data = store_data
                )

# TODO: oligo_c should be a list for multi fit.
def analyze_multi(strands, salt_c, oligo_c, store_cb = store_results, store_data = None, input = None, T=np.arange(5,85.5, 0.5), start=5, end=85, r_base_b_maxT=20, r_base_ub_minT=70, res_diff_method = 'square'):
    r_base_b_maxT=T[0]+10
    r_base_ub_minT=T[-1]-10

    data = input
    TT = np.array(T)    
    c0 = 1e-6 * oligo_c*2

    # NOTE: Redoing singls curve analysis to get parameters for multi fit, any better way?
    #       Should vH analysis be redone as well to get dH dS for single curve fit?
    b1_inits = []
    b2_inits = []
    dh_inits = []
    ds_inits = []
    for d in data:
        dG_37_f, dH_f, dS_f, T_m_f, y_f, base_b_f, base_ub_f, base_med_f = methods.fit_full_function(TT,
                                                                                                     d,
                                                                                                     c0=c0,
                                                                                                     dH_init = -80,
                                                                                                     dS_init = -0.2,
                                                                                                     #max_v = max_value
                                                                                                    )
        b1_inits.append(base_b_f)
        b2_inits.append(base_ub_f)
        dh_inits.append(dH_f)
        ds_inits.append(dS_f)

    # start multi analysis
    #NOTE: Maybe not the best to redefined the same variables as before.
    dG_37_f, dH_f, dS_f, T_m_f, y_f, base_b_f, base_ub_f, base_med_f = methods.fit_full_function_multi(TT,
                                                                                                       data,
                                                                                                       c0=c0, 
                                                                                                       dH_init = mean(dh_inits),
                                                                                                       dS_init = mean(ds_inits),
                                                                                                       b1_inits = b1_inits,
                                                                                                       b2_inits = b2_inits,
                                                                                                       residuals_method=res_diff_method,
                                                                                                      )

    base_b =  [j for a,i in enumerate(base_b_f) for j in i  ]
    base_ub = [j for a,i in enumerate(base_ub_f) for j in i  ] 

    # collect b and ub base lines for each data set as tuples of 4
    # This needs to be done as the full_function_multi() function
    # requires them to be passed that way
    base_params = []
    for i in range(0, len(base_b), 2):
        base_params.extend([base_b[i], base_b[i + 1]])
        base_params.extend([base_ub[i], base_ub[i + 1]])

    #fit = np.array([ functions.full_function_multi(tt, [dH_f, dS_f, *base_params], c0=c0) for tt in TT ])

    if store_cb:
        store_cb(strands   = strands,
                 oligo_c   = oligo_c,
                 salt_c    = salt_c,
                 T_m_fit   = mean(T_m_f),
                 dG_37_fit = dG_37_f,
                 dH_fit    = dH_f,
                 dS_fit    = dS_f,

                 ##Raw data
                 T = T,
                 start = start,
                 end = end,
                 raw_data = input,

                 data = store_data
                )

#def plot_to_svg(fig) -> str:
#    """
#    Saves the last plot made using ``matplotlib.pyplot`` to a SVG string.
#    
#    Returns:
#        The corresponding SVG string.
#    """
#    s = StringIO()
#    f = backend_svg.FigureCanvasSVG(fig)
#    f.print_svg(s)
#
#    return s.getvalue()
#
#def make_plot(oligo_c, data, start, end, T, min_v=None, max_v=None, base_b_offset=10, base_ub_offset=10, dh_init=None, ds_init=None, res_diff_method = 'square'):
#    TT = np.array(T)
#    raw_data = data
#    c0 = 1e-6 * oligo_c*2
#
#    if len(data) == len(T):
#        if start is None:
#            start = TT[0]
#        if end is None or end == TT[-1]:
#            end = TT[-1]
#
#        pos_start = np.where(TT == start)[0][0]
#        pos_end   = np.where(TT == end)[0][0]+1
#
#        TT = TT[pos_start:pos_end]
#        data = data[pos_start:pos_end]
#
#        r_base_b_maxT = TT[0]+base_b_offset
#        r_base_ub_minT = TT[-1]-base_ub_offset
#
#        #NOTE Do we need to be able to adapt baseline bound?
#        T_m_r, y_r, base_b_r, base_ub_r, base_med_r = methods.T_m_ds_raw(TT,
#                                                                         data,
#                                                                         baseline_bound_maxT=r_base_b_maxT,
#                                                                         baseline_unbound_minT=r_base_ub_minT,
#                                                                         #debug=True
#                                                                        )
#
#        # 1.2 van't Hoff analysis with base lines
#        T_m, dG_37, dH, dS, t1, K, xdata, ydata, fit_vh = methods.vantHoff(TT,
#                                                                           data,
#                                                                           *base_b_r,
#                                                                           *base_ub_r,
#                                                                           c0,
#                                                                           #border = border,
#                                                                           #t1_min = t1_min,
#                                                                           #t1_max = t1_max
#                                                                           )
#
#        # 2. fit the flourescence data to a full function with dH and dS
#        if dh_init is None:
#            dH_init = dH #-80
#        else:
#            dH_init = dh_init
#        if ds_init is None:
#            dS_init = dS
#        else:
#            dS_init = ds_init
#
#        width = 12
#        height = 5
#
#        fig = Figure(figsize=(width, height), layout=None)
#        ax = fig.subplots(1, 2) #plt.subplots(1, 2, figsize=(width, height))
#
#        try:
#            dG_37_f, dH_f, dS_f, T_m_f, y_f, base_b_f, base_ub_f, base_med_f = methods.fit_full_function(TT,
#                                                                                                         data,
#                                                                                                         c0=c0,
#                                                                                                         dH_init = dH_init,
#                                                                                                         dS_init = dS_init,
#                                                                                                         )
#
#            # compute list of predicted flourescence data points according to function fit
#            fit = np.array([ functions.full_function(tt, dH_f, dS_f, *base_b_f, *base_ub_f, c0=c0) for tt in TT ])
#
#
#            plots.plot_function_fit(ax[0],
#                                    TT,
#                                    np.array(T),
#                                    fit,
#                                    derivative=False,
#                                    raw_data = raw_data,
#                                    base_bound = base_b_f,
#                                    base_unbound = base_ub_f,
#                                    raw_base_bound = base_b_r,
#                                    raw_base_unbound = base_ub_r,
#                                    melting_point = (T_m_f, y_f)
#                                    )
#
#        except:
#            dG_37_f=0
#            dH_f=0
#            dS_f=0
#            T_m_f=0
#            y_f=0
#            plots.plot_raw_data(ax[0],
#                                np.array(T),
#                                raw_data)
#
#        finally:
#            ax[0].axvline(x = start, color='black', linestyle="--", alpha=1, gid='min_val')
#            ax[0].axvline(x = end, color='black', linestyle="--", alpha=1, gid='max_val')
#            ax[0].axvline(x = T[0], color='black', linestyle="--", alpha=0.1, gid='min_T')
#            ax[0].axvline(x = T[-1], color='black', linestyle="--", alpha=0.1, gid='max_T')
#
#            if min_v and max_v:
#                ax[0].set_ylim(min_v, max_v)
#            else:
#                min_v, max_v = ax[0].get_ylim()
#
#            ax[0].set_xlim(T[0]-10, T[-1]+10)
#
#            #print(ax[0].set_xlim())
#
#        # 4.2 van't Hoff on raw data
#        plots.plot_vantHoff(ax[1], t1, K, c0, xdata, ydata, fit_vh, T_m)
#
#        svg_dta = plot_to_svg(fig)
#                
#        plot_data = {'svg':svg_dta,
#                     'min_v':min_v,
#                     'max_v':max_v,
#                     'start':start,
#                     'end':end,
#                     'base_b_offset': base_b_offset,
#                     'base_ub_offset': base_ub_offset,
#                     'TmRaw':  T_m_r,
#                     'TmVH':   T_m , 
#                     'TmFit':  T_m_f,
#                     'dGVH':   dG_37,
#                     'dHVH':   dH   ,
#                     'dSVH':   dS   ,
#                     'dGFit':  dG_37_f,
#                     'dHFit':  dH_f   ,
#                     'dSFit':  dS_f   ,
#                     'dh_init'   : dH_init,
#                     'ds_init'   : dS_init,
#                     }
#    # Multi curves analysis
#    else:
#        pos_start = np.where(TT == start)[0][0]
#        pos_end   = np.where(TT == end)[0][0]+1
#
#        TT = TT[pos_start:pos_end]
#        data = np.asarray([dd[pos_start:pos_end] for dd in data])
#
#        b1_inits = []
#        b2_inits = []
#        dh_inits = []
#        ds_inits = []
#        for d in data:
#            dG_37_f, dH_f, dS_f, T_m_f, y_f, base_b_f, base_ub_f, base_med_f = methods.fit_full_function(TT,
#                                                                                                         d,
#                                                                                                         c0=c0,
#                                                                                                         dH_init = -80,
#                                                                                                         dS_init = -0.2,
#                                                                                                         #max_v = max_value
#                                                                                                        )
#            b1_inits.append(base_b_f)
#            b2_inits.append(base_ub_f)
#            dh_inits.append(dH_f)
#            ds_inits.append(dS_f)
#
#        #res_diff_method = 'quadratic'
#        dG_37_f, dH_f, dS_f, T_m_f, y_f, base_b_f, base_ub_f, base_med_f = methods.fit_full_function_multi(TT,
#                                                                                                           data,
#                                                                                                           c0=c0, 
#                                                                                                           dH_init = mean(dh_inits),
#                                                                                                           dS_init = mean(ds_inits),
#                                                                                                           b1_inits = b1_inits,
#                                                                                                           b2_inits = b2_inits,
#                                                                                                           residuals_method=res_diff_method,
#                                                                                                          )
#
#        base_b =  [j for a,i in enumerate(base_b_f) for j in i  ]
#        base_ub = [j for a,i in enumerate(base_ub_f) for j in i  ] 
#
#        # collect b and ub base lines for each data set as tuples of 4
#        # This needs to be done as the full_function_multi() function
#        # requires them to be passed that way
#        base_params = []
#        for i in range(0, len(base_b), 2):
#            base_params.extend([base_b[i], base_b[i + 1]])
#            base_params.extend([base_ub[i], base_ub[i + 1]])
#
#        fit = np.array([ functions.full_function_multi(tt, [dH_f, dS_f, *base_params], c0=c0) for tt in TT ])
#
#        width = 8
#        height = 5
#
#        fig = Figure(figsize=(width, height))
#
#        axs = fig.subplots(1)#, layout='constrained')# 4, sharex = False, sharey = False, figsize = (20, 10))
#
#        axs.plot(np.array(T), np.array(raw_data).T, lw = 4, alpha = 0.66)
#        axs.plot(TT, fit, '--', lw = 2)
#
#        axs.axvline(x = start, color='black', linestyle="--", alpha=1, gid='min_val')
#        axs.axvline(x = end, color='black', linestyle="--", alpha=1, gid='max_val')
#        axs.axvline(x = T[0], color='black', linestyle="--", alpha=0.1, gid='min_T')
#        axs.axvline(x = T[-1], color='black', linestyle="--", alpha=0.1, gid='max_T')
#
#        if min_v and max_v:
#            axs.set_ylim(min_v, max_v)
#        else:
#            min_v, max_v = axs.get_ylim()
#
#        axs.set_xlim(T[0]-10, T[-1]+10)
#
#        #fig.tight_layout(rect=[0, 0.03, 1, 0.99])
#
#        svg_dta = plot_to_svg(fig)
#
#        plot_data = {'svg':svg_dta,
#                     'min_v':min_v,
#                     'max_v':max_v,
#                     'start':start,
#                     'end':end,
#                     'base_b_offset': base_b_offset,
#                     'base_ub_offset': base_ub_offset,
#                     'T_m_raw'   : None,
#                     'T_m_vH'    : None, 
#                     'T_m_fit'   : mean(T_m_f),
#                     'dG_37_vH'  : None,
#                     'dH_vH'     : None,
#                     'dS_vH'     : None,
#                     'dG_37_fit' : dG_37_f,
#                     'dH_fit'    : dH_f   ,
#                     'dS_fit'    : dS_f   ,
#                     'dh_init'   : mean(dh_inits),
#                     'ds_init'   : mean(ds_inits),
#                     }

    return plot_data
