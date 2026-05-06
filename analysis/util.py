#import os
import methods, functions
#from analysis import methods, functions
import numpy as np
#from io import StringIO
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
    # store_cb = store_results,
    # store_data = None,
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
                                                                       border = 0.15,
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

    return {'name':             strands,
            'oligoC':           oligo_c,
            'saltC':            salt_c,
            'TmRaw':            T_m_r,
            'TmVH':             T_m,
            'TmFit':            T_m_f,
            'dGVH':             dG_37,
            'dHVH':             dH,
            'dSVH':             dS,
            'dGFit':            dG_37_f,
            'dHFit':            dH_f,
            'dSFit':            dS_f,
            'T_raw':            list(T),
            'T_low':            T_low,
            'T_high':           T_low,
            'T_used':           TT,
            'derivative':       derivative,
            'base_b_offset':    base_b_offset,
            'base_ub_offset':   base_ub_offset,
            'fit':              fit,
            'raw_data':         input
    }

    ## 3. Store results
    #if store_cb:
    #    store_cb(strands = strands,
    #             oligo_c = oligo_c,
    #             salt_c = salt_c,
    #             T_m_raw = T_m_r,
    #             T_m_vH = T_m,
    #             T_m_fit = T_m_f,
    #             dG_37_vH = dG_37,
    #             dH_vH = dH,
    #             dS_vH = dS,
    #             dG_37_fit = dG_37_f,
    #             dH_fit = dH_f,
    #             dS_fit = dS_f,

    #             ##Raw data
    #             T = T,
    #             T_low = T_low,
    #             T_high = T_high,
    #             derivative = derivative,
    #             fit = fit,
    #             raw_data = input,

    #             data = store_data
    #            )

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

