"""Automated baseline detection for melting curves.

Not wired into ``rnamelt/__init__.py``; opt in explicitly:

    from rnamelt.auto_baseline import find_baselines
    fold_low_T, fold_high_T, unf_low_T, unf_high_T = find_baselines(T, signal)

``find_baselines`` returns four temperatures delimiting the two baseline
windows: the folded baseline is fit on data with
``fold_low_T <= T <= fold_high_T``, the unfolded baseline on data with
``unf_low_T <= T <= unf_high_T``, and the transition sits between
``fold_high_T`` and ``unf_low_T``.

Algorithm (signed second derivative + convergence trim)
-------------------------------------------------------
1. Determine transition direction from the sign of ``S[-n_edge:] - S[:n_edge]``.
2. Compute ``dS/dT`` and ``d^2 S/dT^2`` (Savitzky-Golay by default).
3. Locate the transition centre from the signed ``dS/dT`` extremum
   consistent with the detected direction.
4. Locate the two sigmoid shoulders as signed ``d^2`` extrema on each
   side of the centre. Cold and hot shoulders have opposite signs by
   construction of a two-state transition.
5. Walk outward from each shoulder until ``d^2 * shoulder_sign`` drops
   below ``zero_tol_frac * |d^2_shoulder|`` -- a genuine decay to a small
   fraction of the shoulder peak, robust to small noise wiggles.
6. OLS-fit the resulting baseline window; iteratively drop the outermost
   point if it meaningfully improves the RMS residual.

Earlier prototype variants (threshold, detrended, d2-crossings, pipeline,
concord, group-concord, tangent baselines) are preserved verbatim in
``rnamelt/auto_baseline_tests.py`` and re-imported there by the scripts
in ``scripts/``.

The standalone diagnostic ``detect_no_plateau`` -- which flags whether
the folded and/or unfolded plateau is missing from the scan -- is kept
here as it does not depend on any of the removed variants.

CLI demo
--------
    python -m rnamelt.auto_baseline path/to/data.csv

Prints the detected baseline window for every signal column in a
RNAmelt-format CSV (temperature in column 0, one signal per subsequent
column).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _trim_by_convergence(
    T_bl: np.ndarray, S_bl: np.ndarray, side: str,
    conv_tol: float = 0.05, max_drop_frac: float = 0.33,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Iteratively drop points from the OUTER edge (scan edge) of a baseline
    window, stopping when the relative RMS-residual improvement drops below
    ``conv_tol``. Rejects outliers only when they meaningfully hurt the fit.

    side='cold' -> drops from index 0 (lowest T);
    side='hot'  -> drops from index -1 (highest T).
    """
    T_bl = np.asarray(T_bl, dtype=float).copy()
    S_bl = np.asarray(S_bl, dtype=float).copy()
    n0 = len(T_bl)
    if n0 < 4:
        return T_bl, S_bl, 0

    def _rms(T_arr, S_arr):
        m, b = np.polyfit(T_arr, S_arr, 1)
        return float(np.sqrt(np.mean((S_arr - (m * T_arr + b)) ** 2)))

    max_drops = max(1, int(max_drop_frac * n0))
    rms_current = _rms(T_bl, S_bl)
    dropped = 0
    for _ in range(max_drops):
        if len(T_bl) < 4:
            break
        if side == "cold":
            T_try, S_try = T_bl[1:], S_bl[1:]
        else:
            T_try, S_try = T_bl[:-1], S_bl[:-1]
        rms_new = _rms(T_try, S_try)
        rel_impr = (rms_current - rms_new) / max(rms_current, 1e-12)
        if rel_impr < conv_tol:
            break
        T_bl, S_bl = T_try, S_try
        rms_current = rms_new
        dropped += 1
    return T_bl, S_bl, dropped


def detect_no_plateau(
    T: np.ndarray,
    signal: np.ndarray,
    smooth_window: int = 11,
    smooth_polyorder: int = 3,
    edge_frac: float = 0.05,
    edge_deriv_frac: float = 0.30,
) -> Tuple[bool, bool]:
    """Detect whether the folded and/or unfolded plateau is missing from the
    scan (i.e. the transition already extends past the scan edge).

    Signal: on a side that has a real plateau, the smoothed ``|dS/dT|`` in
    the outermost ``edge_frac`` fraction of points is tiny compared to the
    transition peak. When the plateau is off-frame, the transition is still
    ongoing at the scan edge, so the edge-region ``|dS/dT|`` is a large
    fraction of the peak.

    Returns (no_folded, no_unfolded). Both False on a nicely-plateaued curve.

    Rationale for defaults: ``edge_frac=0.05`` uses the outer 5% of the
    curve on each side (minimum 3 points), and ``edge_deriv_frac=0.30``
    calls a side "no plateau" when the mean edge ``|dS/dT|`` exceeds 30%
    of the transition peak -- comfortably above what a linear-drift baseline
    would ever produce (typical drift is 5-15% of peak).
    """
    from scipy.signal import savgol_filter

    T = np.asarray(T, dtype=float)
    S = np.asarray(signal, dtype=float)
    n = len(T)
    if n < smooth_window:
        return False, False

    w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    p = min(smooth_polyorder, w - 1)
    dt = float(np.median(np.diff(T)))
    dSdT = savgol_filter(S, w, p, deriv=1) / dt
    abs_d = np.abs(dSdT)
    peak = float(abs_d.max())
    if peak <= 0.0:
        return False, False

    n_edge = max(3, int(edge_frac * n))
    mean_cold = float(np.mean(abs_d[:n_edge]))
    mean_hot = float(np.mean(abs_d[-n_edge:]))
    thr = edge_deriv_frac * peak
    return (mean_cold > thr), (mean_hot > thr)


def find_baselines(
    T: np.ndarray,
    signal: np.ndarray,
    smooth: bool = True,
    smooth_w1: int = 11,
    smooth_w2: int = 21,
    smooth_p: int = 3,
    zero_tol_frac: float = 0.10,
    outlier_conv_tol: float = 0.05,
    outlier_max_drop_frac: float = 0.33,
    min_baseline_width: float = 5.0,
) -> Tuple[float, float, float, float]:
    """Baseline detection using **signed** ``d^2 S/dT^2`` sign-decays past the
    sigmoid shoulders, followed by convergence-based outlier trimming on
    the outer (scan-edge) side of each baseline window.

    Return
    ------
    ``(fold_low_T, fold_high_T, unf_low_T, unf_high_T)`` -- four
    temperatures delimiting the two baseline windows:

      * folded baseline is fit on data with ``fold_low_T <= T <= fold_high_T``
      * unfolded baseline is fit on data with ``unf_low_T <= T <= unf_high_T``

    ``fold_high_T`` and ``unf_low_T`` are the transition-facing boundaries.
    ``fold_low_T`` and ``unf_high_T`` are the scan-edge boundaries; they
    equal the scan edges unless the trimming step drops outliers there.

    Parameters
    ----------
    T, signal : arrays; T strictly ascending.
    smooth : if False, uses ``np.gradient`` for both derivatives -- useful
        for inspecting how much the smoothing choice biases the boundaries.
    smooth_w1, smooth_w2, smooth_p : Savitzky-Golay parameters (used when
        ``smooth=True``).
    zero_tol_frac : the walk exits when ``d^2 * shoulder_sign`` drops below
        ``zero_tol_frac * |d^2_shoulder|`` -- i.e. when d^2 has decayed to
        that fraction of the shoulder peak. ``0`` recovers strict
        sign-change behaviour (walks until d^2 actually flips). Higher
        values stop earlier (closer to the shoulder, so the baseline
        extends less far); lower values stop later. Default 0.10.
    outlier_conv_tol : relative RMS-residual improvement threshold below
        which outlier trimming stops.
    outlier_max_drop_frac : hard cap on the fraction of points that may
        be trimmed off the outer edge of each baseline.
    min_baseline_width : minimum baseline window width (degrees C).
    """
    T = np.asarray(T, dtype=float)
    S = np.asarray(signal, dtype=float)
    n = len(T)
    if n < 10:
        raise ValueError(f"need at least 10 points; got {n}")
    if not np.all(np.diff(T) > 0):
        raise ValueError("T must be strictly increasing")

    dt = float(np.median(np.diff(T)))
    n_edge = max(3, n // 10)
    direction_up = float(S[-n_edge:].mean()) > float(S[:n_edge].mean())

    if smooth:
        from scipy.signal import savgol_filter
        w1 = smooth_w1 if smooth_w1 % 2 == 1 else smooth_w1 + 1
        w2 = smooth_w2 if smooth_w2 % 2 == 1 else smooth_w2 + 1
        p = min(smooth_p, min(w1, w2) - 1)
        if n < w2:
            raise ValueError(f"need at least {w2} points for smoothing; got {n}")
        d1 = savgol_filter(S, w1, p, deriv=1) / dt
        d2 = savgol_filter(S, w2, p, deriv=2) / (dt * dt)
    else:
        d1 = np.gradient(S, T)
        d2 = np.gradient(d1, T)

    peak_idx = int(np.argmax(d1) if direction_up else np.argmin(d1))

    # Shoulders: opposite signs on each side of Tm by construction.
    if direction_up:
        cold_shoulder = int(np.argmax(d2[: peak_idx + 1]))
        hot_shoulder = int(np.argmin(d2[peak_idx:])) + peak_idx
    else:
        cold_shoulder = int(np.argmin(d2[: peak_idx + 1]))
        hot_shoulder = int(np.argmax(d2[peak_idx:])) + peak_idx

    cold_sign = int(np.sign(d2[cold_shoulder]))
    hot_sign = int(np.sign(d2[hot_shoulder]))
    cold_thresh = zero_tol_frac * abs(float(d2[cold_shoulder]))
    hot_thresh  = zero_tol_frac * abs(float(d2[hot_shoulder]))

    # Walk cold-side outward (toward index 0) until d^2 on the shoulder-signed
    # side has decayed below cold_thresh, i.e. d^2 * cold_sign < cold_thresh.
    # Fires when d^2 has decayed to zero_tol_frac of the shoulder magnitude on
    # the same side, OR when it flips sign -- whichever happens first.
    cold_boundary = 0
    for i in range(cold_shoulder - 1, -1, -1):
        if float(d2[i]) * cold_sign < cold_thresh:
            cold_boundary = i
            break

    # Walk hot-side outward (toward index n-1) with the same rule.
    hot_boundary = n - 1
    for i in range(hot_shoulder + 1, n):
        if float(d2[i]) * hot_sign < hot_thresh:
            hot_boundary = i
            break

    # Initial baseline window edges.
    fold_low_T = float(T[0])
    fold_high_T = float(T[cold_boundary])
    unf_low_T = float(T[hot_boundary])
    unf_high_T = float(T[-1])

    # min_baseline_width guard -- if the transition abuts a scan edge and
    # the boundary landed there, push it in by min_baseline_width.
    if fold_high_T - fold_low_T < min_baseline_width:
        fold_high_T = float(min(T[0] + min_baseline_width, T[peak_idx] - 1.0))
    if unf_high_T - unf_low_T < min_baseline_width:
        unf_low_T = float(max(T[-1] - min_baseline_width, T[peak_idx] + 1.0))

    # Trim outer edges via _trim_by_convergence. Its returned window's
    # OUTER edge is the trimmed fold_low_T / unf_high_T.
    fold_mask = (T >= fold_low_T) & (T <= fold_high_T)
    if fold_mask.sum() >= 4:
        T_f, S_f, _ = _trim_by_convergence(
            T[fold_mask], S[fold_mask], side="cold",
            conv_tol=outlier_conv_tol, max_drop_frac=outlier_max_drop_frac,
        )
        if len(T_f) >= 2:
            fold_low_T = float(T_f[0])

    unf_mask = (T >= unf_low_T) & (T <= unf_high_T)
    if unf_mask.sum() >= 4:
        T_u, S_u, _ = _trim_by_convergence(
            T[unf_mask], S[unf_mask], side="hot",
            conv_tol=outlier_conv_tol, max_drop_frac=outlier_max_drop_frac,
        )
        if len(T_u) >= 2:
            unf_high_T = float(T_u[-1])

    return fold_low_T, fold_high_T, unf_low_T, unf_high_T


def _demo(csv_path: str) -> None:
    """Print detected baselines for each signal column of a RNAmelt CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    T = df.iloc[:, 0].to_numpy(dtype=float)
    print(f"# {csv_path}")
    print(f"# {len(T)} points, T in [{T.min():.2f}, {T.max():.2f}] C")
    print(f"# {'column':<24}  fold_low  fold_high    unf_low   unf_high")
    for col in df.columns[1:]:
        S = df[col].to_numpy(dtype=float)
        try:
            fl, fh, ul, uh = find_baselines(T, S)
            print(f"  {col:<24}  {fl:>8.2f}  {fh:>8.2f}   {ul:>8.2f}  {uh:>8.2f}")
        except Exception as e:
            print(f"  {col:<24}  ERROR: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m rnamelt.auto_baseline path/to/data.csv",
              file=sys.stderr)
        sys.exit(2)
    _demo(sys.argv[1])
