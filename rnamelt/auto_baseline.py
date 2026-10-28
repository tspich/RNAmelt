"""Automated baseline detection for melting curves.

    from rnamelt.auto_baseline import find_baselines
    fold_low_T, fold_high_T, unf_low_T, unf_high_T = find_baselines(T, signal)

``find_baselines`` returns four temperatures delimiting the two baseline
windows: the folded baseline is fit on data with
``fold_low_T <= T <= fold_high_T``, the unfolded baseline on data with
``unf_low_T <= T <= unf_high_T``, and the transition sits between
``fold_high_T`` and ``unf_low_T``. SingleCurve calls it when it is built
without windows.

Algorithm (signed second derivative + convergence trim)
-------------------------------------------------------
1. Transition direction from the sign of ``S[-n_edge:] - S[:n_edge]``.
2. ``dS/dT`` and ``d^2 S/dT^2`` (Savitzky-Golay by default).
3. Transition centre: the signed ``dS/dT`` extremum consistent with the
   direction.
Then, on each side of the centre (the high-T side is the low-T side of the
curve seen backwards, x = -T):
4. Shoulder: the signed ``d^2`` extremum closest to the centre.
5. Walk outward until ``d^2 * shoulder_sign`` drops below
   ``zero_tol_frac * |d^2_shoulder|``: the inner edge of the baseline window.
6. Artifact scan: a same-signed ``d^2`` excursion on the plateau larger than
   30% of the shoulder and than 3x the ``d^2`` noise (instrument warm-up,
   settling) moves the outer edge past it.
7. Trim: drop outer points while that meaningfully improves the RMS of a
   straight-line fit. A window narrower than ``min_baseline_width`` is
   trimmed over that width from its outer edge, but its inner edge stays:
   without a plateau the baseline is flat at the edge (zero-width window).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# A plateau excursion counts as an artifact above this fraction of the
# shoulder's |d^2| and above this many standard deviations of the d^2 noise;
# clean plateaus sit at the noise level, well below both.
_ARTIFACT_FRAC = 0.30
_ARTIFACT_NOISE_K = 3.0


def _trim_by_convergence(x: np.ndarray, S: np.ndarray, conv_tol: float = 0.05,
                         max_drop_frac: float = 0.33) -> np.ndarray:
    """Drop points from the start (outer edge) of a baseline window while the
    relative improvement of the straight-line RMS residual is >= conv_tol,
    at most max_drop_frac of the points. Returns the kept x values."""
    def rms(xa, Sa):
        m, b = np.polyfit(xa, Sa, 1)
        return float(np.sqrt(np.mean((Sa - (m * xa + b)) ** 2)))

    if len(x) < 4:
        return x
    current = rms(x, S)
    for _ in range(max(1, int(max_drop_frac * len(x)))):
        if len(x) < 4:
            break
        new = rms(x[1:], S[1:])
        if (current - new) / max(current, 1e-12) < conv_tol:
            break
        x, S, current = x[1:], S[1:], new
    return x


def _pick_shoulder(oriented: np.ndarray, prom_frac: float = 0.10) -> int:
    """Index of the shoulder: the prominent maximum of the sign-oriented d^2
    with the highest index, i.e. closest to the transition centre (edge
    artifacts sit at low index and are not peaks for find_peaks). Falls back
    to the global maximum on featureless slices."""
    from scipy.signal import find_peaks
    if oriented.size == 0:
        return 0
    peak_mag = float(np.abs(oriented).max())
    if peak_mag <= 0.0:
        return int(np.argmax(oriented))
    peaks, _ = find_peaks(oriented, prominence=prom_frac * peak_mag)
    return int(peaks[-1]) if len(peaks) else int(np.argmax(oriented))


def _side_window(x, S, d2, sign, peak_idx, d2_noise, zero_tol_frac, conv_tol,
                 max_drop_frac, min_width) -> Tuple[float, float]:
    """(outer, inner) edge of the baseline window on the low-x side of the
    transition (steps 4-7 of the module docstring). ``sign`` orients d^2 so
    the shoulder is a maximum."""
    shoulder = _pick_shoulder(sign * d2[: peak_idx + 1])
    s_sign = int(np.sign(d2[shoulder]))
    s_mag = abs(float(d2[shoulder]))

    boundary = next((i for i in range(shoulder - 1, -1, -1)
                     if d2[i] * s_sign < zero_tol_frac * s_mag), 0)

    outer = 0
    if boundary >= 4 and s_mag > 0.0:
        plateau = s_sign * d2[:boundary]
        art = int(np.argmax(plateau))
        if plateau[art] >= max(_ARTIFACT_FRAC * s_mag, _ARTIFACT_NOISE_K * d2_noise):
            past = next((i for i in range(art + 1, boundary)
                         if d2[i] * s_sign < zero_tol_frac * plateau[art]), art)
            if x[boundary] - x[past] >= min_width:
                outer = past

    lo, hi = float(x[outer]), float(x[boundary])
    trim_hi = max(hi, min(lo + min_width, float(x[peak_idx]) - 1.0))
    mask = (x >= lo) & (x <= trim_hi)
    if mask.sum() >= 4:
        kept = _trim_by_convergence(x[mask], S[mask], conv_tol, max_drop_frac)
        if len(kept) >= 2:
            lo = float(kept[0])
    return lo, max(lo, hi)


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
    """``(fold_low_T, fold_high_T, unf_low_T, unf_high_T)`` delimiting the
    folded and unfolded baseline windows (see the module docstring).

    Parameters
    ----------
    T, signal : arrays; T strictly ascending.
    smooth : if False, uses ``np.gradient`` for both derivatives.
    smooth_w1, smooth_w2, smooth_p : Savitzky-Golay windows for d1 / d2 and
        polynomial order.
    zero_tol_frac : the walk-out stops when d^2 has decayed to this fraction
        of the shoulder peak (0 = walk until d^2 changes sign).
    outlier_conv_tol : relative RMS improvement below which trimming stops.
    outlier_max_drop_frac : maximum fraction of a window that may be trimmed.
    min_baseline_width : plateau (°C) the artifact scan must leave, and the
        span from the outer edge the trim looks at in a narrower window.
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
        from scipy.signal import savgol_coeffs, savgol_filter
        w1 = smooth_w1 if smooth_w1 % 2 == 1 else smooth_w1 + 1
        w2 = smooth_w2 if smooth_w2 % 2 == 1 else smooth_w2 + 1
        p = min(smooth_p, min(w1, w2) - 1)
        if n < w2:
            raise ValueError(f"need at least {w2} points for smoothing; got {n}")
        d1 = savgol_filter(S, w1, p, deriv=1) / dt
        d2 = savgol_filter(S, w2, p, deriv=2) / (dt * dt)
        d2_gain = float(np.sqrt(np.sum(savgol_coeffs(w2, p, deriv=2) ** 2)))
    else:
        d1 = np.gradient(S, T)
        d2 = np.gradient(d1, T)
        d2_gain = np.sqrt(6.0) / 4.0    # (S[i+2] - 2 S[i] + S[i-2]) / 4
    # d^2 noise away from the scan edges: the signal's noise (robust sigma of
    # its second differences, which remove the smooth curve) times the gain of
    # the d^2 filter. The one-sided filters at the edges amplify noise further,
    # but instrument warm-up sits there, and an edge-scaled threshold misses it.
    dd = np.diff(S, 2)
    sigma_S = 1.4826 * float(np.median(np.abs(dd - np.median(dd)))) / np.sqrt(6.0)
    d2_noise = sigma_S * d2_gain / (dt * dt)

    peak_idx = int(np.argmax(d1) if direction_up else np.argmin(d1))
    # d^2 is positive on the cold side of a rising transition and negative on
    # the hot side; flipped for a falling one. Seen backwards (x = -T) the hot
    # side is a cold side with the opposite sign, and d^2 is unchanged.
    sign = 1.0 if direction_up else -1.0
    opts = (zero_tol_frac, outlier_conv_tol, outlier_max_drop_frac, min_baseline_width)
    fold_low_T, fold_high_T = _side_window(T, S, d2, sign, peak_idx, d2_noise, *opts)
    x_out, x_in = _side_window(-T[::-1], S[::-1], d2[::-1], -sign, n - 1 - peak_idx,
                               d2_noise, *opts)
    return fold_low_T, fold_high_T, -x_in, -x_out
