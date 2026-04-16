"""
03_wave_front.py — Replicate the Danko 2013 wave-front elongation rate method.

Strategy:
  1. For each time point T (10, 25, 40 min), compute difference map:
         diff(T, p) = coverage(E2_T, p) - coverage(Vehicle, p)
  2. Find the leading edge of the Pol II wave in diff(T) per gene.
  3. Fit linear regression: wave_front_position ~ time_minutes
  4. Slope = elongation rate in bp/min

Reference:
  Danko et al. 2013, Mol Cell 50:212-222. PMID: 23523369.
  "Pol II wave was identified using a 3-state HMM applied to difference maps
   in each biological replicate."

We implement both:
  (A) Simple threshold method (faster, less robust)
  (B) HMM-based method matching Danko's groHMM approach

Author: Simon (Herald AI) + M. Thornton
Date: 2026-04-04
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


# ─── Simple threshold method ────────────────────────────────────────────────

def find_wave_front_threshold(
    diff_map: np.ndarray,
    gene_start: int,
    gene_strand: str = "+",
    smooth_window: int = 5000,
    threshold_quantile: float = 0.75,
) -> Optional[int]:
    """
    Find wave-front position using a simple smoothed threshold on the difference map.

    Parameters
    ----------
    diff_map : np.ndarray, shape (M,)
        Difference in GRO-Seq coverage (treatment - vehicle) along gene body.
        Orientation: 5' → 3' (i.e. position 0 = TSS).
    gene_start : int
        Genomic coordinate of TSS (for reporting).
    gene_strand : str
        '+' or '-' (for orientation correction if diff_map not pre-oriented).
    smooth_window : int
        Smoothing window in bp for running mean before threshold.
    threshold_quantile : float
        Quantile of positive values to use as threshold.

    Returns
    -------
    wave_front_pos : int or None
        Estimated position of wave front in bp from TSS. None if not detected.
    """
    M = len(diff_map)

    # Smooth with running mean
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(diff_map, kernel, mode='same')
    else:
        smoothed = diff_map.copy()

    # Threshold: positive values only
    pos_vals = smoothed[smoothed > 0]
    if len(pos_vals) == 0:
        return None
    thresh = np.quantile(pos_vals, threshold_quantile)

    # Find the last position (furthest from TSS) where smoothed > threshold
    above = np.where(smoothed > thresh)[0]
    if len(above) == 0:
        return None

    wave_front = int(above[-1])
    return wave_front


# ─── 3-State HMM (simplified Danko approach) ────────────────────────────────

def viterbi_3state(obs: np.ndarray, params: dict) -> np.ndarray:
    """
    Viterbi decoding for a simple 3-state HMM matching Danko's approach.

    States:
      0 = Background (no wave signal)
      1 = Wave body (Pol II wave present, upstream of wave front)
      2 = Wave front (leading edge of wave)

    Transition matrix (log-space):
      P(0->0) = high, P(0->1) = low
      P(1->1) = high, P(1->2) = low
      P(2->2) = high, P(2->0) = low (wave passes)

    Emission: Gaussian for each state.
    """
    N = len(obs)
    K = 3  # states

    # Unpack parameters
    log_A = np.log(params["A"] + 1e-300)      # transition matrix K x K
    mu = params["mu"]                           # emission means K
    sigma = params["sigma"]                     # emission stds K
    log_pi = np.log(params["pi"] + 1e-300)     # initial state probs

    # Log emission probabilities
    def log_emit(k, o):
        return -0.5 * ((o - mu[k]) / sigma[k]) ** 2 - np.log(sigma[k] * np.sqrt(2 * np.pi))

    # Viterbi DP
    viterbi = np.full((K, N), -np.inf)
    backptr = np.zeros((K, N), dtype=int)

    for k in range(K):
        viterbi[k, 0] = log_pi[k] + log_emit(k, obs[0])

    for t in range(1, N):
        for k in range(K):
            scores = viterbi[:, t-1] + log_A[:, k]
            best = np.argmax(scores)
            backptr[k, t] = best
            viterbi[k, t] = scores[best] + log_emit(k, obs[t])

    # Backtrack
    states = np.zeros(N, dtype=int)
    states[-1] = np.argmax(viterbi[:, -1])
    for t in range(N-2, -1, -1):
        states[t] = backptr[states[t+1], t+1]

    return states


def find_wave_front_hmm(
    diff_map: np.ndarray,
    resolution_bp: int = 1000,
) -> Optional[int]:
    """
    Find wave-front position using 3-state HMM on binned difference map.

    Parameters
    ----------
    diff_map : np.ndarray
        Difference coverage (treatment - vehicle), 5'→3' orientation.
    resolution_bp : int
        Bin size for HMM (Danko used ~1 kb bins to reduce noise).

    Returns
    -------
    wave_front_pos : int or None
        Wave front position in bp from TSS. None if not detected.
    """
    M = len(diff_map)

    # Bin the difference map
    n_bins = M // resolution_bp
    if n_bins < 10:
        warnings.warn("Too few bins for HMM; try smaller resolution_bp or longer gene.")
        return None

    binned = np.array([
        np.mean(diff_map[i*resolution_bp:(i+1)*resolution_bp])
        for i in range(n_bins)
    ])

    # Estimate emission parameters from data
    # State 0 (background): values near 0
    # State 1 (wave body): positive values
    # State 2 (wave front): peak positive values (transition zone)
    bg_est = float(np.median(binned))
    pos_vals = binned[binned > bg_est]
    if len(pos_vals) < 3:
        # Not enough signal — fall back to threshold
        return find_wave_front_threshold(diff_map)

    wave_mu = float(np.mean(pos_vals))
    wave_sigma = float(np.std(pos_vals) + 1e-6)
    bg_sigma = float(np.std(binned[binned <= bg_est]) + 1e-6)

    params = {
        "pi": np.array([0.8, 0.15, 0.05]),
        "A": np.array([
            [0.98, 0.019, 0.001],   # from bg
            [0.001, 0.98, 0.019],   # from wave body
            [0.001, 0.001, 0.998],  # from wave front (absorbing)
        ]),
        "mu": np.array([bg_est, wave_mu * 0.5, wave_mu]),
        "sigma": np.array([bg_sigma, wave_sigma, wave_sigma * 0.5]),
    }

    states = viterbi_3state(binned, params)

    # Wave front = last position in state 1 (wave body) or state 2 (wave front)
    wave_positions = np.where((states == 1) | (states == 2))[0]
    if len(wave_positions) == 0:
        return None

    wave_front_bin = int(wave_positions[-1])
    wave_front_bp = wave_front_bin * resolution_bp + resolution_bp // 2

    return wave_front_bp


# ─── Elongation rate from multiple time points ───────────────────────────────

def estimate_elongation_rate(
    wave_fronts: Dict[int, int],
    min_points: int = 2,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate elongation rate by linear regression of wave front position vs time.

    Parameters
    ----------
    wave_fronts : dict
        {time_minutes: wave_front_position_bp_from_TSS}
        E.g. {10: 20000, 25: 50000, 40: 80000}
    min_points : int
        Minimum number of valid time points required.

    Returns
    -------
    rate_bp_per_min : float or None
        Elongation rate in bp/min.
    r_squared : float or None
        R^2 of the linear fit.
    """
    valid = {t: p for t, p in wave_fronts.items() if p is not None}
    if len(valid) < min_points:
        return None, None

    times = np.array(sorted(valid.keys()), dtype=float)
    positions = np.array([valid[t] for t in times], dtype=float)

    # Linear regression: position = rate * time + intercept
    A = np.vstack([times, np.ones(len(times))]).T
    result = np.linalg.lstsq(A, positions, rcond=None)
    rate, intercept = result[0]

    # R^2
    y_pred = rate * times + intercept
    ss_res = np.sum((positions - y_pred) ** 2)
    ss_tot = np.sum((positions - np.mean(positions)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(rate), float(r2)


# ─── Main pipeline ───────────────────────────────────────────────────────────

def compute_gene_elongation_rate(
    vehicle_coverage: np.ndarray,
    treatment_coverages: Dict[int, np.ndarray],
    method: str = "hmm",
    hmm_resolution_bp: int = 1000,
) -> dict:
    """
    Full pipeline: compute elongation rate for one gene.

    Parameters
    ----------
    vehicle_coverage : np.ndarray
        GRO-Seq coverage (1 bp res) for vehicle/control condition.
    treatment_coverages : dict
        {time_minutes: coverage_array} for each treatment time point.
    method : str
        'hmm' or 'threshold'.
    hmm_resolution_bp : int
        Binning resolution for HMM method.

    Returns
    -------
    result : dict
        wave_fronts: dict of {time: position}
        rate_bp_per_min: float
        rate_kb_per_min: float
        r_squared: float
    """
    wave_fronts = {}

    for t_min, treatment_cov in treatment_coverages.items():
        diff_map = treatment_cov - vehicle_coverage

        if method == "hmm":
            wf = find_wave_front_hmm(diff_map, resolution_bp=hmm_resolution_bp)
        else:
            wf = find_wave_front_threshold(diff_map)

        wave_fronts[t_min] = wf

    rate_bp_min, r2 = estimate_elongation_rate(wave_fronts)
    rate_kb_min = rate_bp_min / 1000.0 if rate_bp_min is not None else None

    return {
        "wave_fronts_bp": wave_fronts,
        "rate_bp_per_min": rate_bp_min,
        "rate_kb_per_min": rate_kb_min,
        "r_squared": r2,
    }


if __name__ == "__main__":
    # Synthetic test
    np.random.seed(0)
    M = 100_000  # 100 kb gene
    true_rate = 2500  # bp/min (2.5 kb/min)

    vehicle = np.random.exponential(2, M)

    treatment_coverages = {}
    for t in [10, 25, 40]:
        wave_pos = true_rate * t  # expected wave front
        wave_width = 5000
        treatment = vehicle.copy()
        # Add wave signal: boost coverage up to wave_pos
        wave_start = max(0, wave_pos - wave_width)
        wave_end = min(M, wave_pos + wave_width)
        treatment[int(wave_start):int(wave_end)] += np.random.exponential(5, int(wave_end - wave_start))
        treatment_coverages[t] = treatment

    result = compute_gene_elongation_rate(
        vehicle_coverage=vehicle,
        treatment_coverages=treatment_coverages,
        method="hmm",
        hmm_resolution_bp=1000,
    )

    print("Wave Front Analysis — Synthetic 100kb Gene")
    print(f"  True rate: {true_rate} bp/min ({true_rate/1000:.1f} kb/min)")
    print(f"  Wave fronts: {result['wave_fronts_bp']}")
    print(f"  Estimated rate: {result['rate_bp_per_min']:.0f} bp/min "
          f"({result['rate_kb_per_min']:.2f} kb/min)")
    print(f"  R^2: {result['r_squared']:.3f}")
