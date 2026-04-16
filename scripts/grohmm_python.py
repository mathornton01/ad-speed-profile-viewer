"""
groHMM polymeraseWave — Python port
====================================
Faithful port of the Danko lab groHMM R/C package polymeraseWave() function.
Source: https://github.com/Kraus-Lab/groHMM

3-state left-to-right HMM for detecting the RNA Pol II elongation wave front
in GRO-Seq difference maps (treatment minus vehicle).

States:
  0 — upstream / intergenic  → Normal emission
  1 — wave region            → Gamma emission (shifted to be positive)
  2 — downstream of wave     → Gamma emission

Reference:
  Danko et al. (2013) Molecular Cell 50(2):212-222
  groHMM: Danko et al. (2015) BMC Bioinformatics 16:221

Author: Simon (AD Experiments, 2026-04-04)
"""

import numpy as np
from scipy import stats
from scipy.special import digamma, polygamma, logsumexp
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# MLE for Gamma distribution (Newton's method on digamma, port of MLEfit.c)
# ---------------------------------------------------------------------------

def mle_gamma(x):
    """MLE of Gamma(shape, scale) via Newton's method."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 2:
        return 1.0, 1.0  # fallback
    N = len(x)
    mean_x = np.mean(x)
    mean_log_x = np.mean(np.log(x))
    s = np.log(mean_x) - mean_log_x  # always >= 0
    if s <= 0:
        s = 1e-10
    # initial estimate (Choi & Wette 1969)
    shape = (3 - s + np.sqrt((s - 3)**2 + 24 * s)) / (12 * s)
    # Newton's method (max 50 iterations)
    for _ in range(50):
        f = np.log(shape) - digamma(shape) - s
        df = 1.0 / shape - polygamma(1, shape)
        if df == 0:
            break
        shape_new = shape - f / df
        if shape_new <= 0:
            shape_new = shape / 2
        if abs(shape_new - shape) < 1e-8 * abs(shape):
            shape = shape_new
            break
        shape = shape_new
    scale = mean_x / shape
    return float(shape), float(scale)


def mle_normal(x):
    """MLE of Normal(mean, sd)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0, 1.0
    return float(np.mean(x)), float(np.std(x) + 1e-10)


# ---------------------------------------------------------------------------
# Log-emission probabilities (bin-integrated CDFs, port of hmmHeader.h)
# ---------------------------------------------------------------------------

def log_emit_normal(obs, mean, sd):
    """Log P(obs | Normal(mean, sd)) using bin-integrated CDF."""
    obs = np.asarray(obs, dtype=float)
    out = np.where(
        np.isfinite(obs),
        np.log(
            np.maximum(
                stats.norm.cdf(obs + 0.5, mean, sd) - stats.norm.cdf(obs - 0.5, mean, sd),
                1e-300
            )
        ),
        0.0  # NaN obs → neutral log-prob
    )
    return out


def log_emit_gamma(obs, shape, scale):
    """Log P(obs | Gamma(shape, scale)) using bin-integrated CDF."""
    obs = np.asarray(obs, dtype=float)
    out = np.where(
        np.isfinite(obs) & (obs > 0),
        np.log(
            np.maximum(
                stats.gamma.cdf(obs + 0.5, a=shape, scale=scale) -
                stats.gamma.cdf(np.maximum(obs - 0.5, 1e-10), a=shape, scale=scale),
                1e-300
            )
        ),
        0.0
    )
    return out


# ---------------------------------------------------------------------------
# Forward-Backward and Baum-Welch EM
# ---------------------------------------------------------------------------

def forward_backward(log_emit, log_trans, log_init):
    """
    Vectorized forward-backward in log-space. O(T*K²) but with numpy ops.

    log_emit : (T, K) log P(obs_t | state k)
    log_trans: (K, K) log A[i,j] = log P(state j | state i)
    log_init : (K,)   log P(state 0 = k)

    Returns:
      log_alpha, log_beta, log_gamma, log_xi, log_likelihood
    """
    T, K = log_emit.shape

    # Forward pass (vectorized over states)
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_init + log_emit[0]
    for t in range(1, T):
        # log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + log_trans[i, j]) + log_emit[t, j]
        log_alpha[t] = logsumexp(
            log_alpha[t-1, :, None] + log_trans, axis=0   # (K,) broadcast over (K,K)
        ) + log_emit[t]

    log_likelihood = logsumexp(log_alpha[-1])

    # Backward pass (vectorized over states)
    log_beta = np.full((T, K), -np.inf)
    log_beta[-1] = 0.0
    for t in range(T-2, -1, -1):
        # log_beta[t, i] = logsumexp_j(log_trans[i,j] + log_emit[t+1,j] + log_beta[t+1,j])
        log_beta[t] = logsumexp(
            log_trans + log_emit[t+1] + log_beta[t+1], axis=1
        )

    # State posteriors γ(t, k) = α(t,k) + β(t,k) - log P(obs)
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

    # Pairwise posteriors ξ(t,i,j) — fully vectorized
    # Shape: (T-1, K, K)
    log_xi = (log_alpha[:-1, :, None] +   # (T-1, K, 1)
              log_trans[None, :, :] +       # (1,   K, K)
              log_emit[1:, None, :] +       # (T-1, 1, K)
              log_beta[1:, None, :])        # (T-1, 1, K)
    log_xi -= logsumexp(log_xi.reshape(T-1, -1), axis=1)[:, None, None]

    return log_alpha, log_beta, log_gamma, log_xi, log_likelihood


def viterbi(log_emit, log_trans, log_init):
    """Vectorized Viterbi decoding; returns integer state sequence."""
    T, K = log_emit.shape
    delta = np.full((T, K), -np.inf)
    psi   = np.zeros((T, K), dtype=int)
    delta[0] = log_init + log_emit[0]
    for t in range(1, T):
        # scores[i, j] = delta[t-1, i] + log_trans[i, j]
        scores = delta[t-1, :, None] + log_trans   # (K, K)
        psi[t]   = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + log_emit[t]
    # Backtrace
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path


def update_params_left_to_right(log_gamma, log_xi):
    """
    Re-estimate transition matrix from ξ.
    Enforces left-to-right structure (no back-transitions).
    State 2 is absorbing.
    """
    K = log_gamma.shape[1]
    log_trans_new = np.full((K, K), -np.inf)
    for i in range(K):
        denom = np.logaddexp.reduce(log_xi[:, i, :].ravel())
        for j in range(K):
            numer = np.logaddexp.reduce(log_xi[:, i, j])
            log_trans_new[i, j] = numer - denom
    # Enforce left-to-right: zero out back-transitions
    for i in range(K):
        for j in range(i):
            log_trans_new[i, j] = -np.inf
    # Normalize rows
    for i in range(K):
        total = np.logaddexp.reduce(log_trans_new[i, np.isfinite(log_trans_new[i])])
        log_trans_new[i, np.isfinite(log_trans_new[i])] -= total
    return log_trans_new


# ---------------------------------------------------------------------------
# Main polymeraseWave function (faithful port)
# ---------------------------------------------------------------------------

def polymerase_wave(
    coverage_cond,          # 1D array: reads per window at stimulated time point
    coverage_vehicle,       # 1D array: reads per window at vehicle (0 min)
    gene_start,             # genomic start of analysis window (bp)
    gene_end,               # genomic end of analysis window (bp)
    window_size=50,         # bp per window (size parameter in groHMM)
    approx_dist=None,       # expected wave distance in bp; if None, use 2000 * time_min
    time_min=40,            # minutes post-stimulation (used if approx_dist is None)
    upstream_dist=10000,    # bp upstream of TSS to include
    emission="gamma",       # "gamma" | "norm"
    max_iter=10,            # Baum-Welch iterations (groHMM hardcodes 10)
    tol=0.01,               # log-likelihood convergence threshold
    verbose=False,
):
    """
    Detect the RNA Pol II elongation wave front using a 3-state left-to-right HMM.

    Returns dict with keys:
      wave_start_bp    : position (bp from gene_start) where state 0→1 transition occurs
      wave_end_bp      : position (bp from gene_start) where state 1→2 transition occurs
                         = wave FRONT = elongation distance at this time point
      wave_length_bp   : wave_end_bp - wave_start_bp
      viterbi_path     : integer array of decoded states (0/1/2) per window
      log_likelihood   : final log-likelihood after EM
      converged        : bool
      params           : dict with fitted emission parameters
    """
    cond = np.asarray(coverage_cond, dtype=float)
    veh  = np.asarray(coverage_vehicle, dtype=float)

    # Ensure same length
    n = min(len(cond), len(veh))
    cond = cond[:n]
    veh  = veh[:n]

    # Difference signal
    gene = cond - veh

    # Gamma shift: make all values >= 1 (required for Gamma emission)
    if emission == "gamma":
        gene = gene - np.nanmin(gene) + 1.0

    T = len(gene)

    # Window breakpoints (in units of windows)
    u_trans = int(np.ceil((upstream_dist - 5000) / window_size))  # end of upstream
    u_trans = max(1, min(u_trans, T - 2))

    if approx_dist is None:
        approx_dist = 2000.0 * time_min  # bp/min default from Danko

    i_trans = int(np.ceil((upstream_dist + approx_dist) / window_size))  # end of wave
    i_trans = max(u_trans + 1, min(i_trans, T - 1))

    if verbose:
        print(f"T={T} windows, u_trans={u_trans}, i_trans={i_trans}")

    # --- Initialize emission parameters by splitting signal at breakpoints ---
    upstream_seg  = gene[:u_trans]
    wave_seg      = gene[u_trans:i_trans]
    down_seg      = gene[i_trans:]

    if len(wave_seg) < 2:
        wave_seg = gene[u_trans:u_trans+2]
    if len(down_seg) < 2:
        down_seg = gene[-2:]

    if emission == "gamma":
        shape0, scale0 = mle_gamma(upstream_seg)  # will use Normal for state 0 anyway
        mean0, sd0     = mle_normal(upstream_seg)
        shape1, scale1 = mle_gamma(wave_seg)
        shape2, scale2 = mle_gamma(down_seg)
        params = {
            "upstream": {"mean": mean0, "sd": sd0},
            "wave":     {"shape": shape1, "scale": scale1},
            "downstream": {"shape": shape2, "scale": scale2},
        }
    else:  # norm
        mean0, sd0 = mle_normal(upstream_seg)
        mean1, sd1 = mle_normal(wave_seg)
        mean2, sd2 = mle_normal(down_seg)
        params = {
            "upstream":   {"mean": mean0, "sd": sd0},
            "wave":       {"mean": mean1, "sd": sd1},
            "downstream": {"mean": mean2, "sd": sd2},
        }

    # --- Initialize transition matrix (left-to-right, geometric durations) ---
    p01 = 1.0 / u_trans               # upstream → wave
    p12 = 1.0 / max(1, i_trans - u_trans)  # wave → downstream

    log_trans = np.array([
        [np.log(1 - p01),       np.log(p01),       -np.inf],
        [-np.inf,               np.log(1 - p12),   np.log(p12)],
        [-np.inf,               -np.inf,            0.0],
    ])
    log_init = np.array([0.0, -np.inf, -np.inf])  # always start in state 0

    # --- Baum-Welch EM ---
    prev_ll = -np.inf
    converged = False

    for iteration in range(max_iter):
        # Compute log-emission matrix
        if emission == "gamma":
            e0 = log_emit_normal(gene, params["upstream"]["mean"], params["upstream"]["sd"])
            e1 = log_emit_gamma(gene,  params["wave"]["shape"],    params["wave"]["scale"])
            e2 = log_emit_gamma(gene,  params["downstream"]["shape"], params["downstream"]["scale"])
        else:
            e0 = log_emit_normal(gene, params["upstream"]["mean"], params["upstream"]["sd"])
            e1 = log_emit_normal(gene, params["wave"]["mean"],     params["wave"]["sd"])
            e2 = log_emit_normal(gene, params["downstream"]["mean"], params["downstream"]["sd"])

        log_emit_mat = np.column_stack([e0, e1, e2])

        # Forward-backward
        log_alpha, log_beta, log_gamma, log_xi, ll = forward_backward(
            log_emit_mat, log_trans, log_init
        )

        if verbose:
            print(f"  iter {iteration+1}: log-likelihood = {ll:.4f}")

        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

        # M-step: update transition matrix
        log_trans = update_params_left_to_right(log_gamma, log_xi)

        # M-step: update emission parameters (weighted MLE)
        gamma_exp = np.exp(log_gamma)  # (T, 3) posterior state occupancy

        def weighted_normal_mle(obs, weights):
            w = np.maximum(weights, 1e-300)
            w_sum = w.sum()
            mean = np.sum(w * obs) / w_sum
            sd = np.sqrt(np.sum(w * (obs - mean)**2) / w_sum) + 1e-10
            return float(mean), float(sd)

        def weighted_gamma_mle(obs, weights):
            """Approximate weighted MLE for Gamma using method of moments."""
            w = np.maximum(weights, 1e-300)
            valid = np.isfinite(obs) & (obs > 0)
            obs_v = obs[valid]
            w_v = w[valid]
            if len(obs_v) < 2 or w_v.sum() < 1e-10:
                return 1.0, 1.0
            w_v /= w_v.sum()
            mean_x = np.sum(w_v * obs_v)
            mean_log_x = np.sum(w_v * np.log(obs_v))
            s = np.log(mean_x) - mean_log_x
            s = max(s, 1e-10)
            shape = (3 - s + np.sqrt((s - 3)**2 + 24 * s)) / (12 * s)
            # One Newton step
            f = np.log(shape) - digamma(shape) - s
            df = 1.0 / shape - polygamma(1, shape)
            if df != 0:
                shape = max(shape - f / df, 1e-10)
            scale = mean_x / shape
            return float(shape), float(scale)

        if emission == "gamma":
            m, s = weighted_normal_mle(gene, gamma_exp[:, 0])
            params["upstream"]["mean"] = m
            params["upstream"]["sd"]   = s
            sh, sc = weighted_gamma_mle(gene, gamma_exp[:, 1])
            params["wave"]["shape"] = sh
            params["wave"]["scale"] = sc
            sh, sc = weighted_gamma_mle(gene, gamma_exp[:, 2])
            params["downstream"]["shape"] = sh
            params["downstream"]["scale"] = sc
        else:
            for state_idx, key in enumerate(["upstream", "wave", "downstream"]):
                m, s = weighted_normal_mle(gene, gamma_exp[:, state_idx])
                params[key]["mean"] = m
                params[key]["sd"]   = s

    # --- Final Viterbi decoding ---
    if emission == "gamma":
        e0 = log_emit_normal(gene, params["upstream"]["mean"], params["upstream"]["sd"])
        e1 = log_emit_gamma(gene,  params["wave"]["shape"],    params["wave"]["scale"])
        e2 = log_emit_gamma(gene,  params["downstream"]["shape"], params["downstream"]["scale"])
    else:
        e0 = log_emit_normal(gene, params["upstream"]["mean"], params["upstream"]["sd"])
        e1 = log_emit_normal(gene, params["wave"]["mean"],     params["wave"]["sd"])
        e2 = log_emit_normal(gene, params["downstream"]["mean"], params["downstream"]["sd"])

    log_emit_mat = np.column_stack([e0, e1, e2])
    path = viterbi(log_emit_mat, log_trans, log_init)

    # Extract wave boundaries (groHMM: max(which(state==0)) and max(which(state==1)))
    state0_positions = np.where(path == 0)[0]
    state1_positions = np.where(path == 1)[0]

    if len(state0_positions) > 0:
        DTs = int(state0_positions[-1])
    else:
        DTs = 0

    if len(state1_positions) > 0:
        DTe = int(state1_positions[-1])
    else:
        DTe = DTs + 1

    wave_start_bp  = DTs * window_size
    wave_end_bp    = DTe * window_size
    wave_length_bp = (DTe - DTs) * window_size

    return {
        "wave_start_bp":  wave_start_bp,
        "wave_end_bp":    wave_end_bp,
        "wave_length_bp": wave_length_bp,
        "viterbi_path":   path,
        "log_likelihood": prev_ll,
        "converged":      converged,
        "params":         params,
        "DTs_window":     DTs,
        "DTe_window":     DTe,
    }


# ---------------------------------------------------------------------------
# Genome-wide wave-front rate estimation (linear regression over time points)
# ---------------------------------------------------------------------------

def estimate_elongation_rate(wave_positions_by_time):
    """
    Fit linear regression: wave_position (bp) ~ time (min) → slope = rate (bp/min).

    wave_positions_by_time: dict {time_min: wave_end_bp} or list of (time, position) tuples

    Returns:
      rate_bp_per_min : float
      r_squared       : float
      intercept       : float
    """
    if isinstance(wave_positions_by_time, dict):
        times = np.array(sorted(wave_positions_by_time.keys()), dtype=float)
        positions = np.array([wave_positions_by_time[t] for t in times], dtype=float)
    else:
        times, positions = zip(*wave_positions_by_time)
        times     = np.array(times, dtype=float)
        positions = np.array(positions, dtype=float)

    if len(times) < 2:
        return np.nan, np.nan, np.nan

    slope, intercept, r, p, se = stats.linregress(times, positions)
    return float(slope), float(r**2), float(intercept)


# ---------------------------------------------------------------------------
# Window-based coverage aggregation (port of windowAnalysis)
# ---------------------------------------------------------------------------

def window_coverage(coverage_1bp, window_size=50):
    """
    Aggregate 1 bp coverage array into non-overlapping windows.
    Returns array of length ceil(len(coverage_1bp) / window_size).
    """
    n = len(coverage_1bp)
    n_windows = int(np.ceil(n / window_size))
    result = np.zeros(n_windows)
    for i in range(n_windows):
        start = i * window_size
        end   = min(start + window_size, n)
        result[i] = np.sum(coverage_1bp[start:end])
    return result


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    T_bp = 150_000  # 150 kb gene
    wp = 50         # window size

    # Simulate a wave at 80 kb
    wave_pos_bp = 80_000
    pos = np.arange(T_bp)
    wave_cov = np.where(pos < wave_pos_bp, 30.0, 5.0) + np.random.exponential(3, T_bp)
    veh_cov  = np.full(T_bp, 5.0) + np.random.exponential(2, T_bp)

    w_cond = window_coverage(wave_cov, wp)
    w_veh  = window_coverage(veh_cov, wp)

    result = polymerase_wave(
        w_cond, w_veh,
        gene_start=0, gene_end=T_bp,
        window_size=wp,
        approx_dist=70_000,
        upstream_dist=10_000,
        verbose=True,
    )

    print(f"\n--- Smoke test result ---")
    print(f"  True wave front: {wave_pos_bp:,} bp")
    print(f"  Detected start:  {result['wave_start_bp']:,} bp")
    print(f"  Detected end:    {result['wave_end_bp']:,} bp")
    print(f"  Wave length:     {result['wave_length_bp']:,} bp")
    print(f"  Converged:       {result['converged']}")
    print(f"  Final LL:        {result['log_likelihood']:.2f}")
    print(f"  Error:           {abs(result['wave_end_bp'] - wave_pos_bp):,} bp")
