"""
AD Estimator — Algebraic Diversity spectral analysis of GRO-Seq coverage profiles.

Applies the cyclic group Z_M group-averaged estimator to a single gene's
GRO-Seq coverage vector x ∈ R^M (1 bp resolution), computing:
  - Full DFT power spectrum (periodogram = eigenvalues of R_hat under Z_M)
  - Spectral concentration ψ
  - Dominant frequency f*
  - Harmonic mean instantaneous speed (v_hat ∝ 1/x)
  - Spectral entropy

Author: Simon (Herald AI) + M. Thornton
Date: 2026-04-04
"""

import numpy as np
from typing import Tuple


def group_averaged_estimator_cyclic(x: np.ndarray) -> np.ndarray:
    """
    Compute the cyclic group Z_M group-averaged outer product estimator.

    Under Z_M, this is equivalent to the DFT periodogram.
    The eigenvalues of R_hat are |X[k]|^2 / M.

    Parameters
    ----------
    x : np.ndarray, shape (M,)
        Single observation vector (e.g. GRO-Seq coverage at 1 bp resolution).

    Returns
    -------
    eigenvalues : np.ndarray, shape (M,)
        Periodogram ordinates |X[k]|^2 / M. These are the eigenvalues of R_hat.
        Sorted in descending order.
    """
    M = len(x)
    X = np.fft.fft(x)
    periodogram = (np.abs(X) ** 2) / M
    return np.sort(periodogram)[::-1]


def spectral_concentration(eigenvalues: np.ndarray) -> float:
    """
    Compute spectral concentration ψ = (λ_max - λ_min) / trace(R_hat).

    ψ ∈ [0, 1]. High ψ → structured signal. Low ψ → flat spectrum (noise-like).

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of the group-averaged estimator (from group_averaged_estimator_cyclic).

    Returns
    -------
    psi : float
        Spectral concentration ψ.
    """
    lam_max = eigenvalues[0]
    lam_min = eigenvalues[-1]
    trace = np.sum(eigenvalues)
    if trace == 0:
        return 0.0
    return float((lam_max - lam_min) / trace)


def dominant_frequency(x: np.ndarray, bp_resolution: int = 1) -> Tuple[float, float]:
    """
    Find the dominant frequency (excluding DC) in the coverage spectrum.

    Parameters
    ----------
    x : np.ndarray, shape (M,)
        Coverage vector.
    bp_resolution : int
        Base pairs per sample (1 for 1 bp resolution).

    Returns
    -------
    freq_per_bp : float
        Dominant frequency in cycles per bp.
    period_bp : float
        Corresponding period in bp (e.g. ~200 bp for nucleosomal periodicity).
    """
    M = len(x)
    X = np.fft.fft(x)
    power = np.abs(X) ** 2
    # Exclude DC (k=0) and find max
    freqs = np.fft.fftfreq(M, d=bp_resolution)
    # Only look at positive frequencies
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = power[pos_mask]
    if len(pos_power) == 0:
        return 0.0, np.inf
    idx_max = np.argmax(pos_power)
    freq_star = pos_freqs[idx_max]
    period = 1.0 / freq_star if freq_star > 0 else np.inf
    return float(freq_star), float(period)


def harmonic_mean_speed(x: np.ndarray, pseudo: float = 1e-3) -> float:
    """
    Estimate per-gene mean elongation speed using harmonic mean of coverage-inverse.

    The harmonic mean is the correct average for speeds derived as v ∝ 1/density,
    because:
        mean(v) = mean(1/x) → harmonic mean of x (appropriately weighted)

    The harmonic mean of speeds = 1 / mean(1/v) = 1 / mean(x/C) = C / mean(x)
    But since we want proportional estimate, we return: 1 / mean(x + pseudo)

    Parameters
    ----------
    x : np.ndarray
        Coverage vector (RPM or raw counts). Must be non-negative.
    pseudo : float
        Pseudocount added before inversion to avoid division by zero.

    Returns
    -------
    speed_proxy : float
        Proportional estimate of mean elongation speed. Higher = faster.
    """
    # Harmonic mean of (x + pseudo): HM = n / sum(1/(x_i + pseudo))
    # But "speed" ∝ 1/x, so mean speed ∝ mean(1/x) → harmonic of x is not same
    # Correct: speed_proxy = 1 / mean(x + pseudo)  [arithmetic mean of v ∝ 1/x]
    # For per-gene summary: arithmetic mean of instantaneous speeds
    inv_speeds = 1.0 / (x + pseudo)
    mean_speed = float(np.mean(inv_speeds))
    return mean_speed


def instantaneous_speed_profile(x: np.ndarray, pseudo: float = 1e-3) -> np.ndarray:
    """
    Compute instantaneous speed estimate at every position.

    v_hat(p) = 1 / (x(p) + pseudo)   [up to unknown constant C]

    Parameters
    ----------
    x : np.ndarray
        Coverage vector.
    pseudo : float
        Pseudocount.

    Returns
    -------
    v_hat : np.ndarray, same shape as x
        Instantaneous speed proxy (proportional, not in kb/min without calibration).
    """
    return 1.0 / (x + pseudo)


def spectral_entropy(x: np.ndarray) -> float:
    """
    Compute normalized spectral entropy of the coverage profile.

    H = -sum(p_k * log(p_k)) / log(M)  where p_k = |X[k]|^2 / sum(|X|^2)

    H ∈ [0, 1]. High H → flat spectrum (uniform elongation). Low H → structured.

    Parameters
    ----------
    x : np.ndarray
        Coverage vector.

    Returns
    -------
    H : float
        Normalized spectral entropy.
    """
    M = len(x)
    X = np.fft.fft(x)
    power = np.abs(X) ** 2
    total = np.sum(power)
    if total == 0:
        return 1.0
    p = power / total
    # Avoid log(0)
    p = p[p > 0]
    H = -np.sum(p * np.log(p)) / np.log(M)
    return float(H)


def analyze_gene_coverage(
    x: np.ndarray,
    gene_id: str = "",
    bp_resolution: int = 1,
    pseudo: float = 1e-3
) -> dict:
    """
    Full AD analysis pipeline for a single gene's GRO-Seq coverage vector.

    Parameters
    ----------
    x : np.ndarray, shape (M,)
        Coverage vector (e.g. RPM at 1 bp resolution over gene body).
    gene_id : str
        Gene identifier (for logging).
    bp_resolution : int
        Base pairs per position.
    pseudo : float
        Pseudocount for inversion stability.

    Returns
    -------
    results : dict
        Dictionary containing all computed AD statistics.
    """
    M = len(x)

    # AD spectral estimation (Z_M group)
    eigenvalues = group_averaged_estimator_cyclic(x)
    psi = spectral_concentration(eigenvalues)
    f_star, period_star = dominant_frequency(x, bp_resolution)
    H = spectral_entropy(x)

    # Speed estimates
    v_hat = instantaneous_speed_profile(x, pseudo)
    mean_speed = float(np.mean(v_hat))
    median_speed = float(np.median(v_hat))

    # Coverage statistics
    mean_cov = float(np.mean(x))
    max_cov = float(np.max(x))
    cv_cov = float(np.std(x) / (mean_cov + pseudo))  # coefficient of variation

    return {
        "gene_id": gene_id,
        "M": M,
        "psi": psi,
        "spectral_entropy": H,
        "dominant_freq_per_bp": f_star,
        "dominant_period_bp": period_star,
        "lambda_max": float(eigenvalues[0]),
        "lambda_min": float(eigenvalues[-1]),
        "trace_Rhat": float(np.sum(eigenvalues)),
        "mean_speed_proxy": mean_speed,
        "median_speed_proxy": median_speed,
        "mean_coverage_rpm": mean_cov,
        "max_coverage_rpm": max_cov,
        "cv_coverage": cv_cov,
    }


if __name__ == "__main__":
    # Quick sanity check on synthetic data
    np.random.seed(42)
    M = 50000  # 50 kb gene at 1 bp resolution

    # Simulate GRO-Seq coverage: decreasing gradient (RNAPII slows near TSS, speeds up)
    # with superimposed ~200 bp nucleosomal periodicity
    pos = np.arange(M)
    base = 10 * np.exp(-pos / 20000) + 2  # background + promoter peak
    nucleosomal = 0.5 * np.cos(2 * np.pi * pos / 200)  # 200 bp periodicity
    noise = np.random.exponential(0.5, M)
    x = base + nucleosomal + noise
    x = np.clip(x, 0, None)

    results = analyze_gene_coverage(x, gene_id="TEST_GENE_50kb", bp_resolution=1)

    print("AD Analysis Results — Synthetic 50kb Gene")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.6f}")
        else:
            print(f"  {k:30s}: {v}")

    # Processing gain
    gain_dB = 10 * np.log10(M)
    print(f"\n  Processing gain (10*log10(M))  : {gain_dB:.1f} dB")
    print(f"  (Equivalent to {M:,} independent observations)")
