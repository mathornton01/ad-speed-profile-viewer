#!/usr/bin/env python3
"""
pipeline.py - Full GRO-Seq → AD-enhanced instantaneous speed pipeline.

Strategy (from STRATEGY_LOG.md):
  Phase 1: Replicate Danko wave-front rates as baseline.
  Phase 2: Compute 1/density instantaneous speed at each position.
  Phase 3: Apply AD (cyclic group Z_m) to reduce noise across replicates.
  Phase 4: Benchmark all methods vs Danko regression rates.

Timepoints: 0m (vehicle, 3 reps), 10m (3 reps), 25m (2 reps), 40m (3 reps)
Ground truth: GSE41324 regression rate TSV files (Danko et al. 2013, Mol Cell).
"""

import numpy as np
import pandas as pd
import gzip, os, sys, pickle
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE    = Path("/home/mathornton/herald-workspace/simon/Projects/AD Experiments/read accumulation rate experiments")
RAW     = BASE / "data" / "raw"
PROC    = BASE / "data" / "processed"
RATES   = BASE / "data" / "danko_rates"
RESULTS = BASE / "results"
PROC.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SAMPLE_FILES = {
    "0m_R1":  "GSM1014637_E2.0m_R1.noAdapt.bed.gz",
    "0m_R2":  "GSM1014638_E2.0m_R2.noAdapt.bed.gz",
    "0m_R3":  "GSM1014639_E2.0m_R3.noAdapt.bed.gz",
    "10m_R1": "GSM1014640_E2.10m_R1.noAdapt.bed.gz",
    "10m_R2": "GSM1014641_E2.10m_R2.noAdapt.bed.gz",
    "10m_R3": "GSM1014642_E2.10m_R3.noAdapt.bed.gz",
    "25m_R1": "GSM1014643_E2.25m_R1.noAdapt.bed.gz",
    "25m_R3": "GSM1014644_E2.25m_R3.noAdapt.bed.gz",
    "40m_R1": "GSM1014645_E2.40m_R1.noAdapt.bed.gz",
    "40m_R2": "GSM1014646_E2.40m_R2.noAdapt.bed.gz",
    "40m_R3": "GSM1014647_E2.40m_R3.noAdapt.bed.gz",
}

TIMEPOINTS = {
    "0m":  ["0m_R1", "0m_R2", "0m_R3"],
    "10m": ["10m_R1", "10m_R2", "10m_R3"],
    "25m": ["25m_R1", "25m_R3"],
    "40m": ["40m_R1", "40m_R2", "40m_R3"],
}

TP_MINUTES = {"0m": 0, "10m": 10, "25m": 25, "40m": 40}


# ──────────────────────────────────────────────
# 1. Load ground-truth rates
# ──────────────────────────────────────────────
def load_danko_rates():
    """Load and merge all three Danko rate files. Use 10-40m as primary."""
    df = pd.read_csv(RATES / "MCF7.10-40m.regressionRate.tsv", sep="\t")
    # add source label
    df["source"] = "10-40m"
    df25 = pd.read_csv(RATES / "MCF7.25m.regressionRate.tsv", sep="\t")
    df25["source"] = "25m"
    df40 = pd.read_csv(RATES / "MCF7.40m.regressionRate.tsv", sep="\t")
    df40["source"] = "40m"
    # Union: keep 10-40m first (most reliable, uses 3 timepoints), fill with others
    all_ids = df.uniqueID.tolist()
    extra25 = df25[~df25.uniqueID.isin(all_ids)][["chrom","chromStart","chromEnd","uniqueID","rate","strand","source"]]
    df40_sub = df40[~df40.uniqueID.isin(all_ids + extra25.uniqueID.tolist())][["chrom","chromStart","chromEnd","uniqueID","rate","strand","source"]]
    df_merged = pd.concat([
        df[["chrom","chromStart","chromEnd","uniqueID","rate","strand","source"]],
        extra25, df40_sub
    ], ignore_index=True)
    print(f"Ground truth: {len(df_merged)} unique genes, {len(df[df.source=='10-40m'])} from 10-40m regression")
    return df_merged


# ──────────────────────────────────────────────
# 2. Genome-wide BED loading (chromosome-chunked)
# ──────────────────────────────────────────────
def load_bed_chromosome(path: Path, chrom: str) -> pd.DataFrame:
    """Load reads from one chromosome only (memory efficient)."""
    chrom_b = chrom.encode()
    rows = []
    with gzip.open(str(path), 'rb') as fh:
        for line in fh:
            if line[:len(chrom_b)] == chrom_b and line[len(chrom_b):len(chrom_b)+1] == b'\t':
                parts = line.split(b'\t')
                rows.append((int(parts[1]), parts[5].strip()))
    if not rows:
        return pd.DataFrame(columns=["pos","strand"])
    pos, strand = zip(*rows)
    return pd.DataFrame({"pos": np.array(pos, np.int32),
                         "strand": pd.Categorical([s.decode() for s in strand])})


def count_reads(path: Path) -> int:
    """Count total reads in BED.gz (for RPM normalisation)."""
    n = 0
    with gzip.open(str(path), 'rb') as fh:
        for _ in fh:
            n += 1
    return n


# ──────────────────────────────────────────────
# 3. Coverage extraction
# ──────────────────────────────────────────────
def gene_coverage(df_chrom: pd.DataFrame, start: int, end: int,
                  strand: str, lib_size: int) -> np.ndarray:
    """1bp strand-specific RPM coverage over a locus."""
    mask = (df_chrom.strand == strand) & (df_chrom.pos >= start) & (df_chrom.pos < end)
    positions = df_chrom.loc[mask, "pos"].values - start
    length = end - start
    cov = np.zeros(length, dtype=np.float32)
    np.add.at(cov, positions, 1)
    return cov * (1e6 / lib_size)


def ad_average(cov_list: list) -> np.ndarray:
    """
    Algorithmic Diversity (AD) group average via cyclic group Z_m.
    
    Given m replicates (same channel, same service, periodic structure),
    the group-averaged estimator achieves 10*log10(m) dB noise reduction.
    
    Implementation: for the cyclic group Z_m acting on R^M, the group
    average is simply the arithmetic mean — this is the fixed-point of
    all cyclic permutations, and is the minimum-variance unbiased estimator
    under stationarity. The processing gain arises from the m-fold averaging
    of uncorrelated noise.
    
    Parameters
    ----------
    cov_list : list of np.ndarray, each of shape (L,) — same gene, same TP, 
               different replicates.
    
    Returns
    -------
    avg : np.ndarray, shape (L,) — AD group average (RPM)
    gain_db : float — processing gain in dB = 10*log10(m)
    """
    m = len(cov_list)
    stack = np.stack(cov_list, axis=0)          # (m, L)
    avg = stack.mean(axis=0)                    # group average = fixed point of Z_m
    gain_db = 10 * np.log10(m)
    return avg, gain_db


def ad_spectral_concentration(cov: np.ndarray, window: int = 100_000) -> float:
    """
    Compute the AD spectral concentration psi of a coverage vector.
    
    psi = ||FFT(x)||_inf^2 / ||FFT(x)||_F^2  (energy fraction in dominant mode)
    
    High psi indicates structured periodic pausing; low psi = uniform elongation.
    Only uses first `window` positions to keep computation tractable.
    """
    x = cov[:window].astype(np.float64)
    x -= x.mean()  # remove DC
    if x.std() < 1e-10:
        return 0.0
    X = np.abs(np.fft.rfft(x))**2
    return float(X.max() / X.sum()) if X.sum() > 0 else 0.0


# ──────────────────────────────────────────────
# 4. Wave-front speed estimation
# ──────────────────────────────────────────────
def wavefront_position(diff_cov: np.ndarray, smooth_kb: int = 2) -> int:
    """
    Estimate wave-front position from difference coverage (treatment - control).
    
    Method: smooth with a moving average, then find the last position where
    cumulative sum exceeds 5% of total cumulative sum (90th percentile of 
    the wave mass), similar in spirit to Danko's HMM leading-edge state.
    
    Returns position in bp from TSS.
    """
    # smooth at `smooth_kb` kb scale
    kernel = np.ones(smooth_kb * 1000) / (smooth_kb * 1000)
    smoothed = np.convolve(diff_cov, kernel, mode='same')
    smoothed = np.maximum(smoothed, 0.0)
    total = smoothed.sum()
    if total < 1.0:
        return 0
    cumsum = np.cumsum(smoothed)
    # 90th percentile position = leading edge of wave
    idx = np.searchsorted(cumsum, 0.90 * total)
    return int(idx)


def instantaneous_speed(cov: np.ndarray, bin_size: int = 1000) -> np.ndarray:
    """
    Estimate instantaneous RNAPII speed at each position.
    
    Under steady-state occupancy: density ∝ 1/speed.
    Therefore: v(p) = C / rho(p), where rho(p) is local RPM coverage.
    
    C is calibrated so that the mean speed over the gene body equals
    the Danko regression rate (or 2.0 kb/min if unknown).
    
    Parameters
    ----------
    cov      : 1bp coverage array (RPM)
    bin_size : bin width for local density (default 1 kb)
    
    Returns
    -------
    speed : np.ndarray of shape (len(cov)//bin_size,) — speed in kb/min
            (before calibration, relative units)
    """
    n_bins = len(cov) // bin_size
    binned = cov[:n_bins * bin_size].reshape(n_bins, bin_size).mean(axis=1)
    # avoid divide-by-zero: add small floor
    floor = max(binned[binned > 0].min() * 0.1 if binned[binned > 0].size > 0 else 0.001, 1e-4)
    speed = 1.0 / (binned + floor)
    return speed.astype(np.float32)


if __name__ == "__main__":
    print("Loading Danko ground-truth rates...")
    rates = load_danko_rates()
    print(rates[["uniqueID","rate","strand","source"]].head(5).to_string())
    print("\nPipeline modules loaded OK.")
    print("Run compute_all_coverages() to process all genes.")
