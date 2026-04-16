#!/usr/bin/env python3
"""
benchmark.py - Estimate elongation rates for all 81 Danko genes using:
  1. Wave-front leading-edge method (noise-corrected, AD-averaged replicates)
  2. Instantaneous speed via 1/density (with AD noise reduction)
  
Compare both against Danko regression rates. Generate correlation plots.
"""
import numpy as np
import pandas as pd
import pickle, json
from pathlib import Path
from scipy.stats import linregress, spearmanr, pearsonr
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE      = Path("/home/mathornton/herald-workspace/simon/Projects/AD Experiments/read accumulation rate experiments")
RATES_DIR = BASE / "data" / "danko_rates"
PROC      = BASE / "data" / "processed"
RESULTS   = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(PROC / "gene_coverages.pkl", "rb") as f:
    coverages = pickle.load(f)

rates = pd.read_csv(RATES_DIR / "MCF7.10-40m.regressionRate.tsv", sep="\t")
rates = rates.set_index("uniqueID")
TP_MIN = {"10m": 10, "25m": 25, "40m": 40}

# ── Wave-front detector ───────────────────────────────────────────────────────
def wavefront_leading_edge(cov_t, cov_0, bin_size=2000, smooth_bins=5):
    """
    Noise-corrected leading-edge detector.
    
    Subtracts the distal noise floor (beyond 80 kb) from the diff coverage,
    then finds the last bin significantly above noise (2-sigma threshold).
    The global transcription increase from E2 treatment (uniform RPM increase)
    is removed by the noise-floor subtraction.
    """
    L = min(len(cov_t), len(cov_0))
    n_bins = L // bin_size
    if n_bins < 5:
        return 0
    b_t = cov_t[:n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)
    b_0 = cov_0[:n_bins*bin_size].reshape(n_bins, bin_size).mean(axis=1)
    diff = b_t - b_0
    # smooth
    kernel = np.ones(smooth_bins) / smooth_bins
    sdiff  = np.convolve(diff, kernel, mode="same")
    # distal noise floor
    distal_bin = max(80_000 // bin_size, n_bins // 2)
    if distal_bin < n_bins:
        noise_med = np.median(sdiff[distal_bin:])
        noise_std = sdiff[distal_bin:].std()
    else:
        noise_med = 0.0
        noise_std = sdiff.std() * 0.5
    corrected  = sdiff - noise_med
    threshold  = max(2.0 * noise_std, 1e-6)
    above = np.where(corrected > threshold)[0]
    return int(above[-1]) * bin_size if len(above) else 0


def estimate_rate_wavefront(uid, coverages):
    """Estimate elongation rate via linear regression of wave-front positions."""
    cov0 = coverages[uid]["0m"]
    edges = {}
    for tp, t in TP_MIN.items():
        if tp in coverages[uid]:
            edges[t] = wavefront_leading_edge(coverages[uid][tp], cov0)
    if len(edges) < 2:
        return np.nan, np.nan, edges
    ts    = sorted(edges.keys())
    dists = [edges[t] / 1000 for t in ts]
    if max(dists) < 0.1:
        return np.nan, np.nan, edges
    slope, intercept, r, p, se = linregress(ts, dists)
    return slope, r**2, edges


# ── Instantaneous speed (1/density) ──────────────────────────────────────────
def instantaneous_speed_summary(uid, coverages, bin_size=5000):
    """
    Compute speed(p) = C/rho(p) for each bin in the gene body.
    Summarize as median speed over the gene body.
    Uses AD-averaged coverage (already averaged across replicates).
    
    For the AD application: treats each gene-body bin as a "sensor" observation.
    The group-average (already done across reps) reduces noise so 1/rho is
    a more reliable estimator of speed.
    """
    results = {}
    cov0 = coverages[uid]["0m"]
    
    # Use the 40m timepoint for best signal (longest elapsed time → wave covers most of gene)
    # Also try 25m as a check
    for tp, t in TP_MIN.items():
        if tp not in coverages[uid]:
            continue
        cov_t = coverages[uid][tp]
        # Difference coverage (new RNAPII contribution)
        diff = np.maximum(cov_t - cov0, 0)
        L = len(diff)
        n_bins = L // bin_size
        if n_bins < 2:
            continue
        # Edge estimate for this timepoint
        edge_bp = wavefront_leading_edge(cov_t, cov0)
        edge_bins = max(edge_bp // bin_size, 2)
        # Only use bins within the wave (0 to edge)
        wave_bins = diff[:edge_bins*bin_size].reshape(edge_bins, bin_size).mean(axis=1)
        floor = max(wave_bins[wave_bins>0].min()*0.1, 1e-5) if wave_bins[wave_bins>0].size > 0 else 1e-5
        speed = 1.0 / (wave_bins + floor)   # relative units (1/RPM)
        # Calibrate: scale so that mean speed × time = edge position
        if speed.mean() > 0 and t > 0:
            pred_dist = speed.mean() * t
            target_dist = edge_bp / 1000  # in kb
            if pred_dist > 0:
                scale = target_dist / pred_dist
                speed_kb_min = speed * scale
                results[tp] = {
                    "median_speed": float(np.median(speed_kb_min)),
                    "mean_speed":   float(np.mean(speed_kb_min)),
                    "edge_kb":      edge_bp / 1000,
                    "n_wave_bins":  edge_bins,
                }
    return results


# ── Run for all 81 genes ──────────────────────────────────────────────────────
print(f"Benchmarking {len(rates)} genes...", flush=True)

results_rows = []
for uid, row in rates.iterrows():
    if uid not in coverages:
        continue
    
    wf_rate, wf_r2, edges = estimate_rate_wavefront(uid, coverages)
    speed_res = instantaneous_speed_summary(uid, coverages)
    
    # best instantaneous speed estimate: use 40m if available, else 25m
    inst_speed = np.nan
    for tp in ["40m","25m","10m"]:
        if tp in speed_res:
            inst_speed = speed_res[tp]["mean_speed"]
            break
    
    results_rows.append({
        "uniqueID":       uid,
        "danko_rate":     row["rate"],
        "wf_rate":        wf_rate,
        "wf_r2":          wf_r2,
        "inst_speed_mean":inst_speed,
        "edge_10m_kb":    edges.get(10, np.nan)/1000 if edges.get(10) else np.nan,
        "edge_25m_kb":    edges.get(25, np.nan)/1000 if edges.get(25) else np.nan,
        "edge_40m_kb":    edges.get(40, np.nan)/1000 if edges.get(40) else np.nan,
        "true_dist10_kb": row.get("medianDist10", np.nan)/1000 if not pd.isna(row.get("medianDist10",np.nan)) else np.nan,
        "true_dist25_kb": row.get("medianDist25", np.nan)/1000 if not pd.isna(row.get("medianDist25",np.nan)) else np.nan,
        "true_dist40_kb": row.get("medianDist40", np.nan)/1000 if not pd.isna(row.get("medianDist40",np.nan)) else np.nan,
        "locus_len_kb":   (row["chromEnd"] - row["chromStart"])/1000,
        "strand":         row["strand"],
    })

df = pd.DataFrame(results_rows)
df.to_csv(RESULTS / "benchmark_results.csv", index=False)
print(f"Saved benchmark_results.csv ({len(df)} genes)")

# ── Filter valid estimates and compute correlations ───────────────────────────
valid = df.dropna(subset=["danko_rate","wf_rate"])
valid = valid[valid.wf_rate > 0]
print(f"\nValid wave-front estimates: {len(valid)}/{len(df)}")

if len(valid) >= 5:
    r_pearson, p_pearson = pearsonr(valid.danko_rate, valid.wf_rate)
    r_spearman, p_spearman = spearmanr(valid.danko_rate, valid.wf_rate)
    print(f"Wave-front vs Danko:  Pearson r={r_pearson:.3f} (p={p_pearson:.3e})  Spearman ρ={r_spearman:.3f}")
    
    valid_inst = df.dropna(subset=["danko_rate","inst_speed_mean"])
    valid_inst = valid_inst[valid_inst.inst_speed_mean > 0]
    if len(valid_inst) >= 5:
        r_p2, _ = pearsonr(valid_inst.danko_rate, valid_inst.inst_speed_mean)
        r_s2, _ = spearmanr(valid_inst.danko_rate, valid_inst.inst_speed_mean)
        print(f"1/density vs Danko:   Pearson r={r_p2:.3f}  Spearman ρ={r_s2:.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: wave-front rate vs Danko
if len(valid) >= 5:
    ax = axes[0]
    ax.scatter(valid.danko_rate, valid.wf_rate, alpha=0.7, s=40, color="#2196F3")
    lims = [0, max(valid.danko_rate.max(), valid.wf_rate.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="y=x")
    ax.set_xlabel("Danko rate (kb/min)")
    ax.set_ylabel("Wave-front regression rate (kb/min)")
    ax.set_title(f"Wave-front vs Danko\nr={r_pearson:.3f}, ρ={r_spearman:.3f}, n={len(valid)}")
    ax.legend(fontsize=8)

# Plot 2: edge positions per timepoint vs Danko truth
ax = axes[1]
for tp_col, true_col, color, label in [
    ("edge_10m_kb","true_dist10_kb","#2196F3","10m"),
    ("edge_25m_kb","true_dist25_kb","#FF9800","25m"),
    ("edge_40m_kb","true_dist40_kb","#F44336","40m"),
]:
    sub = df.dropna(subset=[tp_col, true_col])
    if len(sub) > 0:
        ax.scatter(sub[true_col], sub[tp_col], alpha=0.6, s=25, color=color, label=label)
lims = [0, 200]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlabel("Danko medianDist (kb)")
ax.set_ylabel("Estimated wave-front (kb)")
ax.set_title("Wave-front position accuracy per timepoint")
ax.legend(fontsize=8)

# Plot 3: instantaneous speed vs Danko
if len(valid_inst) >= 5:
    ax = axes[2]
    ax.scatter(valid_inst.danko_rate, valid_inst.inst_speed_mean, alpha=0.7, s=40, color="#9C27B0")
    lims = [0, max(valid_inst.danko_rate.max(), valid_inst.inst_speed_mean.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="y=x")
    ax.set_xlabel("Danko rate (kb/min)")
    ax.set_ylabel("Mean instantaneous speed 1/ρ (kb/min)")
    ax.set_title(f"Inst. speed vs Danko\nr={r_p2:.3f}, ρ={r_s2:.3f}, n={len(valid_inst)}")
    ax.legend(fontsize=8)

plt.suptitle("AD-Enhanced GRO-Seq Speed Estimation vs Danko et al. 2013", fontsize=12)
plt.tight_layout()
plt.savefig(str(RESULTS / "benchmark_correlation.png"), dpi=150)
print("\nPlot saved: results/benchmark_correlation.png")

# Summary stats
print("\n=== Summary ===")
print(df[["danko_rate","wf_rate","inst_speed_mean"]].describe().round(3))
