"""
06_grohmm_rates.py
==================
Apply the Python groHMM polymeraseWave HMM to all 81 genes across the
E2 GRO-Seq time course (10m, 25m, 40m).

For each gene × time point:
  1. Extract diff coverage = cov(Xm) - cov(0m) in 5'→3' orientation
  2. Run 3-state HMM → wave front position (bp from TSS)
  3. Linear regression across time points → elongation rate (bp/min)
  4. Benchmark vs Danko medianDist values and published rates

Author: Simon (AD Experiments, 2026-04-04)
"""

import pickle
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from grohmm_python import polymerase_wave, window_coverage, estimate_elongation_rate

BASE = Path(__file__).parent.parent
DATA_PROC  = BASE / "data" / "processed"
DATA_DANKO = BASE / "data" / "danko_rates"
RESULTS    = BASE / "results"
RESULTS.mkdir(exist_ok=True)

WINDOW_SIZE   = 50      # bp — matches groHMM default
TIME_POINTS   = [10, 25, 40]
APPROX_DIST   = {10: 20_000, 25: 50_000, 40: 80_000}  # 2000 bp/min * t

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading gene coverages...")
with open(DATA_PROC / "gene_coverages.pkl", "rb") as f:
    gene_data = pickle.load(f)
print(f"  {len(gene_data)} genes loaded")

# Load library sizes for converting RPM back to approximate counts.
# groHMM requires raw read counts (not RPM) for its Gamma emission model.
# RPM was stored as: count * (1e6 / lib_size), so count ≈ RPM * mean_lib_size / 1e6.
with open(DATA_PROC / "library_sizes.json") as f:
    lib_sizes = json.load(f)

MEAN_LIB = {
    "0m":  np.mean([lib_sizes["0m_R1"], lib_sizes["0m_R2"], lib_sizes["0m_R3"]]),
    "10m": np.mean([lib_sizes["10m_R1"], lib_sizes["10m_R2"], lib_sizes["10m_R3"]]),
    "25m": np.mean([lib_sizes["25m_R1"], lib_sizes["25m_R3"]]),
    "40m": np.mean([lib_sizes["40m_R1"], lib_sizes["40m_R2"], lib_sizes["40m_R3"]]),
}
print(f"  Mean library sizes: {', '.join(f'{k}={v/1e6:.1f}M' for k,v in MEAN_LIB.items())}")

def rpm_to_counts(arr, tp):
    """Convert RPM-normalized array back to approximate read counts."""
    return arr.astype(float) * MEAN_LIB[tp] / 1e6

print("Loading Danko reference rates...")
danko_df = pd.read_csv(DATA_DANKO / "MCF7.10-40m.regressionRate.tsv", sep="\t")
print(f"  {len(danko_df)} genes, columns: {list(danko_df.columns)}")

# Build lookup: "chrom_start_end_strand" → row
danko_df["key"] = (danko_df["chrom"].astype(str) + "_" +
                   danko_df["chromStart"].astype(str) + "_" +
                   danko_df["chromEnd"].astype(str) + "_" +
                   danko_df["strand"].astype(str))
danko_lookup = danko_df.set_index("key").to_dict("index")

# ---------------------------------------------------------------------------
# Match gene_data keys to Danko entries
# ---------------------------------------------------------------------------

matched_genes = []
unmatched = []

for gkey in gene_data.keys():
    if gkey in danko_lookup:
        matched_genes.append(gkey)
    else:
        unmatched.append(gkey)

print(f"\nGene matching: {len(matched_genes)} matched, {len(unmatched)} unmatched")
if unmatched:
    print(f"  First unmatched: {unmatched[:3]}")
    print(f"  First Danko key: {list(danko_lookup.keys())[:3]}")

    # Try to match by coordinates with some tolerance
    # Parse gene_data keys
    gene_data_parsed = {}
    for k in gene_data.keys():
        parts = k.split("_")
        if len(parts) == 4:
            chrom, start, end, strand = parts
            gene_data_parsed[k] = (chrom, int(start), int(end), strand)

    # Check if coordinates are close
    for gkey, (chrom, start, end, strand) in list(gene_data_parsed.items())[:3]:
        print(f"  gene_data: {gkey} → {chrom}:{start}-{end} {strand}")
    for dkey in list(danko_lookup.keys())[:3]:
        row = danko_lookup[dkey]
        print(f"  danko:     {dkey}")

# ---------------------------------------------------------------------------
# Main HMM loop
# ---------------------------------------------------------------------------

results = []
failed  = []

use_genes = matched_genes if matched_genes else list(gene_data.keys())
print(f"\n--- Running HMM on {len(use_genes)} genes ---")
print(f"Window: {WINDOW_SIZE} bp | Max iter: 10 | Emission: gamma")

for gene_idx, gkey in enumerate(use_genes):
    entry    = gene_data[gkey]
    # Convert RPM → approximate counts for HMM (Gamma emissions need integer-like counts)
    cov_0m   = rpm_to_counts(entry["0m"], "0m")
    gene_len = len(cov_0m)

    # Parse strand from key
    parts  = gkey.split("_")
    strand = parts[-1] if len(parts) == 4 else "+"

    # Get Danko ground-truth for this gene (if available)
    danko_row = danko_lookup.get(gkey, None)

    wave_positions = {}  # time_min → wave_end_bp_from_TSS
    wave_details   = {}

    for tp in TIME_POINTS:
        cov_tp = rpm_to_counts(entry[f"{tp}m"], f"{tp}m")

        # Ensure same length
        n = min(len(cov_0m), len(cov_tp))
        c0 = cov_0m[:n]
        ct = cov_tp[:n]

        # Orient 5'→3': for - strand, reverse the array
        if strand == "-":
            c0 = c0[::-1]
            ct = ct[::-1]

        # Window-aggregate
        w_veh  = window_coverage(c0, WINDOW_SIZE)
        w_cond = window_coverage(ct, WINDOW_SIZE)

        # Prepend 200 zero-windows (10 kb) as upstream region —
        # groHMM uses upstreamDist=10000 to anchor state 0 before TSS.
        N_PAD = 200
        w_veh  = np.concatenate([np.zeros(N_PAD), w_veh])
        w_cond = np.concatenate([np.zeros(N_PAD), w_cond])

        if len(w_cond) < 20:
            continue

        # approxDist: use Danko medianDist for this time point if available,
        # else fall back to 2000 bp/min default.
        danko_dist_key = f"medianDist{tp}"
        danko_row_for_tp = danko_lookup.get(gkey, {})
        approx = danko_row_for_tp.get(danko_dist_key, None)
        if approx is None or (isinstance(approx, float) and np.isnan(approx)):
            approx = APPROX_DIST[tp]
        else:
            approx = float(approx)

        try:
            res = polymerase_wave(
                w_cond, w_veh,
                gene_start=0, gene_end=len(w_cond) * WINDOW_SIZE,
                window_size=WINDOW_SIZE,
                approx_dist=approx,
                time_min=tp,
                upstream_dist=10000,   # matches groHMM default
                emission="gamma",
                max_iter=10,
                verbose=False,
            )
            # Subtract padding offset to get bp from TSS
            wave_bp = max(0, res["wave_end_bp"] - N_PAD * WINDOW_SIZE)
            wave_positions[tp] = wave_bp
            wave_details[tp]   = res
        except Exception as e:
            if gene_idx < 5:
                print(f"  ERROR gene {gkey} tp={tp}m: {e}")

    if len(wave_positions) < 2:
        failed.append((gkey, f"only {len(wave_positions)} valid time points"))
        continue

    # Linear regression: wave_end_bp ~ time_min → rate
    rate, r2, intercept = estimate_elongation_rate(wave_positions)

    row = {
        "gene_id":       gkey,
        "gene_len_bp":   gene_len,
        "strand":        strand,
        "rate_bp_min":   rate,
        "rate_kb_min":   rate / 1000.0 if not np.isnan(rate) else np.nan,
        "rate_r2":       r2,
        "intercept_bp":  intercept,
        "n_timepoints":  len(wave_positions),
    }

    for tp in TIME_POINTS:
        if tp in wave_positions:
            row[f"wave_end_{tp}m_bp"] = wave_positions[tp]
        else:
            row[f"wave_end_{tp}m_bp"] = np.nan

    # Add Danko ground truth if available
    if danko_row:
        row["danko_rate_bp_min"]   = danko_row.get("rate", np.nan)
        row["danko_medianDist10"]  = danko_row.get("medianDist10", np.nan)
        row["danko_medianDist25"]  = danko_row.get("medianDist25", np.nan)
        row["danko_medianDist40"]  = danko_row.get("medianDist40", np.nan)
    else:
        row["danko_rate_bp_min"]   = np.nan
        row["danko_medianDist10"]  = np.nan
        row["danko_medianDist25"]  = np.nan
        row["danko_medianDist40"]  = np.nan

    results.append(row)

    if (gene_idx + 1) % 10 == 0 or gene_idx == 0:
        print(f"  [{gene_idx+1}/{len(use_genes)}] {gkey[:40]} rate={rate/1000:.2f} kb/min  r²={r2:.3f}")

print(f"\nFinished: {len(results)} OK | {len(failed)} failed")

# ---------------------------------------------------------------------------
# Save & summarize
# ---------------------------------------------------------------------------

df = pd.DataFrame(results)
df.to_csv(RESULTS / "grohmm_wave_rates.csv", index=False)
print(f"\nSaved: results/grohmm_wave_rates.csv ({len(df)} rows)")

valid_rates = df["rate_kb_min"].dropna()
print(f"\n--- Rate distribution ---")
print(f"  n:      {len(valid_rates)}")
print(f"  median: {valid_rates.median():.2f} kb/min")
print(f"  mean:   {valid_rates.mean():.2f} kb/min")
print(f"  std:    {valid_rates.std():.2f} kb/min")
print(f"  range:  {valid_rates.min():.2f} – {valid_rates.max():.2f} kb/min")

# ---------------------------------------------------------------------------
# Benchmark: compare wave positions vs Danko medianDist
# ---------------------------------------------------------------------------

print("\n--- Wave position benchmark vs Danko medianDist ---")
for tp in TIME_POINTS:
    col_our   = f"wave_end_{tp}m_bp"
    col_danko = f"danko_medianDist{tp}"
    sub = df[[col_our, col_danko]].dropna()
    if len(sub) > 5:
        r, p = stats.pearsonr(sub[col_danko], sub[col_our])
        rho, _ = stats.spearmanr(sub[col_danko], sub[col_our])
        print(f"  {tp}m: r={r:.3f}, ρ={rho:.3f}  (n={len(sub)})")

# ---------------------------------------------------------------------------
# Benchmark: our rate vs Danko published rate
# ---------------------------------------------------------------------------

print("\n--- Rate benchmark vs Danko published rate ---")
sub = df[["rate_bp_min", "danko_rate_bp_min"]].dropna()
if len(sub) > 5:
    r, p = stats.pearsonr(sub["danko_rate_bp_min"], sub["rate_bp_min"])
    rho, ps = stats.spearmanr(sub["danko_rate_bp_min"], sub["rate_bp_min"])
    print(f"  Pearson  r = {r:.3f}  (p={p:.2e})")
    print(f"  Spearman ρ = {rho:.3f}  (p={ps:.2e})")
    print(f"  n = {len(sub)}")

# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: wave position per time point
for col_idx, tp in enumerate(TIME_POINTS):
    ax = axes[0, col_idx]
    col_our   = f"wave_end_{tp}m_bp"
    col_danko = f"danko_medianDist{tp}"
    sub = df[[col_our, col_danko]].dropna()
    if len(sub) > 3:
        r, _ = stats.pearsonr(sub[col_danko], sub[col_our])
        ax.scatter(sub[col_danko] / 1000, sub[col_our] / 1000,
                   alpha=0.6, s=40, color="#1976D2")
        lim = [0, max(sub[col_danko].max(), sub[col_our].max()) / 1000 * 1.1]
        ax.plot(lim, lim, "k--", alpha=0.3)
        ax.set_xlabel(f"Danko medianDist{tp} (kb)")
        ax.set_ylabel(f"groHMM wave front (kb)")
        ax.set_title(f"{tp} min: r={r:.3f}, n={len(sub)}")
    else:
        ax.text(0.5, 0.5, f"n={len(sub)}\n(too few)", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title(f"{tp} min")

# Row 2: rate comparison, rate distribution, residuals
ax = axes[1, 0]
sub = df[["rate_kb_min", "danko_rate_bp_min"]].dropna()
if len(sub) > 3:
    danko_kb = sub["danko_rate_bp_min"] / 1000
    r, p = stats.pearsonr(danko_kb, sub["rate_kb_min"])
    ax.scatter(danko_kb, sub["rate_kb_min"], alpha=0.6, s=40, color="#E53935")
    lim = [0, max(danko_kb.max(), sub["rate_kb_min"].max()) * 1.1]
    ax.plot(lim, lim, "k--", alpha=0.3)
    ax.set_xlabel("Danko rate (kb/min)")
    ax.set_ylabel("groHMM rate (kb/min)")
    ax.set_title(f"Rates: r={r:.3f}, n={len(sub)}")

ax = axes[1, 1]
valid_rates.hist(bins=20, ax=ax, color="#43A047", edgecolor="white")
if len(sub) > 3:
    danko_kb = sub["danko_rate_bp_min"] / 1000
    danko_kb.hist(bins=20, ax=ax, color="#FB8C00", alpha=0.6, edgecolor="white")
    ax.legend(["groHMM", "Danko"])
ax.set_xlabel("Rate (kb/min)")
ax.set_ylabel("Count")
ax.set_title("Rate distributions")

ax = axes[1, 2]
# Load psi results for three-way comparison
psi_path = RESULTS / "psi_vs_danko_rates.csv"
if psi_path.exists():
    psi_df = pd.read_csv(psi_path)
    # Merge with groHMM results
    # gene_id format may differ; try various merge strategies
    if "gene_id" in psi_df.columns and "gene_id" in df.columns:
        merged = psi_df.merge(df, on="gene_id", how="inner")
    else:
        # positional merge
        merged = pd.concat([psi_df.reset_index(drop=True),
                           df.reset_index(drop=True)], axis=1)

    psi_col = [c for c in merged.columns if "psi" in c.lower()]
    danko_col = [c for c in merged.columns if "danko_rate" in c.lower()]

    if psi_col and danko_col and "rate_kb_min" in merged.columns:
        pc, dc = psi_col[0], danko_col[0]
        sub3 = merged[[pc, dc, "rate_kb_min"]].dropna()
        if len(sub3) > 5:
            r_psi, _ = stats.pearsonr(sub3[dc], sub3[pc])
            r_hmm, _ = stats.pearsonr(sub3[dc], sub3["rate_kb_min"])
            ax.scatter(sub3[dc], sub3[pc], alpha=0.6, s=30, color="#9C27B0",
                       label=f"ψ (r={r_psi:.3f})")
            ax2 = ax.twinx()
            ax2.scatter(sub3[dc], sub3["rate_kb_min"], alpha=0.6, s=30,
                        color="#FF5722", marker="^", label=f"HMM r (r={r_hmm:.3f})")
            ax.set_xlabel("Danko rate")
            ax.set_ylabel("AD ψ")
            ax2.set_ylabel("groHMM rate (kb/min)")
            ax.set_title(f"AD ψ vs HMM rate vs Danko")
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="lower right", fontsize=8)
else:
    ax.text(0.5, 0.5, "psi_vs_danko_rates.csv\nnot found", ha="center",
            va="center", transform=ax.transAxes)
    ax.set_title("ψ comparison (N/A)")

plt.suptitle("groHMM Python Port — Wave-Front Rate Estimation vs Danko et al. 2013",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(RESULTS / "grohmm_benchmark.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: results/grohmm_benchmark.png")
print("\nDone.")
