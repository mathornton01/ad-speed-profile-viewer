"""
07_ad_speed_profiles.py
========================
AD-enhanced instantaneous RNAPII speed estimation at 1 bp resolution.

Three approaches are tested and compared:

  Option A — Raw 40m coverage as denominator:
    v_A(p) = C_A / rho_40m(p)
    ρ = raw 40m RPM coverage within wave region
    C calibrated so mean(v_A) = wave_end_40m / 40 min → units: bp/min

  Option B — Differential coverage as denominator:
    v_B(p) = C_B / max(rho_40m(p) - rho_0m(p), eps)
    ρ = [40m − 0m] diff coverage (isolates E2-induced RNAPII only)
    C calibrated the same way → units: bp/min

  Option C — Multi-timepoint product group Z_M × Z_3:
    v_C(p, t) = C_t / rho_t(p) for t ∈ {10, 25, 40}
    Each timepoint calibrated independently using groHMM wave_end_t / t
    Summary: harmonic mean over position and time → single rate per gene

All three approaches apply Z_M group-averaged estimation (FFT periodogram) to
the speed profile within the wave region. The DC component |V[0]|²/M is the
AD-enhanced mean speed estimate with processing gain 10·log10(M) dB.

AD noise reduction across replicates (already applied in gene_coverages.pkl):
  0m/10m/40m: 3 reps → 4.77 dB
  25m:        2 reps → 3.01 dB

Output:
  results/speed_profiles/        — per-gene JSON with speed profiles
  results/ad_speed_rates.csv     — per-gene rates and correlations
  results/ad_speed_comparison.png — figure comparing all 3 approaches

Author: Simon (AD Experiments, 2026-04-04)
"""

import pickle
import json
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).parent.parent
DATA_PROC   = BASE / "data" / "processed"
DATA_DANKO  = BASE / "data" / "danko_rates"
RESULTS     = BASE / "results"
PROFILES_DIR = RESULTS / "speed_profiles"
PROFILES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load gene coverages (RPM, already AD-averaged across replicates)
# ---------------------------------------------------------------------------

print("Loading gene coverages (RPM, AD-averaged across replicates)...")
with open(DATA_PROC / "gene_coverages.pkl", "rb") as f:
    gene_data = pickle.load(f)
print(f"  {len(gene_data)} genes")

with open(DATA_PROC / "library_sizes.json") as f:
    lib_sizes = json.load(f)

MEAN_LIB = {
    "0m":  np.mean([lib_sizes["0m_R1"], lib_sizes["0m_R2"], lib_sizes["0m_R3"]]),
    "10m": np.mean([lib_sizes["10m_R1"], lib_sizes["10m_R2"], lib_sizes["10m_R3"]]),
    "25m": np.mean([lib_sizes["25m_R1"], lib_sizes["25m_R3"]]),
    "40m": np.mean([lib_sizes["40m_R1"], lib_sizes["40m_R2"], lib_sizes["40m_R3"]]),
}

# ---------------------------------------------------------------------------
# Load groHMM wave positions (bp from TSS) and Danko reference rates
# ---------------------------------------------------------------------------

print("Loading groHMM wave positions...")
hmm_df = pd.read_csv(RESULTS / "grohmm_wave_rates.csv")
hmm_df = hmm_df.set_index("gene_id")
print(f"  {len(hmm_df)} genes in groHMM results")

print("Loading Danko reference rates...")
danko_df = pd.read_csv(DATA_DANKO / "MCF7.10-40m.regressionRate.tsv", sep="\t")
danko_df["gene_id"] = (danko_df["chrom"].astype(str) + "_" +
                       danko_df["chromStart"].astype(str) + "_" +
                       danko_df["chromEnd"].astype(str) + "_" +
                       danko_df["strand"].astype(str))
danko_df = danko_df.set_index("gene_id")
print(f"  {len(danko_df)} genes in Danko reference")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def smooth_coverage(arr, window=50):
    """Moving-average smoothing with boundary handling."""
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr.astype(float), kernel, mode="same")

def ad_periodogram(v):
    """
    Z_M group-averaged estimator applied to speed vector v ∈ R^M.

    The group average of outer products under cyclic shifts equals:
        R_hat = (1/M) * DFT^H * diag(|V[f]|^2) * DFT
    where V = DFT(v).

    Eigenvalues: lambda_f = |V[f]|^2 / M  (periodogram ordinates)
    Processing gain over outer product: 10*log10(M) dB

    Returns:
        freqs:      frequency indices 0..M-1
        power:      |V[f]|^2 / M  (eigenvalues of R_hat)
        psi:        spectral concentration = max(power) / sum(power)
        gain_dB:    10*log10(M)
        dc_speed:   |V[0]| / M = mean(v)  [the AD-enhanced mean speed]
    """
    M = len(v)
    V = np.fft.fft(v)
    power = (np.abs(V) ** 2) / M
    freqs = np.arange(M)
    total = power.sum()
    psi = power.max() / total if total > 0 else 0.0
    gain_dB = 10 * np.log10(M) if M > 1 else 0.0
    # DC component: V[0] = sum(v) = M * mean(v), so |V[0]|/M = mean(v)
    dc_speed = np.abs(V[0]) / M
    return {
        "freqs":    freqs,
        "power":    power,
        "psi":      psi,
        "gain_dB":  gain_dB,
        "dc_speed": dc_speed,
    }

def calibrate_constant(v_raw, wave_end_bp, elapsed_min, eps=1e-9):
    """
    Calibrate the constant C so that mean(v_calibrated) = wave_end_bp / elapsed_min.

    v_raw(p) = 1 / rho(p) (unnormalized)
    v_calibrated(p) = C * v_raw(p)

    mean(C * v_raw) = wave_end_bp / elapsed_min
    => C = (wave_end_bp / elapsed_min) / mean(v_raw)

    Returns C in bp/min.
    """
    mean_v_raw = np.mean(v_raw)
    if mean_v_raw < eps:
        return 1.0
    target = wave_end_bp / elapsed_min  # bp/min
    return target / mean_v_raw

def compute_speed_profile_A(cov_40m, wave_end_bp, smooth_win=50, eps=1e-6):
    """
    Option A: v(p) = C / rho_40m(p)
    Uses raw 40m coverage as denominator.
    Restricted to wave region: positions 0..wave_end_bp from TSS.
    """
    M = min(int(wave_end_bp), len(cov_40m))
    if M < 100:
        return None
    rho = cov_40m[:M].astype(float)
    rho_smooth = smooth_coverage(rho, smooth_win)
    rho_smooth = np.maximum(rho_smooth, eps)
    v_raw = 1.0 / rho_smooth
    C = calibrate_constant(v_raw, wave_end_bp, 40.0)
    v = C * v_raw  # bp/min at each position
    return v

def compute_speed_profile_B(cov_40m, cov_0m, wave_end_bp, smooth_win=50, eps=1e-6):
    """
    Option B: v(p) = C / max(rho_40m(p) - rho_0m(p), eps)
    Uses differential (E2-induced only) coverage as denominator.
    """
    M = min(int(wave_end_bp), len(cov_40m), len(cov_0m))
    if M < 100:
        return None
    diff = cov_40m[:M].astype(float) - cov_0m[:M].astype(float)
    diff_smooth = smooth_coverage(diff, smooth_win)
    diff_smooth = np.maximum(diff_smooth, eps)
    v_raw = 1.0 / diff_smooth
    C = calibrate_constant(v_raw, wave_end_bp, 40.0)
    v = C * v_raw  # bp/min at each position
    return v

def compute_speed_profile_C(cov_10m, cov_25m, cov_40m, cov_0m,
                             wave_end_10m, wave_end_25m, wave_end_40m,
                             smooth_win=50, eps=1e-6):
    """
    Option C: multi-timepoint Z_M x Z_3 product group.

    For each timepoint t ∈ {10, 25, 40}:
      - wave region: 0..wave_end_t
      - v_t(p) = C_t / rho_t(p), calibrated to bp/min using wave_end_t / t

    Product group summary:
      - Time-average: v_avg(p) = mean_t(v_t(p)) over shared wave region
      - This corresponds to the temporal Z_3 group average (averaging across the
        Z_3 orbit in time), then Z_M spatial averaging for the final mean speed.

    Also saves per-timepoint profiles for visualization.
    """
    results_c = {}
    profiles = {}
    shared_M = None

    tp_data = [
        (10, cov_10m, wave_end_10m),
        (25, cov_25m, wave_end_25m),
        (40, cov_40m, wave_end_40m),
    ]

    for (t, cov_t, wave_end_t) in tp_data:
        if np.isnan(wave_end_t) or wave_end_t < 1000:
            continue
        M_t = min(int(wave_end_t), len(cov_t))
        if M_t < 100:
            continue
        rho_t = cov_t[:M_t].astype(float)
        rho_smooth = smooth_coverage(rho_t, smooth_win)
        rho_smooth = np.maximum(rho_smooth, eps)
        v_raw = 1.0 / rho_smooth
        C_t = calibrate_constant(v_raw, wave_end_t, float(t))
        v_t = C_t * v_raw
        profiles[t] = v_t
        results_c[t] = {"v": v_t, "M": M_t, "C": C_t}

    if len(profiles) < 2:
        return None, profiles

    # Temporal Z_3 product group average over the shared (minimum) wave region
    min_M = min(len(v) for v in profiles.values())
    stacked = np.stack([v[:min_M] for v in profiles.values()], axis=0)  # (n_tp, M)

    # Product group Z_M x Z_T:
    # Z_T averaging: mean over time axis → reduces to Z_M problem in space
    v_time_avg = np.mean(stacked, axis=0)  # (M,)

    return v_time_avg, profiles

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

print("\n--- Computing AD speed profiles for all genes ---")
print(f"  Output: {PROFILES_DIR}/")

all_results = []
n_saved = 0

for gene_idx, (gkey, entry) in enumerate(gene_data.items()):

    cov_0m  = entry["0m"].astype(float)
    cov_10m = entry["10m"].astype(float)
    cov_25m = entry["25m"].astype(float)
    cov_40m = entry["40m"].astype(float)

    # Strand: flip negative-strand genes to 5'→3' orientation
    strand = gkey.split("_")[-1]
    if strand == "-":
        cov_0m  = cov_0m[::-1]
        cov_10m = cov_10m[::-1]
        cov_25m = cov_25m[::-1]
        cov_40m = cov_40m[::-1]

    # Get groHMM wave positions
    if gkey not in hmm_df.index:
        continue
    hmm_row = hmm_df.loc[gkey]
    wave_10m = hmm_row.get("wave_end_10m_bp", np.nan)
    wave_25m = hmm_row.get("wave_end_25m_bp", np.nan)
    wave_40m = hmm_row.get("wave_end_40m_bp", np.nan)
    grohmm_rate = hmm_row.get("rate_kb_min", np.nan)

    # Danko reference rate
    danko_rate = danko_df.loc[gkey, "rate"] if gkey in danko_df.index else np.nan

    # Parse gene coordinates for annotation
    parts = gkey.split("_")
    chrom, gstart, gend = parts[0], int(parts[1]), int(parts[2])
    gene_len = gend - gstart

    # ---- Option A ----
    v_A = None
    if not np.isnan(wave_40m) and wave_40m > 1000:
        v_A = compute_speed_profile_A(cov_40m, wave_40m)

    # ---- Option B ----
    v_B = None
    if not np.isnan(wave_40m) and wave_40m > 1000:
        v_B = compute_speed_profile_B(cov_40m, cov_0m, wave_40m)

    # ---- Option C ----
    v_C_avg, v_C_per_tp = compute_speed_profile_C(
        cov_10m, cov_25m, cov_40m, cov_0m,
        wave_10m, wave_25m, wave_40m,
    )

    # ---- AD periodogram for each approach ----
    row = {
        "gene_id":       gkey,
        "chrom":         chrom,
        "gene_start":    gstart,
        "gene_end":      gend,
        "strand":        strand,
        "gene_len_bp":   gene_len,
        "grohmm_rate_kb_min": grohmm_rate,
        "danko_rate_kb_min":  danko_rate / 1000 if not np.isnan(danko_rate) else np.nan,
        "wave_end_40m_bp":    wave_40m,
        "wave_end_25m_bp":    wave_25m,
        "wave_end_10m_bp":    wave_10m,
    }

    profile_data = {
        "gene_id":   gkey,
        "chrom":     chrom,
        "start":     gstart,
        "end":       gend,
        "strand":    strand,
        "gene_len":  gene_len,
        "wave_40m":  float(wave_40m) if not np.isnan(wave_40m) else None,
        "wave_25m":  float(wave_25m) if not np.isnan(wave_25m) else None,
        "wave_10m":  float(wave_10m) if not np.isnan(wave_10m) else None,
        "grohmm_rate_kb_min": float(grohmm_rate) if not np.isnan(grohmm_rate) else None,
        "danko_rate_kb_min":  float(danko_rate / 1000) if not np.isnan(danko_rate) else None,
        "profiles":  {},
    }

    for label, v in [("A_raw40m", v_A), ("B_diff40m", v_B), ("C_multitp", v_C_avg)]:
        if v is None:
            row[f"mean_speed_{label}_bp_min"] = np.nan
            row[f"psi_{label}"] = np.nan
            row[f"gain_dB_{label}"] = np.nan
            continue

        ad = ad_periodogram(v)

        # Subsample profile for storage (store every 10 bp to keep JSON small)
        step = max(1, len(v) // 5000)
        positions = list(range(0, len(v), step))
        speeds_kb = (v[::step] / 1000).tolist()  # convert to kb/min

        row[f"mean_speed_{label}_bp_min"] = float(ad["dc_speed"])
        row[f"psi_{label}"] = float(ad["psi"])
        row[f"gain_dB_{label}"] = float(ad["gain_dB"])
        row[f"M_{label}"] = len(v)

        profile_data["profiles"][label] = {
            "positions_bp": positions,
            "speed_kb_min": speeds_kb,
            "mean_speed_bp_min": float(ad["dc_speed"]),
            "psi": float(ad["psi"]),
            "gain_dB": float(ad["gain_dB"]),
            "M": len(v),
        }

    # Add per-timepoint profiles for Option C
    for t, v_t in v_C_per_tp.items():
        step = max(1, len(v_t) // 5000)
        positions = list(range(0, len(v_t), step))
        speeds_kb = (v_t[::step] / 1000).tolist()
        profile_data["profiles"][f"C_tp{t}m"] = {
            "positions_bp": positions,
            "speed_kb_min": speeds_kb,
        }

    # Add raw coverage profile (downsampled) for reference
    for tp_label, cov in [("0m", cov_0m), ("40m", cov_40m)]:
        M_cov = min(int(wave_40m) if not np.isnan(wave_40m) else len(cov), len(cov))
        step = max(1, M_cov // 5000)
        profile_data["profiles"][f"coverage_{tp_label}"] = {
            "positions_bp": list(range(0, M_cov, step)),
            "rpm": cov[:M_cov:step].tolist(),
        }

    all_results.append(row)

    # Save per-gene JSON
    safe_key = gkey.replace("/", "_").replace(":", "_")
    json_path = PROFILES_DIR / f"{safe_key}.json"
    with open(json_path, "w") as f:
        json.dump(profile_data, f, separators=(",", ":"))
    n_saved += 1

    if (gene_idx + 1) % 10 == 0 or gene_idx == 0:
        vA_mean = row.get("mean_speed_A_raw40m_bp_min", np.nan)
        print(f"  [{gene_idx+1}/{len(gene_data)}] {gkey[:40]}"
              f"  vA={vA_mean/1000:.2f}kb/min"
              f"  ψA={row.get('psi_A_raw40m', 0):.3f}"
              f"  Danko={row.get('danko_rate_kb_min', np.nan):.2f}kb/min")

print(f"\n{n_saved} gene profiles saved to results/speed_profiles/")

# ---------------------------------------------------------------------------
# Save summary CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame(all_results)
df.to_csv(RESULTS / "ad_speed_rates.csv", index=False)
print(f"Saved: results/ad_speed_rates.csv ({len(df)} rows)")

# ---------------------------------------------------------------------------
# Correlation benchmark
# ---------------------------------------------------------------------------

print("\n--- Correlation with Danko rates ---")
sub = df[df["danko_rate_kb_min"].notna()].copy()
print(f"  n (with Danko rate): {len(sub)}")

for label, col in [
    ("A: C/rho_40m",      "mean_speed_A_raw40m_bp_min"),
    ("B: C/diff_40m",     "mean_speed_B_diff40m_bp_min"),
    ("C: multi-tp mean",  "mean_speed_C_multitp_bp_min"),
    ("groHMM",            "grohmm_rate_kb_min"),
]:
    if col not in sub.columns:
        continue
    s = sub[[col, "danko_rate_kb_min"]].dropna()
    if len(s) < 5:
        print(f"  {label:25s}: n={len(s)} (too few)")
        continue
    r, p = stats.pearsonr(s["danko_rate_kb_min"], s[col])
    rho, ps = stats.spearmanr(s["danko_rate_kb_min"], s[col])
    print(f"  {label:25s}: Pearson r={r:.3f} (p={p:.2e}), Spearman ρ={rho:.3f}  n={len(s)}")

# Save correlation results to JSON for web app
corr_results = {}
for label, col in [
    ("A_raw40m", "mean_speed_A_raw40m_bp_min"),
    ("B_diff40m", "mean_speed_B_diff40m_bp_min"),
    ("C_multitp", "mean_speed_C_multitp_bp_min"),
    ("groHMM", "grohmm_rate_kb_min"),
]:
    if col not in sub.columns:
        continue
    s = sub[[col, "danko_rate_kb_min"]].dropna()
    if len(s) < 5:
        continue
    r, p = stats.pearsonr(s["danko_rate_kb_min"], s[col])
    rho, ps = stats.spearmanr(s["danko_rate_kb_min"], s[col])
    corr_results[label] = {
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(ps),
        "n": int(len(s)),
    }
with open(RESULTS / "speed_profiles" / "_correlations.json", "w") as f:
    json.dump(corr_results, f, indent=2)

# Save gene index for web app
gene_index = []
for gkey in all_results:
    gkey_str = gkey["gene_id"]
    safe_key = gkey_str.replace("/", "_").replace(":", "_")
    gene_index.append({
        "gene_id":   gkey_str,
        "file":      safe_key + ".json",
        "chrom":     gkey["chrom"],
        "start":     gkey["gene_start"],
        "end":       gkey["gene_end"],
        "strand":    gkey["strand"],
        "grohmm_rate_kb_min": gkey.get("grohmm_rate_kb_min"),
        "danko_rate_kb_min":  gkey.get("danko_rate_kb_min"),
        "psi_A":     gkey.get("psi_A_raw40m"),
        "psi_B":     gkey.get("psi_B_diff40m"),
        "psi_C":     gkey.get("psi_C_multitp"),
    })
with open(RESULTS / "speed_profiles" / "_gene_index.json", "w") as f:
    json.dump(gene_index, f, separators=(",", ":"))
print(f"Saved: results/speed_profiles/_gene_index.json ({len(gene_index)} entries)")

# ---------------------------------------------------------------------------
# Comparison figure: 3 panels (A vs Danko, B vs Danko, C vs Danko)
# ---------------------------------------------------------------------------

print("\nGenerating comparison figure...")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("AD Instantaneous Speed Estimates vs Danko et al. 2013 Elongation Rates",
             fontsize=13)

panels = [
    ("A: C/rho_40m",     "mean_speed_A_raw40m_bp_min", "#1976D2"),
    ("B: C/diff_40m",    "mean_speed_B_diff40m_bp_min", "#388E3C"),
    ("C: multi-tp mean", "mean_speed_C_multitp_bp_min", "#7B1FA2"),
    ("groHMM (ref)",     "grohmm_rate_kb_min",          "#E53935"),
]

for ax, (label, col, color) in zip(axes, panels):
    if col not in sub.columns:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label)
        continue

    s = sub[[col, "danko_rate_kb_min"]].dropna()
    if len(s) < 3:
        ax.text(0.5, 0.5, f"n={len(s)}\n(too few)", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_title(label)
        continue

    # Convert to kb/min for display
    x_vals = s["danko_rate_kb_min"]
    if "bp_min" in col:
        y_vals = s[col] / 1000  # convert bp/min → kb/min
    else:
        y_vals = s[col]

    r, p = stats.pearsonr(x_vals, y_vals)
    rho, _ = stats.spearmanr(x_vals, y_vals)

    ax.scatter(x_vals, y_vals, alpha=0.6, s=35, color=color, edgecolors="white", lw=0.3)

    # Regression line
    slope, intercept, *_ = stats.linregress(x_vals, y_vals)
    xl = np.array([x_vals.min(), x_vals.max()])
    ax.plot(xl, slope * xl + intercept, "--", color=color, alpha=0.7, lw=1.5)

    ax.set_xlabel("Danko rate (kb/min)", fontsize=10)
    ax.set_ylabel("Estimated rate (kb/min)", fontsize=10)
    ax.set_title(f"{label}\nr={r:.3f}, ρ={rho:.3f}, n={len(s)}", fontsize=10)

    # Processing gain annotation
    if label != "groHMM (ref)":
        psi_col = col.replace("mean_speed_", "psi_").replace("_bp_min", "")
        if psi_col in sub.columns:
            mean_gain = sub[col.replace("mean_speed_", "gain_dB_").replace("_bp_min","")].mean()
            ax.text(0.05, 0.92, f"AD gain ≈ {mean_gain:.0f} dB",
                    transform=ax.transAxes, fontsize=8, color="gray")

plt.tight_layout()
plt.savefig(RESULTS / "ad_speed_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/ad_speed_comparison.png")

print("\nDone. All outputs in results/speed_profiles/ and results/ad_speed_rates.csv")
