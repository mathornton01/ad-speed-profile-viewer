"""
Script 14: Cross-Spectrum & Lag Analysis — v(p) vs ΔG(p)
=========================================================
Purpose: Determine whether local RNA folding free energy ΔG(p) predicts future
elongation speed v(p + L) (folding drives speed), or conversely, whether local
speed v(p) predicts future folding ΔG(p + L) (speed determines how long RNA
has to fold before exiting the exit channel).

Two causal hypotheses tested
-----------------------------
H1 (folding → speed):  Xcorr(ΔG, v) peaks at lag L > 0
                        i.e., ΔG at position p is correlated with v at p+L
H2 (speed → folding):  Xcorr(v, ΔG) peaks at lag L > 0
                        i.e., v at p is correlated with ΔG at p+L

Both are tested via the lagged cross-correlation and the cross-spectrum.

Method
------
For each gene g that has both a speed profile (A_raw40m) and an MFE profile:
  1. Interpolate ΔG(p) onto the same 1000 bp grid as the MFE samples.
  2. Interpolate v(p) onto the same 1000 bp grid.
  3. Compute cross-spectrum S_vG[f] = FFT(v) · conj(FFT(ΔG))
  4. IFFT → cross-correlation xcorr_vG(τ)
  5. Find dominant lag τ* = argmax |xcorr_vG|
  6. Record sign(τ*): positive → v precedes ΔG (H2), negative → ΔG precedes v (H1)

Z_N averaging across all valid genes reduces noise.

Outputs
-------
    results/folding/cross_spectrum_summary.json   — per-gene lags, averaged spectra
    results/folding/xcorr_vG_avg.npy              — averaged cross-correlation
    results/folding/cross_spectrum_vG_avg.npy     — averaged cross-spectrum (complex)
    results/folding/coherence_vG_avg.npy          — coherence spectrum

A text summary of the lag distribution and causal-direction vote is printed.

Usage
-----
    python scripts/14_folding_vs_speed.py
(run from project root)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.stats
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
GENE_INDEX      = PROJECT_ROOT / "results" / "speed_profiles" / "_gene_index.json"
SPEED_PROF_DIR  = PROJECT_ROOT / "results" / "speed_profiles"
FOLD_DIR        = PROJECT_ROOT / "results" / "folding"
OUT_DIR         = FOLD_DIR          # write outputs into the same folder

STEP_BP     = 1000     # MFE was sampled every 1000 bp
PROFILE_KEY = "A_raw40m"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_mfe_profile(gene_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = FOLD_DIR / f"{gene_id}_mfe.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    pos = np.array(d.get("positions_bp", []), dtype=float)
    mfe = np.array(d.get("mfe_kcal_mol", []), dtype=float)
    if len(pos) < 5:
        return None
    return pos, mfe


def load_speed_on_grid(gene: dict, pos_grid: np.ndarray) -> np.ndarray | None:
    """
    Load the A_raw40m speed profile and interpolate onto pos_grid (bp from TSS).
    Returns speed in bp/min.
    """
    path = SPEED_PROF_DIR / gene["file"]
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    prof = data.get("profiles", {}).get(PROFILE_KEY)
    if prof is None:
        return None
    pos_sp  = np.array(prof["positions_bp"],  dtype=float)
    spd_sp  = np.array(prof["speed_kb_min"],  dtype=float) * 1000.0  # → bp/min
    spd_sp  = np.where(spd_sp < 1e-6, 1e-6, spd_sp)

    spd_interp = np.interp(pos_grid, pos_sp, spd_sp,
                           left=spd_sp[0], right=spd_sp[-1])
    return spd_interp


# ---------------------------------------------------------------------------
# Cross-spectrum utilities
# ---------------------------------------------------------------------------
def normalise_zm(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    s = x.std()
    return x / s if s > 0 else x


def cross_spectrum_1d(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (S_ab, S_aa, S_bb) — one-sided, length N//2+1.
    S_ab is complex, S_aa and S_bb are real.
    """
    N  = len(a)
    A  = np.fft.rfft(a) / N
    B  = np.fft.rfft(b) / N
    return A * np.conj(B), (A * np.conj(A)).real, (B * np.conj(B)).real


def dominant_lag(xcorr: np.ndarray, step_bp: int) -> float:
    """
    Return lag in bp corresponding to the peak of |xcorr|.
    Positive lag → first series leads second (v leads ΔG if computed as xcorr(v, ΔG)).
    """
    N = len(xcorr)
    peak = int(np.argmax(np.abs(xcorr)))
    if peak <= N // 2:
        return peak * step_bp
    else:
        return (peak - N) * step_bp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading gene index from {GENE_INDEX}")
    with open(GENE_INDEX) as f:
        genes = json.load(f)
    print(f"  {len(genes)} genes.")

    # --- pass 1: determine common grid length (use shortest valid MFE) -
    # All genes may have different numbers of MFE points.
    # We align them to a common n_pts = min over genes that have MFE data.
    n_pts_list = []
    for gene in genes:
        r = load_mfe_profile(gene["gene_id"])
        if r is not None:
            n_pts_list.append(len(r[0]))

    if not n_pts_list:
        print("No MFE profiles found in results/folding/.  Run script 13 first.")
        return

    n_pts = min(n_pts_list)
    print(f"  {len(n_pts_list)} genes have MFE profiles.  "
          f"Using common grid of {n_pts} points ({n_pts * STEP_BP / 1000:.0f} kb).")

    # Uniform lag axis
    pos_grid = np.arange(1, n_pts + 1, dtype=float) * STEP_BP

    n_freq = n_pts // 2 + 1
    freqs_per_kbp = np.fft.rfftfreq(n_pts, d=1.0)  # cycles per sample (1 sample = STEP_BP bp)

    # Accumulators
    S_vG_sum  = np.zeros(n_freq, dtype=complex)
    S_vv_sum  = np.zeros(n_freq, dtype=float)
    S_GG_sum  = np.zeros(n_freq, dtype=float)
    # H1: ΔG → speed
    S_Gv_sum  = np.zeros(n_freq, dtype=complex)

    per_gene: list[dict] = []
    n_valid  = 0

    for gene in tqdm(genes, desc="Folding vs speed", unit="gene"):
        gene_id = gene["gene_id"]

        mfe_result = load_mfe_profile(gene_id)
        if mfe_result is None:
            continue
        pos_mfe, mfe_vals = mfe_result

        # truncate to common grid length
        if len(pos_mfe) < n_pts:
            continue
        pos_use  = pos_mfe[:n_pts]
        mfe_use  = mfe_vals[:n_pts]

        # Verify grid matches (positions should be 1000, 2000, …)
        # If not exactly aligned, interpolate onto pos_grid
        if not np.allclose(pos_use, pos_grid, rtol=0.01):
            mfe_use = np.interp(pos_grid, pos_use, mfe_use)

        # load speed interpolated onto same grid
        spd = load_speed_on_grid(gene, pos_grid)
        if spd is None:
            continue

        # replace any nonfinite
        if not np.isfinite(mfe_use).all():
            mfe_use = np.where(np.isfinite(mfe_use), mfe_use, np.nanmedian(mfe_use))
        if not np.isfinite(spd).all():
            spd = np.where(np.isfinite(spd), spd, np.nanmedian(spd))

        v_n  = normalise_zm(spd)
        G_n  = normalise_zm(mfe_use)

        # H2: v → ΔG  (does v at p predict ΔG at p+L?)
        S_vG, S_vv, S_GG = cross_spectrum_1d(v_n, G_n)
        # H1: ΔG → v  (does ΔG at p predict v at p+L?)
        S_Gv, _,    _    = cross_spectrum_1d(G_n, v_n)

        S_vG_sum += S_vG
        S_Gv_sum += S_Gv
        S_vv_sum += S_vv
        S_GG_sum += S_GG

        # per-gene lags
        xcorr_vG_g = np.fft.irfft(S_vG, n=n_pts)
        xcorr_Gv_g = np.fft.irfft(S_Gv, n=n_pts)
        lag_vG = dominant_lag(xcorr_vG_g, STEP_BP)   # positive → v precedes ΔG (H2)
        lag_Gv = dominant_lag(xcorr_Gv_g, STEP_BP)   # positive → ΔG precedes v (H1)

        # Pearson r at zero lag
        r_zero, p_zero = scipy.stats.pearsonr(v_n, G_n)

        per_gene.append({
            "gene_id":         gene_id,
            "lag_vG_bp":       lag_vG,
            "lag_Gv_bp":       lag_Gv,
            "pearson_r_zero":  float(r_zero),
            "pearson_p_zero":  float(p_zero),
            "n_pts":           n_pts,
        })
        n_valid += 1

    print(f"\n{n_valid} genes with both speed and MFE profiles.")
    if n_valid == 0:
        print("Nothing to compute.  Run scripts 07/13 first.")
        return

    # --- averaged spectra ------------------------------------------
    S_vG_avg = S_vG_sum / n_valid
    S_Gv_avg = S_Gv_sum / n_valid
    S_vv_avg = S_vv_sum / n_valid
    S_GG_avg = S_GG_sum / n_valid

    xcorr_vG_avg = np.fft.irfft(S_vG_avg, n=n_pts)
    xcorr_Gv_avg = np.fft.irfft(S_Gv_avg, n=n_pts)

    denom = S_vv_avg * S_GG_avg
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.where(denom > 0, np.abs(S_vG_avg)**2 / denom, 0.0)

    lag_vG_avg = dominant_lag(xcorr_vG_avg, STEP_BP)
    lag_Gv_avg = dominant_lag(xcorr_Gv_avg, STEP_BP)

    # --- lag distributions -----------------------------------------
    lags_vG = np.array([g["lag_vG_bp"] for g in per_gene])
    lags_Gv = np.array([g["lag_Gv_bp"] for g in per_gene])

    h2_votes = int((lags_vG > 0).sum())   # v precedes ΔG
    h1_votes = int((lags_vG < 0).sum())   # ΔG precedes v

    print("\n" + "=" * 60)
    print("Causal direction vote (per gene dominant lag):")
    print(f"  H1 (ΔG → speed): {h1_votes}/{n_valid} genes  ({100*h1_votes/n_valid:.1f}%)")
    print(f"  H2 (speed → ΔG): {h2_votes}/{n_valid} genes  ({100*h2_votes/n_valid:.1f}%)")
    print(f"\nAveraged cross-spectrum dominant lag (v vs ΔG): {lag_vG_avg:+.0f} bp")
    print(f"  positive → speed precedes folding (H2: speed determines fold time)")
    print(f"  negative → folding precedes speed (H1: structure affects speed)")
    print(f"\nMedian per-gene lag (v→ΔG): {np.median(lags_vG):+.0f} bp  "
          f"± {np.std(lags_vG):.0f} bp")
    print(f"AD gain estimate: {10*np.log10(n_valid * n_pts):.2f} dB")
    print("=" * 60)

    # --- save numpy arrays -----------------------------------------
    np.save(OUT_DIR / "xcorr_vG_avg.npy",            xcorr_vG_avg)
    np.save(OUT_DIR / "xcorr_Gv_avg.npy",            xcorr_Gv_avg)
    np.save(OUT_DIR / "cross_spectrum_vG_avg.npy",   S_vG_avg)
    np.save(OUT_DIR / "coherence_vG_avg.npy",        coherence)
    np.save(OUT_DIR / "frequencies_per_kbp.npy",     freqs_per_kbp)

    # --- lag distribution plot (optional) --------------------------
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(lags_vG / 1000.0, bins=20, edgecolor="black",
                     color="steelblue", alpha=0.8)
        axes[0].axvline(0, color="red", lw=2, ls="--", label="zero lag")
        axes[0].axvline(np.median(lags_vG) / 1000.0, color="orange",
                        lw=2, ls="-", label=f"median={np.median(lags_vG)/1000:.1f} kb")
        axes[0].set_xlabel("Dominant lag (kb)", fontsize=12)
        axes[0].set_ylabel("Number of genes", fontsize=12)
        axes[0].set_title("Lag distribution: v(p) vs ΔG(p)\n"
                          "positive = speed precedes ΔG (H2)", fontsize=11)
        axes[0].legend()

        axes[1].plot(freqs_per_kbp * 1000 / STEP_BP,   # convert to cycles/Mbp
                     coherence, lw=1.5, color="darkorange")
        axes[1].set_xlabel("Frequency (cycles/Mbp)", fontsize=12)
        axes[1].set_ylabel("Coherence", fontsize=12)
        axes[1].set_title("Coherence: v(p) vs ΔG(p)\n(averaged across genes)", fontsize=11)
        axes[1].set_ylim(0, 1)

        fig.tight_layout()
        plot_path = OUT_DIR / "lag_distribution.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"\nLag distribution plot saved to {plot_path}")
    else:
        print("\n(matplotlib not available — skipping lag distribution plot)")

    # --- summary JSON ----------------------------------------------
    summary = {
        "n_genes_valid":        n_valid,
        "n_pts_per_gene":       n_pts,
        "step_bp":              STEP_BP,
        "ad_gain_dB":           10.0 * np.log10(n_valid * n_pts),
        "dominant_lag_avg_vG_bp": lag_vG_avg,
        "dominant_lag_avg_Gv_bp": lag_Gv_avg,
        "h1_votes":             h1_votes,
        "h2_votes":             h2_votes,
        "median_lag_vG_bp":     float(np.median(lags_vG)),
        "std_lag_vG_bp":        float(np.std(lags_vG)),
        "interpretation": {
            "H1": "ΔG(p) predicts v(p+L) — RNA structure affects elongation speed",
            "H2": "v(p) predicts ΔG(p+L) — speed determines how long RNA folds",
            "dominant": "H2" if h2_votes > h1_votes else "H1",
        },
        "per_gene": per_gene,
        "files": {
            "xcorr_vG_avg":          "results/folding/xcorr_vG_avg.npy",
            "xcorr_Gv_avg":          "results/folding/xcorr_Gv_avg.npy",
            "cross_spectrum_vG_avg": "results/folding/cross_spectrum_vG_avg.npy",
            "coherence_vG_avg":      "results/folding/coherence_vG_avg.npy",
            "frequencies":           "results/folding/frequencies_per_kbp.npy",
        },
    }

    out_path = OUT_DIR / "cross_spectrum_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
