"""
Script 12: Cross-Spectrum Analysis — v(t) vs Epigenomic Marks (AD Averaging)
============================================================================
Purpose: Apply the Z_M AD group-averaging framework to compute cross-spectra
between the instantaneous speed v(t) and each epigenomic mark m(t), both
expressed in uniform transcription-time coordinates (from script 11).

Method
------
For each gene g and mark m, with N time-domain samples:
    V[f]     = FFT(v_g(t))
    M[f]     = FFT(m_g(t))
    S_vm[f]  = (1/N) * V[f] * conj(M[f])   (cross-power spectrum)
    S_vv[f]  = (1/N) * |V[f]|²
    S_mm[f]  = (1/N) * |M[f]|²

Z_N averaging across all G genes:
    <S_vm>[f] = (1/G) Σ_g S_vm_g[f]

Coherence:
    C[f] = |<S_vm>[f]|² / (<S_vv>[f] * <S_mm>[f])

Dominant lag (via IFFT of cross-spectrum):
    xcorr(τ) = IFFT(<S_vm>[f])
    τ_peak   = argmax |xcorr|    (converted to minutes)

AD gain estimate:
    Gain_dB = 10 * log10(N_genes * N_timepoints)

Outputs
-------
    results/cross_spectra/summary.json
    results/cross_spectra/<mark>_cross_spectrum.npy   (complex, length N//2+1)
    results/cross_spectra/<mark>_coherence.npy
    results/cross_spectra/<mark>_xcorr.npy

Usage
-----
    python scripts/12_cross_spectrum_ad.py
(run from project root)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import scipy.signal
import scipy.stats
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
GENE_INDEX     = PROJECT_ROOT / "results" / "speed_profiles" / "_gene_index.json"
TEMP_DIR       = PROJECT_ROOT / "results" / "temporal_profiles"
OUT_DIR        = PROJECT_ROOT / "results" / "cross_spectra"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TIME_POINTS  = 500
TIME_MAX_MIN   = 40.0
DT             = TIME_MAX_MIN / (N_TIME_POINTS - 1)   # minutes per sample
FS             = 1.0 / DT                              # samples per minute


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_temporal_profile(gene_id: str) -> dict | None:
    path = TEMP_DIR / f"{gene_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def normalise_zero_mean(x: np.ndarray) -> np.ndarray:
    """Remove DC and unit-variance normalise so cross-spectrum is scale-free."""
    x = x - np.mean(x)
    std = np.std(x)
    if std > 0:
        x = x / std
    return x


def compute_cross_spectrum(v: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute one-sided cross-spectrum, auto-spectra.
    Returns (S_vm, S_vv, S_mm) each of length N//2+1 (complex for S_vm).
    """
    N   = len(v)
    V   = np.fft.rfft(v) / N
    M   = np.fft.rfft(m) / N
    S_vm = V * np.conj(M)
    S_vv = (V * np.conj(V)).real
    S_mm = (M * np.conj(M)).real
    return S_vm, S_vv, S_mm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading gene index from {GENE_INDEX}")
    with open(GENE_INDEX) as f:
        genes = json.load(f)
    n_genes = len(genes)
    print(f"  {n_genes} genes.")

    # --- discover marks from first available temporal profile ----------
    marks: list[str] = []
    for gene in genes:
        tp = load_temporal_profile(gene["gene_id"])
        if tp is not None:
            marks = sorted(tp.get("marks", {}).keys())
            break
    if not marks:
        warnings.warn("No temporal profiles found.  Run script 11 first.")
        marks = []
    print(f"  {len(marks)} marks: {marks}")

    N_fft  = N_TIME_POINTS
    n_freq = N_fft // 2 + 1
    freqs  = np.fft.rfftfreq(N_fft, d=DT)   # cycles per minute

    # --- accumulate cross-spectra per mark ----------------------------
    # Shape (n_genes, n_freq) → averaged across genes
    mark_S_vm  : dict[str, np.ndarray] = {m: np.zeros(n_freq, dtype=complex)  for m in marks}
    mark_S_vv  : dict[str, np.ndarray] = {m: np.zeros(n_freq, dtype=float)    for m in marks}
    mark_S_mm  : dict[str, np.ndarray] = {m: np.zeros(n_freq, dtype=float)    for m in marks}
    mark_n_gene: dict[str, int]        = {m: 0 for m in marks}

    # Also store per-gene dominant lags for distribution reporting
    lag_lists: dict[str, list[float]] = {m: [] for m in marks}

    for gene in tqdm(genes, desc="Computing cross-spectra", unit="gene"):
        gene_id = gene["gene_id"]
        tp = load_temporal_profile(gene_id)
        if tp is None:
            continue

        v_arr = np.array(tp.get("speed_at_time", []), dtype=float)
        if len(v_arr) != N_TIME_POINTS:
            warnings.warn(f"{gene_id}: speed array length {len(v_arr)} ≠ {N_TIME_POINTS}, skipping.")
            continue

        if not np.isfinite(v_arr).all():
            v_arr = np.where(np.isfinite(v_arr), v_arr, np.nanmedian(v_arr))

        v_norm = normalise_zero_mean(v_arr)

        for mark in marks:
            m_list = tp.get("marks", {}).get(mark, None)
            if m_list is None:
                continue
            m_arr = np.array(m_list, dtype=float)
            if len(m_arr) != N_TIME_POINTS:
                continue
            if not np.isfinite(m_arr).all():
                if np.isnan(m_arr).all():
                    continue
                m_arr = np.where(np.isfinite(m_arr), m_arr, np.nanmedian(m_arr))

            m_norm = normalise_zero_mean(m_arr)

            S_vm, S_vv, S_mm = compute_cross_spectrum(v_norm, m_norm)
            mark_S_vm[mark]   += S_vm
            mark_S_vv[mark]   += S_vv
            mark_S_mm[mark]   += S_mm
            mark_n_gene[mark] += 1

            # per-gene dominant lag (full xcorr for this gene)
            xcorr_g = np.fft.irfft(S_vm, n=N_fft)
            peak_idx = int(np.argmax(np.abs(xcorr_g)))
            # convert index to time lag (minutes)
            if peak_idx <= N_fft // 2:
                lag_min = peak_idx * DT
            else:
                lag_min = (peak_idx - N_fft) * DT
            lag_lists[mark].append(lag_min)

    # --- finalise cross-spectra ---------------------------------------
    summary: dict = {
        "n_genes_total": n_genes,
        "n_time_points": N_TIME_POINTS,
        "time_max_min":  TIME_MAX_MIN,
        "dt_min":        DT,
        "fs_per_min":    FS,
        "ad_gain_dB":    None,
        "marks":         {},
    }

    print("\nCross-spectrum summary:")
    print(f"{'Mark':<30}  {'N genes':>8}  {'Dom lag (min)':>14}  {'Interp':>12}  {'AD gain dB':>12}")
    print("-" * 78)

    for mark in marks:
        G = mark_n_gene[mark]
        if G == 0:
            print(f"{mark:<30}  {'0':>8}  {'—':>14}  {'—':>12}  {'—':>12}")
            continue

        S_vm_avg = mark_S_vm[mark] / G
        S_vv_avg = mark_S_vv[mark] / G
        S_mm_avg = mark_S_mm[mark] / G

        # coherence
        denom = S_vv_avg * S_mm_avg
        with np.errstate(divide="ignore", invalid="ignore"):
            coherence = np.where(denom > 0, np.abs(S_vm_avg)**2 / denom, 0.0)

        # dominant lag via IFFT of averaged cross-spectrum
        xcorr_avg = np.fft.irfft(S_vm_avg, n=N_fft)
        peak_idx  = int(np.argmax(np.abs(xcorr_avg)))
        if peak_idx <= N_fft // 2:
            dominant_lag_min = peak_idx * DT
        else:
            dominant_lag_min = (peak_idx - N_fft) * DT

        # positive lag → mark leads speed; negative → speed leads mark
        direction = "mark precedes speed" if dominant_lag_min > 0 else \
                    ("speed precedes mark" if dominant_lag_min < 0 else "simultaneous")

        # AD gain estimate
        ad_gain_dB = 10.0 * np.log10(G * N_TIME_POINTS)

        # peak coherence frequency
        peak_coh_idx  = int(np.argmax(coherence))
        peak_coh_freq = float(freqs[peak_coh_idx])
        peak_coh_val  = float(coherence[peak_coh_idx])

        lag_arr = np.array(lag_lists[mark])
        lag_median = float(np.median(lag_arr)) if len(lag_arr) > 0 else np.nan
        lag_std    = float(np.std(lag_arr))     if len(lag_arr) > 1 else np.nan

        print(f"{mark:<30}  {G:>8}  {dominant_lag_min:>+14.2f}  {direction:>12}  {ad_gain_dB:>12.2f}")

        # save numpy arrays
        np.save(OUT_DIR / f"{mark}_cross_spectrum.npy",  S_vm_avg)
        np.save(OUT_DIR / f"{mark}_coherence.npy",       coherence)
        np.save(OUT_DIR / f"{mark}_xcorr.npy",           xcorr_avg)
        np.save(OUT_DIR / f"{mark}_auto_vv.npy",         S_vv_avg)
        np.save(OUT_DIR / f"{mark}_auto_mm.npy",         S_mm_avg)

        summary["marks"][mark] = {
            "n_genes":                  G,
            "dominant_lag_min":         dominant_lag_min,
            "lag_direction":            direction,
            "ad_gain_dB":               ad_gain_dB,
            "peak_coherence_freq_cpm":  peak_coh_freq,
            "peak_coherence_value":     peak_coh_val,
            "per_gene_lag_median_min":  lag_median,
            "per_gene_lag_std_min":     lag_std,
            "files": {
                "cross_spectrum": str((OUT_DIR / f"{mark}_cross_spectrum.npy").relative_to(PROJECT_ROOT)),
                "coherence":      str((OUT_DIR / f"{mark}_coherence.npy").relative_to(PROJECT_ROOT)),
                "xcorr":          str((OUT_DIR / f"{mark}_xcorr.npy").relative_to(PROJECT_ROOT)),
            },
        }

    print("-" * 78)

    # overall AD gain using all marks × genes
    total_gene_mark_pairs = sum(mark_n_gene[m] for m in marks)
    if total_gene_mark_pairs > 0:
        summary["ad_gain_dB"] = 10.0 * np.log10(total_gene_mark_pairs * N_TIME_POINTS)
        print(f"\nOverall AD gain estimate: {summary['ad_gain_dB']:.2f} dB")
        print(f"  (based on {total_gene_mark_pairs} gene-mark pairs × {N_TIME_POINTS} time points)")

    # save frequency axis
    np.save(OUT_DIR / "frequencies_cpm.npy", freqs)
    summary["freq_axis_file"] = str((OUT_DIR / "frequencies_cpm.npy").relative_to(PROJECT_ROOT))

    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
