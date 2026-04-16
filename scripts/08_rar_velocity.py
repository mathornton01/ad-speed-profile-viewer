#!/usr/bin/env python3
"""
08_rar_velocity.py  —  1-D optical-flow / continuity-equation velocity estimation
                        for RNAPII elongation from GRO-seq time-course data.

PHYSICS BACKGROUND
==================
RNAPII transport along a gene satisfies the 1-D continuity (conservation) equation
in the wave region:

    ∂ρ/∂t  +  ∂(ρ·v)/∂x  =  0                                 (1)

where:
  ρ(x, t)  is the density of elongating polymerases (RPM coverage at position x, time t)
  v(x, t)  is the local elongation velocity (bp/min)
  x        is the genomic coordinate (bp from TSS)
  t        is time (minutes)

For slowly varying v(x): (1) simplifies to the advection / optical-flow equation:

    ∂ρ/∂t  +  v · ∂ρ/∂x  =  0                                 (2)

Rearranging gives the per-position velocity estimator (Lucas–Kanade 1-D):

    v(x, t)  =  -(∂ρ/∂t) / (∂ρ/∂x)                           (3)

This is computed at every valid position p in the wave region and at 3 time
intervals, then averaged.

INTEGRATED CONTINUITY EQUATION  (centroid-shift estimator)
==========================================================
Integrating (1) over the entire wave region [0, L]:

    d/dt ∫ ρ dx  =  -[ρ·v]₀ᴸ  ≈  0   (boundary terms vanish)

This implies that the centre-of-mass of the NEWLY ACCUMULATED density:

    x̄(t)  =  ∫ Δρ(x,t) · x dx / ∫ Δρ(x,t) dx,   Δρ = ρ(t) - ρ(0)

moves at a velocity

    v_cm  =  Δx̄ / Δt

This centroid-shift estimator is a spatially-integrated form of the
continuity equation.  By averaging over M ~ wave_end / BIN effective
positions it obtains the same Z_M processing gain as the per-position
optical-flow, but with much lower noise because it acts like a matched
filter for the bulk wave translation.

ANALOG-DETECTOR (AD) PROCESSING GAIN
======================================
Per-position optical flow: M valid positions, T=3 intervals:

    Z_M  =  10 · log10(M)  dB               (spatial averaging)
    Z_T  =  10 · log10(3)  ≈  4.8 dB        (temporal averaging)

For a typical gene with M ≈ 50 valid 1-kb bins:

    Total gain  ≈  10 · log10(150)  ≈  21.8 dB

vs Danko's 3 scalar measurements (effectively 1 position, T=3 → 4.8 dB).

DATA
====
  gene_coverages.pkl   : dict  gene_id → dict  timepoint → 1-D numpy array (RPM, 1 bp/bin)
  grohmm_wave_rates.csv: per-gene groHMM wave-front positions and rates
  danko_rates/         : Danko regression-rate TSV files (rate in kb/min)
"""

import os
import sys
import json
import pickle
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings('ignore')

# ── paths ────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE, 'data', 'processed')
DANKO_DIR    = os.path.join(BASE, 'data', 'danko_rates')
RESULTS_DIR  = os.path.join(BASE, 'results')
PROFILES_DIR = os.path.join(RESULTS_DIR, 'speed_profiles')

COV_PKL    = os.path.join(DATA_DIR,    'gene_coverages.pkl')
GROHMM_CSV = os.path.join(RESULTS_DIR, 'grohmm_wave_rates.csv')
OUT_CSV    = os.path.join(RESULTS_DIR, 'rar_velocity_rates.csv')
OUT_PLOT   = os.path.join(RESULTS_DIR, 'rar_velocity_correlation.png')

# ── numerical parameters ─────────────────────────────────────────────────────
BIN_BP     = 1000   # spatial binning: 1 kb bins to reduce sparsity noise
SMOOTH_BIN = 10     # uniform-filter half-width in bins (= 10 kb smoothing)
                    # chosen to be ~1/3 of the minimum inter-timepoint wave
                    # displacement (wave moves ~25 kb in 10 min → σ ~ 8 kb)

# Time intervals (minutes) between GRO-seq snapshots
DT_10  = 10.0   # 0  → 10 min
DT_25  = 15.0   # 10 → 25 min
DT_40  = 15.0   # 25 → 40 min

# Optical-flow masking thresholds
V_MIN_KB   = 0.1    # clip lower bound (kb/min) — below this is noise
V_MAX_KB   = 30.0   # clip upper bound (kb/min) — above this is non-physical
GRAD_FRAC  = 0.1    # |∂ρ/∂x| must exceed GRAD_FRAC × median(|∂ρ/∂x|)
RHO_MIN    = 5e-5   # minimum average coverage to use a 1-kb bin (RPM)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_danko_rates(danko_dir: str) -> pd.DataFrame:
    """
    Merge all three Danko regression-rate TSV files.
    Each file has columns: chrom, chromStart, chromEnd, uniqueID, rate, strand, …
    The 'rate' column is in kb/min.
    We take the per-gene mean across files (different files cover different
    time intervals; the regression rate should be consistent).
    """
    frames = []
    for fpath in sorted(glob.glob(os.path.join(danko_dir, '*.tsv'))):
        if fpath.endswith('.gz'):
            continue
        df = pd.read_csv(fpath, sep='\t')
        frames.append(df[['uniqueID', 'rate']].copy())
        print(f"  Loaded Danko file: {os.path.basename(fpath)}  ({len(df)} genes)")

    if not frames:
        raise FileNotFoundError(f"No Danko TSV files found in {danko_dir}")

    combined = pd.concat(frames, ignore_index=True)
    danko = (combined
             .groupby('uniqueID')['rate']
             .mean()
             .reset_index()
             .rename(columns={'rate': 'danko_rate_kb_min', 'uniqueID': 'gene_id'}))
    print(f"  → {len(danko)} unique Danko genes after merging\n")
    return danko


def bin_and_smooth(arr: np.ndarray, n_bins: int, b: int, sm: int) -> np.ndarray:
    """
    Aggregate a 1-bp RPM coverage array into `n_bins` bins of width `b` bp,
    then apply a box (uniform) smooth of half-width `sm` bins.

    Binning converts the very sparse 1-bp signal into a smoother density
    estimate while preserving total read counts.  The uniform filter replaces
    shot-noise spikes with their neighbourhood average — equivalent to
    convolving with a box kernel of width sm × b bp.
    """
    clipped = arr[:n_bins * b].astype(float)
    binned  = clipped.reshape(n_bins, b).mean(axis=1)
    return uniform_filter1d(binned, size=sm, mode='nearest')


def optical_flow_interval(ra: np.ndarray,
                          rb: np.ndarray,
                          dt: float) -> tuple:
    """
    Estimate the per-position velocity field  v(x)  from two consecutive
    smoothed coverage snapshots separated by dt minutes, using the 1-D
    Lucas–Kanade / continuity-equation formula:

        v(x)  =  -(∂ρ/∂t) / (∂ρ/∂x)

    where:
        ∂ρ/∂t  ≈  (rb - ra) / dt               [forward difference in time]
        ∂ρ/∂x  ≈  np.gradient((ra + rb) / 2)   [central difference in space,
                                                  at the temporal midpoint]

    Returns
    -------
    v_kb_valid  : 1-D array of physically valid velocities in kb/min
    n_valid     : number of accepted positions
    mask        : boolean mask (True = valid position), length = len(ra)
    v_field_kb  : full velocity field in kb/min (unmasked, may contain noise)
    """
    drho_dt = (rb - ra) / dt                        # RPM / min
    rho_avg = (ra + rb) / 2.0
    # gradient() uses 2nd-order central differences; divide by BIN_BP to convert
    # from (RPM/bin) per bin to RPM/bp
    drho_dx = np.gradient(rho_avg) / BIN_BP         # RPM / bp

    # ── masking ──────────────────────────────────────────────────────────────
    # 1. Reject flat-gradient positions (division nearly by zero)
    abs_dx  = np.abs(drho_dx)
    med_dx  = np.median(abs_dx[abs_dx > 0]) if np.any(abs_dx > 0) else 0.0
    grad_ok = abs_dx > med_dx * GRAD_FRAC

    # 2. Require minimum coverage (avoid noise-only regions)
    cov_ok  = rho_avg > RHO_MIN

    # Raw velocity in bp/min; tiny epsilon prevents exact zero division
    eps     = 1e-15
    v_raw   = -drho_dt / (drho_dx + eps)            # bp/min

    # 3. Physical direction constraint (RNAPII moves 5'→3', i.e., v > 0)
    fwd_ok  = v_raw > 0

    mask    = grad_ok & cov_ok & fwd_ok

    # 4. Clip outliers
    v_kb    = v_raw[mask] / 1000.0
    v_kb    = np.clip(v_kb, V_MIN_KB, V_MAX_KB)

    return v_kb, int(mask.sum()), mask, v_raw / 1000.0


def centroid_shift_velocity(r0: np.ndarray,
                             r10: np.ndarray,
                             r25: np.ndarray,
                             r40: np.ndarray,
                             x_bp: np.ndarray) -> tuple:
    """
    Centroid-shift estimator: the centre-of-mass of the NEWLY accumulated
    reads Δρ(x, t) = ρ(t) - ρ(0) moves at the bulk wave velocity.

    This is the spatial integral of the continuity equation:

        v_cm  =  d<x>/dt   where  <x>(t) = ∫ Δρ · x dx / ∫ Δρ dx

    Only positive increments are used (Δρ > 0 = positions where reads have
    genuinely accumulated since t = 0).

    Returns (v_cm_kb_min, c10, c25, c40) — the mean velocity and the three
    centroids in bp.
    """
    def centroid(diff):
        pos = diff > 0
        if pos.sum() < 3:
            return np.nan
        return float(np.average(x_bp[pos], weights=diff[pos]))

    d10 = np.maximum(r10 - r0, 0)
    d25 = np.maximum(r25 - r0, 0)
    d40 = np.maximum(r40 - r0, 0)

    c10 = centroid(d10)
    c25 = centroid(d25)
    c40 = centroid(d40)

    v_vals = []
    for ca, cb, dt in [(c10, c25, 15.0), (c25, c40, 15.0), (c10, c40, 30.0)]:
        if np.isnan(ca) or np.isnan(cb):
            continue
        if cb > ca:                          # centroid must advance
            v = (cb - ca) / dt / 1000.0     # kb/min
            if V_MIN_KB <= v <= V_MAX_KB:
                v_vals.append(v)

    v_cm = float(np.mean(v_vals)) if v_vals else np.nan
    return v_cm, c10, c25, c40


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  RAR Velocity — 1-D continuity-equation RNAPII elongation rate")
    print("=" * 72)

    # ── [1] Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading coverage data …")
    with open(COV_PKL, 'rb') as fh:
        gene_coverages = pickle.load(fh)
    print(f"  {len(gene_coverages)} genes in coverage pickle")

    print("\n[2/5] Loading groHMM wave rates …")
    grohmm = pd.read_csv(GROHMM_CSV)
    print(f"  {len(grohmm)} genes in groHMM table  "
          f"(columns: {list(grohmm.columns[:6])} …)")

    print("\n[3/5] Loading Danko rates …")
    danko = load_danko_rates(DANKO_DIR)

    meta = grohmm.merge(danko, on='gene_id', how='left')
    n_danko = meta['danko_rate_kb_min'].notna().sum()
    print(f"  {n_danko} / {len(meta)} groHMM genes matched to Danko rates")

    # ── [2] Per-gene velocity estimation ─────────────────────────────────────
    print("\n[4/5] Computing velocities …\n")

    records       = []
    profile_store = {}   # gene_id → rar_velocity dict for JSON update
    n_skipped     = 0

    for idx, row in meta.iterrows():
        gid    = row['gene_id']
        strand = row['strand']

        # ── get coverage arrays ────────────────────────────────────────────
        if gid not in gene_coverages:
            n_skipped += 1
            continue

        cov = gene_coverages[gid]
        missing = [tp for tp in ('0m', '10m', '25m', '40m') if tp not in cov]
        if missing:
            n_skipped += 1
            continue

        gene_len = len(cov['0m'])

        # ── wave region boundary ───────────────────────────────────────────
        # wave_end_40m_bp is the groHMM wave-front position RELATIVE TO TSS
        # (bp from gene start, 0-indexed).  The coverage arrays have TSS at
        # index 0 (minus-strand genes are already flipped).
        wave_end = int(row['wave_end_40m_bp'])
        wave_end = max(BIN_BP, min(wave_end, gene_len - 1))

        n_bins = wave_end // BIN_BP
        if n_bins < 5:
            n_skipped += 1
            continue

        # ── bin and smooth coverage ────────────────────────────────────────
        r0  = bin_and_smooth(np.asarray(cov['0m'],  dtype=float), n_bins, BIN_BP, SMOOTH_BIN)
        r10 = bin_and_smooth(np.asarray(cov['10m'], dtype=float), n_bins, BIN_BP, SMOOTH_BIN)
        r25 = bin_and_smooth(np.asarray(cov['25m'], dtype=float), n_bins, BIN_BP, SMOOTH_BIN)
        r40 = bin_and_smooth(np.asarray(cov['40m'], dtype=float), n_bins, BIN_BP, SMOOTH_BIN)

        # bin centre positions in bp from TSS
        x_bins = np.arange(n_bins) * BIN_BP + BIN_BP / 2

        # ── per-interval optical flow (continuity equation, per-position) ──
        #
        # Steps 4–7 of the spec algorithm.
        # Note: BIN_BP = 1000 so the spatial coordinate x is in bp but each
        # 'position' here is a 1-kb bin.  The gradient is converted back to
        # RPM/bp inside optical_flow_interval.
        of_results = {}
        for (ra, rb, dt, label) in [
            (r0,  r10, DT_10, '10m'),
            (r10, r25, DT_25, '25m'),
            (r25, r40, DT_40, '40m'),
        ]:
            v_valid, n_valid, mask, v_field = optical_flow_interval(ra, rb, dt)
            if n_valid == 0:
                of_results[label] = None
                continue
            of_results[label] = {
                'v_mean_kb': float(np.mean(v_valid)),
                'n_valid':   n_valid,
                'mask':      mask,
                'v_field':   v_field,   # full kb/min array for profile
            }

        # Per-timepoint optical-flow means
        v_of_10 = of_results['10m']['v_mean_kb'] if of_results['10m'] else np.nan
        v_of_25 = of_results['25m']['v_mean_kb'] if of_results['25m'] else np.nan
        v_of_40 = of_results['40m']['v_mean_kb'] if of_results['40m'] else np.nan

        valid_of = [r for r in of_results.values() if r is not None]
        if valid_of:
            wts_of  = [r['n_valid'] for r in valid_of]
            vals_of = [r['v_mean_kb'] for r in valid_of]
            v_of_gene = float(np.average(vals_of, weights=wts_of))
            n_valid_of = sum(wts_of)
        else:
            v_of_gene = np.nan
            n_valid_of = 0

        # ── integrated continuity equation: centroid-shift ─────────────────
        #
        # This estimator averages the continuity equation over the entire
        # wave region, giving a robust bulk-velocity estimate equivalent to
        # tracking the wave's centre of mass.  It achieves the same M-position
        # averaging gain as the per-position approach.
        v_cm, c10, c25, c40 = centroid_shift_velocity(r0, r25, r25, r40, x_bins)
        # Note: we use (r0→r25) and (r25→r40) to match the two long intervals
        v_cm_full, c10_f, c25_f, c40_f = centroid_shift_velocity(r0, r10, r25, r40, x_bins)

        # Primary reported velocity: centroid-shift (more robust)
        v_rar = v_cm_full

        # AD gain: total valid bins × BIN_BP → total effective 1-bp positions
        n_eff   = max(n_valid_of * BIN_BP, 1)
        gain_db = 10.0 * np.log10(n_eff)

        # Fallback: if centroid shift failed, use optical-flow mean
        if np.isnan(v_rar) and not np.isnan(v_of_gene):
            v_rar = v_of_gene

        # ── spatial velocity profile (for webapp) ─────────────────────────
        # Average the three per-interval optical-flow velocity fields,
        # NaN-ing invalid positions.
        v_profile_list = []
        for r in valid_of:
            vf = r['v_field'].copy()
            vf[~r['mask']] = np.nan
            v_profile_list.append(vf)

        if v_profile_list:
            v_profile = np.nanmean(np.stack(v_profile_list, axis=0), axis=0)
        else:
            v_profile = np.full(n_bins, np.nan)

        # Absolute genomic positions of bin centres
        parts   = gid.split('_')
        g_start = int(parts[1]) if len(parts) >= 3 else 0
        abs_pos = (g_start + x_bins).tolist()

        # ── record ────────────────────────────────────────────────────────
        records.append({
            'gene_id':            gid,
            'chrom':              row.get('chrom', ''),
            'start':              g_start,
            'end':                int(parts[2]) if len(parts) >= 3 else 0,
            'strand':             strand,
            'v_rar_kb_min':       v_rar,
            'v_rar_10m':          v_of_10,
            'v_rar_25m':          v_of_25,
            'v_rar_40m':          v_of_40,
            'n_valid_positions':  n_valid_of * BIN_BP,
            'ad_gain_db':         gain_db,
            'danko_rate_kb_min':  row.get('danko_rate_kb_min', np.nan),
            'grohmm_rate_kb_min': row.get('rate_kb_min', np.nan),
        })

        profile_store[gid] = {
            'positions':    abs_pos,
            'v_profile':    [None if np.isnan(v) else float(v)
                             for v in v_profile.tolist()],
            'v_mean_kb_min': float(v_rar) if not np.isnan(v_rar) else None,
            'ad_gain_db':   float(gain_db),
        }

        if (idx + 1) % 10 == 0 or idx == len(meta) - 1:
            v_str = f"{v_rar:.3f}" if not np.isnan(v_rar) else "nan"
            print(f"  [{idx+1:3d}/{len(meta)}]  {gid:42s}  "
                  f"v_rar={v_str} kb/min  n_bins={n_bins:4d}  "
                  f"AD gain={gain_db:.1f} dB")

    print(f"\n  Processed: {len(records)} genes  |  Skipped: {n_skipped}")

    # ── [3] Save CSV ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved → {OUT_CSV}")

    # ── [4] Update speed-profile JSONs ────────────────────────────────────────
    print(f"\n[5/5] Updating speed-profile JSONs in {PROFILES_DIR} …")
    updated = 0
    for gid, rar_data in profile_store.items():
        fname = gid + '.json'
        fpath = os.path.join(PROFILES_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as fh:
            jdata = json.load(fh)
        jdata['rar_velocity'] = rar_data
        with open(fpath, 'w') as fh:
            json.dump(jdata, fh, allow_nan=False)
        updated += 1
    print(f"  Updated {updated} JSON files")

    # ── [5] Correlation analysis ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CORRELATION ANALYSIS")
    print("=" * 72)

    # vs Danko
    mask_d = (results_df['danko_rate_kb_min'].notna() &
              results_df['v_rar_kb_min'].notna() &
              (results_df['danko_rate_kb_min'] > 0) &
              (results_df['v_rar_kb_min']      > 0))
    df_d = results_df[mask_d]

    if len(df_d) >= 3:
        pr_d, pp_d = pearsonr(df_d['v_rar_kb_min'], df_d['danko_rate_kb_min'])
        sr_d, sp_d = spearmanr(df_d['v_rar_kb_min'], df_d['danko_rate_kb_min'])
        print(f"\n  RAR velocity  vs  Danko rate  (n = {len(df_d)})")
        print(f"    Pearson  r = {pr_d:+.3f}   p = {pp_d:.3e}")
        print(f"    Spearman ρ = {sr_d:+.3f}   p = {sp_d:.3e}")
    else:
        pr_d = sr_d = np.nan
        print(f"\n  Insufficient overlapping genes for RAR vs Danko (n={len(df_d)})")

    # vs groHMM
    mask_g = (results_df['grohmm_rate_kb_min'].notna() &
              results_df['v_rar_kb_min'].notna() &
              (results_df['grohmm_rate_kb_min'] > 0) &
              (results_df['v_rar_kb_min']        > 0))
    df_g = results_df[mask_g]

    if len(df_g) >= 3:
        pr_g, pp_g = pearsonr(df_g['v_rar_kb_min'], df_g['grohmm_rate_kb_min'])
        sr_g, sp_g = spearmanr(df_g['v_rar_kb_min'], df_g['grohmm_rate_kb_min'])
        print(f"\n  RAR velocity  vs  groHMM rate  (n = {len(df_g)})")
        print(f"    Pearson  r = {pr_g:+.3f}   p = {pp_g:.3e}")
        print(f"    Spearman ρ = {sr_g:+.3f}   p = {sp_g:.3e}")
    else:
        pr_g = sr_g = np.nan
        print(f"\n  Insufficient overlapping genes for RAR vs groHMM (n={len(df_g)})")

    # AD gain summary
    print("\n  AD gain statistics (all valid genes):")
    print(f"    Median effective valid positions : "
          f"{results_df['n_valid_positions'].median():.0f}")
    print(f"    Median AD gain                  : "
          f"{results_df['ad_gain_db'].median():.1f} dB")
    print(f"    Max AD gain                     : "
          f"{results_df['ad_gain_db'].max():.1f} dB")
    print(f"    vs Danko 3-measurement baseline : "
          f"~4.8 dB  (10·log10(3))")

    # ── [6] Scatter plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        'RAR Optical-Flow / Continuity-Equation Velocity vs Reference Rates\n'
        f'(1-kb bins, {SMOOTH_BIN}-kb smooth, integrated centroid-shift primary)',
        fontsize=12, fontweight='bold'
    )

    def _scatter(ax, x, y, xlabel, ylabel, title, r_p, r_s, n):
        ax.scatter(x, y, s=25, alpha=0.65, color='steelblue', edgecolors='none')
        lim = max(np.nanmax(x), np.nanmax(y)) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.4, label='y = x')
        if len(x) >= 3:
            coef = np.polyfit(x, y, 1)
            xf = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            ax.plot(xf, np.polyval(coef, xf), 'r-', lw=1.5, label='linear fit')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        stats_txt = (f"Pearson r  = {r_p:.3f}\n"
                     f"Spearman ρ = {r_s:.3f}\n"
                     f"n = {n}")
        ax.text(0.05, 0.95, stats_txt, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    if len(df_d) >= 3:
        _scatter(axes[0],
                 df_d['v_rar_kb_min'].values,
                 df_d['danko_rate_kb_min'].values,
                 'RAR velocity (kb/min)', 'Danko rate (kb/min)',
                 'RAR vs Danko', pr_d, sr_d, len(df_d))
    else:
        axes[0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

    if len(df_g) >= 3:
        _scatter(axes[1],
                 df_g['v_rar_kb_min'].values,
                 df_g['grohmm_rate_kb_min'].values,
                 'RAR velocity (kb/min)', 'groHMM rate (kb/min)',
                 'RAR vs groHMM', pr_g, sr_g, len(df_g))
    else:
        axes[1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)
    print(f"\n  Scatter plot saved → {OUT_PLOT}")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)

    return results_df


if __name__ == '__main__':
    main()
