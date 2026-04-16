"""
Script 11: Temporal Rephasing — Genomic Position → Transcription Time
=====================================================================
Purpose: Convert epigenomic signals from genomic-position space (bp from TSS)
to transcription-time space (minutes from transcription start).

The coordinate transform is:

    t(p) = ∫₀ᵖ dp' / v(p')   [minutes]

where v(p) is the instantaneous speed in bp/min.

Because v(p) is measured on a sparse grid (positions_bp may skip bp), the
integral is computed as:

    t[i] = Σ_{j=0}^{i-1}  Δp_j / v_j          (trapezoidal, Δp_j = p_{j+1} - p_j)

Then t is normalised so that t(wave_end_40m) = 40 min.

All epigenomic mark arrays (from results/epigenomics/) are then interpolated
from their native 1 bp grid onto a uniform time grid of 500 points spanning
[0, 40] min.

Outputs
-------
    results/temporal_profiles/<gene_id>.json
        {
          "gene_id":        str,
          "time_grid_min":  [0.0, 0.08, ..., 40.0],   // 500 points
          "speed_at_time":  [...],                     // v(t) in kb/min
          "marks": {
              "<mark>": [...],   // signal interpolated onto time grid
              ...
          }
        }

Usage
-----
    python scripts/11_temporal_rephase.py
(run from project root)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import scipy.interpolate
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
GENE_INDEX        = PROJECT_ROOT / "results" / "speed_profiles" / "_gene_index.json"
SPEED_PROFILE_DIR = PROJECT_ROOT / "results" / "speed_profiles"
EPI_DIR           = PROJECT_ROOT / "results" / "epigenomics"
EPI_INDEX         = EPI_DIR / "_epi_index.json"
OUT_DIR           = PROJECT_ROOT / "results" / "temporal_profiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TIME_POINTS  = 500
TIME_MAX_MIN   = 40.0
TIME_GRID      = np.linspace(0.0, TIME_MAX_MIN, N_TIME_POINTS)

PROFILE_KEY    = "A_raw40m"   # which speed-profile approach to use


# ---------------------------------------------------------------------------
# Speed profile loading
# ---------------------------------------------------------------------------
def load_speed_profile(gene: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Returns (positions_bp, speed_bp_min) arrays for the gene, using
    PROFILE_KEY.  Returns None if the profile cannot be loaded.
    """
    profile_path = SPEED_PROFILE_DIR / gene["file"]
    if not profile_path.exists():
        warnings.warn(f"Speed profile not found: {profile_path}")
        return None

    with open(profile_path) as f:
        data = json.load(f)

    profiles = data.get("profiles", {})
    if PROFILE_KEY not in profiles:
        available = list(profiles.keys())
        warnings.warn(
            f"Profile key '{PROFILE_KEY}' not in {gene['gene_id']}. "
            f"Available: {available}.  Skipping."
        )
        return None

    p   = data["profiles"][PROFILE_KEY]
    pos = np.array(p["positions_bp"], dtype=float)      # bp from TSS
    spd = np.array(p["speed_kb_min"], dtype=float)      # kb/min

    # convert to bp/min
    spd_bp_min = spd * 1000.0

    # clamp speeds to a minimum to avoid division-by-zero
    spd_bp_min = np.where(spd_bp_min < 1e-6, 1e-6, spd_bp_min)

    return pos, spd_bp_min


# ---------------------------------------------------------------------------
# Temporal transform
# ---------------------------------------------------------------------------
def genomic_to_time(pos_bp: np.ndarray, speed_bp_min: np.ndarray,
                    wave_40m: float | None) -> np.ndarray:
    """
    Compute t(p) via cumulative trapezoidal integration of 1/v(p).

    If wave_40m is provided, normalise so t(wave_40m) = 40 min.
    Otherwise normalise so t(pos_bp[-1]) = 40 min.
    """
    # segment lengths between consecutive positions
    delta_p = np.diff(pos_bp)                  # length n-1
    inv_v   = 1.0 / speed_bp_min               # 1/(bp/min) = min/bp

    # trapezoidal: use average speed over each segment
    avg_inv_v = 0.5 * (inv_v[:-1] + inv_v[1:])

    # cumulative time at each position (starts at 0)
    t = np.zeros(len(pos_bp))
    t[1:] = np.cumsum(delta_p * avg_inv_v)

    # normalise
    if wave_40m is not None and wave_40m > 0:
        # find the time at wave_40m (interpolate)
        if wave_40m <= pos_bp[-1]:
            t_at_wave = float(np.interp(wave_40m, pos_bp, t))
        else:
            t_at_wave = t[-1]
    else:
        t_at_wave = t[-1]

    if t_at_wave > 0:
        t = t * (TIME_MAX_MIN / t_at_wave)

    return t


# ---------------------------------------------------------------------------
# Epigenomic mark discovery
# ---------------------------------------------------------------------------
def discover_marks() -> list[str]:
    """
    Returns a sorted list of unique mark names inferred from the epi index
    or from the numpy files present in EPI_DIR.
    """
    if EPI_INDEX.exists():
        with open(EPI_INDEX) as f:
            idx = json.load(f)
        # top-level keys are mark names (except entries starting with "_")
        return sorted(k for k in idx if not k.startswith("_"))

    # fallback: infer from filenames  <gene_id>_<mark>.npy
    marks = set()
    for p in EPI_DIR.glob("*.npy"):
        # gene_id may contain underscores, but mark is the last segment
        # gene_id format: chr<chrom>_<start>_<end>_<strand>
        # so we split off the last component after the 4th underscore
        parts = p.stem.split("_")
        # gene_id has exactly 4 parts: chrX, start, end, strand
        # mark is everything after
        mark = "_".join(parts[4:])
        if mark:
            marks.add(mark)
    return sorted(marks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading gene index from {GENE_INDEX}")
    with open(GENE_INDEX) as f:
        genes = json.load(f)
    print(f"  {len(genes)} genes loaded.")

    marks = discover_marks()
    if not marks:
        warnings.warn(
            "No epigenomic marks found.  Run script 10 first.  "
            "Temporal profiles will be written without mark data."
        )
    else:
        print(f"  {len(marks)} epigenomic marks: {marks}")

    n_ok = 0
    n_skip = 0

    for gene in tqdm(genes, desc="Rephasing genes", unit="gene"):
        gene_id  = gene["gene_id"]
        wave_40m = gene.get("wave_40m", None)   # may not be present in index

        # --- load wave_40m from speed profile if not in index ----------
        profile_path = SPEED_PROFILE_DIR / gene["file"]
        if wave_40m is None and profile_path.exists():
            with open(profile_path) as f:
                _data = json.load(f)
            wave_40m = _data.get("wave_40m", None)

        # --- load speed profile ----------------------------------------
        result = load_speed_profile(gene)
        if result is None:
            n_skip += 1
            continue
        pos_bp, speed_bp_min = result

        # --- compute time coordinate -----------------------------------
        t_of_p = genomic_to_time(pos_bp, speed_bp_min, wave_40m)

        # --- interpolate speed onto uniform time grid ------------------
        speed_kb_min = speed_bp_min / 1000.0
        # clamp t_of_p to [0, TIME_MAX_MIN] for safe interpolation
        t_clamp = np.clip(t_of_p, 0.0, TIME_MAX_MIN)
        # build interpolator (extrapolate with edge values)
        spd_interp_fn = scipy.interpolate.interp1d(
            t_clamp, speed_kb_min,
            kind="linear", bounds_error=False,
            fill_value=(speed_kb_min[0], speed_kb_min[-1])
        )
        speed_at_time = spd_interp_fn(TIME_GRID)

        # --- interpolate epigenomic marks onto time grid ---------------
        marks_on_time: dict[str, list] = {}

        for mark in marks:
            epi_path = EPI_DIR / f"{gene_id}_{mark}.npy"
            if not epi_path.exists():
                marks_on_time[mark] = [None] * N_TIME_POINTS
                continue

            signal_pos = np.load(epi_path)   # 1 bp resolution, length = end - start

            # The signal array is indexed by bp-from-TSS (0 … gene_len-1)
            # We need to map positions_bp (which may not be 0,1,2,…) to this.
            # But pos_bp from the speed profile is also bp-from-TSS at sparse points.
            # Build a dense 0-based bp grid matching the signal array length.
            gene_len = len(signal_pos)
            dense_pos = np.arange(gene_len, dtype=float)

            # Compute t for each bp in [0, gene_len) by interpolating from
            # the sparse (pos_bp, t_of_p) mapping.
            t_dense = np.interp(dense_pos, pos_bp, t_of_p)
            t_dense = np.clip(t_dense, 0.0, TIME_MAX_MIN)

            # Replace NaN in signal with nearest finite value
            sig = signal_pos.astype(float)
            nan_mask = np.isnan(sig)
            if nan_mask.all():
                marks_on_time[mark] = [np.nan] * N_TIME_POINTS
                continue
            if nan_mask.any():
                finite_idx = np.where(~nan_mask)[0]
                nan_idx    = np.where(nan_mask)[0]
                nearest    = finite_idx[np.searchsorted(finite_idx,
                                                         nan_idx).clip(0, len(finite_idx)-1)]
                sig[nan_idx] = sig[nearest]

            # Interpolate signal onto TIME_GRID via t_dense
            sig_interp_fn = scipy.interpolate.interp1d(
                t_dense, sig,
                kind="linear", bounds_error=False,
                fill_value=(sig[0], sig[-1])
            )
            marks_on_time[mark] = sig_interp_fn(TIME_GRID).tolist()

        # --- write output ---------------------------------------------
        out = {
            "gene_id":       gene_id,
            "time_grid_min": TIME_GRID.tolist(),
            "speed_at_time": speed_at_time.tolist(),
            "marks":         marks_on_time,
            "_meta": {
                "profile_key":    PROFILE_KEY,
                "n_time_points":  N_TIME_POINTS,
                "time_max_min":   TIME_MAX_MIN,
                "wave_40m_bp":    wave_40m,
                "pos_bp_range":   [float(pos_bp[0]), float(pos_bp[-1])],
            },
        }
        out_path = OUT_DIR / f"{gene_id}.json"
        with open(out_path, "w") as fout:
            json.dump(out, fout, indent=2)
        n_ok += 1

    print(f"\nDone.  Wrote {n_ok} temporal profiles, skipped {n_skip}.")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
