"""
Script 10: Epigenomic Signal Extraction Along Gene Bodies
=========================================================
Purpose: For each gene in the index and each bigWig file in data/epigenomics/,
extract the ChIP-seq / ATAC / methylation signal at 1 bp resolution over the
gene body (TSS → gene end).  Strand is handled so that position 0 always
corresponds to the TSS regardless of strand.

COORDINATE HANDLING
-------------------
GRO-Seq gene coordinates are in hg19.  ENCODE bigWigs are in hg38/GRCh38.
This script automatically detects the mismatch and uses pyliftover to convert
hg19 gene boundaries → hg38, extracts signal from the hg38 bigWig, then
resamples back to the original hg19 gene length so all arrays are compatible
with the speed profiles.

Outputs
-------
  results/epigenomics/<gene_id>_<mark>.npy   — 1-D numpy array, one value per bp
  results/epigenomics/_epi_index.json        — summary: file list, mean signal,
                                               Pearson r vs grohmm_rate and danko_rate

Usage
-----
    python scripts/10_epigenomic_profiles.py
(run from project root)

Dependencies: pyBigWig, pyliftover, numpy, scipy, tqdm
"""

import json
import warnings
from pathlib import Path

import numpy as np
import scipy.stats
import scipy.interpolate
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Liftover helper (hg19 → hg38)
# ---------------------------------------------------------------------------
_liftover_cache = {}

def _get_liftover(src='hg19', tgt='hg38'):
    """Return a cached LiftOver object, downloading the chain file if needed."""
    key = (src, tgt)
    if key not in _liftover_cache:
        try:
            from pyliftover import LiftOver
            print(f"  Loading liftover chain {src} → {tgt} (downloads on first use)…")
            _liftover_cache[key] = LiftOver(src, tgt)
            print("  Chain loaded.")
        except ImportError:
            print("  pyliftover not installed — install with: pip install pyliftover")
            _liftover_cache[key] = None
        except Exception as e:
            print(f"  Could not load liftover chain: {e}")
            _liftover_cache[key] = None
    return _liftover_cache[key]


def liftover_region(chrom, start, end, src='hg19', tgt='hg38'):
    """
    Convert a region (chrom, start, end) from src to tgt assembly.
    Returns (new_chrom, new_start, new_end) or None if conversion fails.
    Uses the start and end boundary points for efficiency.
    """
    lo = _get_liftover(src, tgt)
    if lo is None:
        return None
    try:
        r_start = lo.convert_coordinate(chrom, start)
        r_end   = lo.convert_coordinate(chrom, end - 1)  # end is exclusive
        if not r_start or not r_end:
            return None
        new_chrom  = r_start[0][0]
        new_start  = int(r_start[0][1])
        new_end    = int(r_end[0][1]) + 1
        # Ensure start < end (strand flip can invert)
        if new_start > new_end:
            new_start, new_end = new_end, new_start
        return (new_chrom, new_start, new_end)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENE_INDEX   = PROJECT_ROOT / "results" / "speed_profiles" / "_gene_index.json"
EPI_DATA_DIR = PROJECT_ROOT / "data" / "epigenomics"
EPI_META     = EPI_DATA_DIR / "metadata.json"
OUT_DIR      = PROJECT_ROOT / "results" / "epigenomics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: load pyBigWig lazily so the script still imports without it
# ---------------------------------------------------------------------------
def _require_pybigwig():
    try:
        import pyBigWig
        return pyBigWig
    except ImportError:
        raise SystemExit(
            "pyBigWig is required.  Install with:  pip install pyBigWig"
        )


# ---------------------------------------------------------------------------
# Epigenomic signal extraction
# ---------------------------------------------------------------------------
def extract_signal(bw, chrom: str, start: int, end: int, strand: str,
                   genome_build_bw: str = 'unknown',
                   target_len: int = None) -> np.ndarray:
    """
    Return a float32 array with signal values, always of length target_len
    (defaults to end - start).

    If the bigWig is hg38 and gene coords are hg19, lifts over the region
    to hg38 before querying, then resamples back to target_len so arrays
    are compatible with hg19-space speed profiles.

    pyBigWig.values() returns None for positions with no data — those are
    replaced with np.nan then filled with local (±1 kb) median.
    """
    orig_len = end - start
    n = target_len if target_len else orig_len

    query_chrom, query_start, query_end = chrom, start, end

    # --- coordinate liftover for hg38 bigWigs -------------------------
    if genome_build_bw == 'hg38':
        result = liftover_region(chrom, start, end, 'hg19', 'hg38')
        if result is not None:
            query_chrom, query_start, query_end = result
        else:
            return np.full(n, np.nan, dtype=np.float32)

    try:
        raw = bw.values(query_chrom, query_start, query_end, numpy=True)
    except RuntimeError:
        return np.full(n, np.nan, dtype=np.float32)

    if raw is None or len(raw) == 0:
        return np.full(n, np.nan, dtype=np.float32)

    arr = np.asarray(raw, dtype=np.float32)

    # --- fill NaN with local median (window ±1 kb) --------------------
    nan_mask = np.isnan(arr)
    if nan_mask.any() and not nan_mask.all():
        window = 1000
        for idx in np.where(nan_mask)[0]:
            lo_i = max(0, idx - window)
            hi_i = min(len(arr), idx + window)
            local = arr[lo_i:hi_i]
            finite = local[np.isfinite(local)]
            arr[idx] = np.nanmedian(finite) if len(finite) > 0 else 0.0

    # --- flip minus-strand so index 0 = TSS ---------------------------
    if strand == "-":
        arr = arr[::-1].copy()

    # --- resample to target length if lifted region changed size ------
    if len(arr) != n:
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, n)
        arr = scipy.interpolate.interp1d(x_old, arr, kind='linear',
                                          fill_value='extrapolate')(x_new).astype(np.float32)

    return arr


def detect_genome_build(bw) -> str:
    """
    Heuristic: if the bigWig header lists chr1 length ≈ 248 Mb → hg19,
    ≈ 249 Mb → hg38.  Returns 'hg19', 'hg38', or 'unknown'.
    """
    chroms = bw.chroms()
    chr1_len = chroms.get("chr1", chroms.get("1", None))
    if chr1_len is None:
        return "unknown"
    if 248_900_000 <= chr1_len <= 249_000_000:
        return "hg19"
    if 249_200_000 <= chr1_len <= 249_400_000:
        return "hg38"
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pyBigWig = _require_pybigwig()

    # --- load gene index ------------------------------------------------
    print(f"Loading gene index from {GENE_INDEX}")
    with open(GENE_INDEX) as f:
        genes = json.load(f)
    print(f"  {len(genes)} genes loaded.")

    # --- discover bigWig files -----------------------------------------
    bw_files: dict[str, Path] = {}   # mark_name → Path

    if EPI_META.exists():
        with open(EPI_META) as f:
            meta = json.load(f)
        # Support two formats:
        #   Format A (download script): {"tracks": {"mark": {"local_file": ..., "status": ...}}}
        #   Format B (legacy):          [{"mark": ..., "file": ...}]
        tracks = meta.get("tracks", meta) if isinstance(meta, dict) else {}
        if isinstance(tracks, dict):
            for mark_name, entry in tracks.items():
                if isinstance(entry, dict):
                    if entry.get("status") == "not_found":
                        print(f"  Skipping {mark_name}: not_found in metadata")
                        continue
                    local = entry.get("local_file")
                    if local:
                        bw_files[mark_name] = Path(local)
        elif isinstance(meta, list):
            for entry in meta:
                name = entry.get("mark") or entry.get("name") or Path(entry["file"]).stem
                bw_files[name] = EPI_DATA_DIR / entry["file"]
        print(f"Loaded {len(bw_files)} bigWig entries from metadata.json")
    else:
        warnings.warn(
            f"metadata.json not found at {EPI_META}. "
            "Auto-detecting bigWig files in data/epigenomics/ ..."
        )
        if not EPI_DATA_DIR.exists():
            raise SystemExit(
                f"Epigenomics data directory does not exist: {EPI_DATA_DIR}\n"
                "Please run the ENCODE download script first."
            )
        for bw_path in sorted(EPI_DATA_DIR.glob("**/*.bw")) + sorted(EPI_DATA_DIR.glob("**/*.bigWig")):
            mark_name = bw_path.parent.name  # use directory name as mark
            bw_files[mark_name] = bw_path
        print(f"Auto-detected {len(bw_files)} bigWig files.")

    if not bw_files:
        raise SystemExit("No bigWig files found.  Aborting.")

    # --- per-gene extraction -------------------------------------------
    # epi_index structure:
    #   { mark_name: { gene_id: { "file": str, "mean": float, "n_nan_frac": float } } }
    epi_index: dict[str, dict] = {}

    # We also accumulate per-mark arrays of mean signal for correlation
    per_mark_means: dict[str, list[float]] = {m: [] for m in bw_files}
    gene_grohmm    = [g["grohmm_rate_kb_min"] for g in genes]
    gene_danko     = [g["danko_rate_kb_min"]  for g in genes]

    for mark_name, bw_path in bw_files.items():
        print(f"\n--- Processing mark: {mark_name}  ({bw_path.name}) ---")
        epi_index[mark_name] = {}

        if not bw_path.exists():
            warnings.warn(f"  bigWig not found: {bw_path} — skipping mark.")
            per_mark_means[mark_name] = [np.nan] * len(genes)
            continue

        bw = pyBigWig.open(str(bw_path))
        genome_build = detect_genome_build(bw)
        if genome_build == "hg38":
            warnings.warn(
                f"  {mark_name}: bigWig appears to be hg38.  "
                "Gene coordinates are hg19.  Signal will be misaligned. "
                "Run liftover before using these arrays."
            )
        elif genome_build == "unknown":
            warnings.warn(f"  {mark_name}: could not determine genome build.")

        mark_means: list[float] = []

        for gene in tqdm(genes, desc=mark_name, unit="gene", leave=False):
            gene_id = gene["gene_id"]
            chrom   = gene["chrom"]
            start   = int(gene["start"])
            end     = int(gene["end"])
            strand  = gene["strand"]

            arr = extract_signal(bw, chrom, start, end, strand,
                                genome_build_bw=genome_build,
                                target_len=(end - start))

            # save
            out_path = OUT_DIR / f"{gene_id}_{mark_name}.npy"
            np.save(out_path, arr)

            mean_val   = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else np.nan
            nan_frac   = float(np.isnan(arr).mean())

            epi_index[mark_name][gene_id] = {
                "file":      str(out_path.relative_to(PROJECT_ROOT)),
                "length_bp": len(arr),
                "mean":      mean_val,
                "nan_frac":  nan_frac,
                "genome_build_bw": genome_build,
            }
            mark_means.append(mean_val)

        bw.close()
        per_mark_means[mark_name] = mark_means

    # --- correlation table -------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Mark':<30}  {'r vs grohmm':>12}  {'p grohmm':>10}  "
          f"{'r vs danko':>11}  {'p danko':>10}")
    print("-" * 70)

    for mark_name in bw_files:
        means = np.array(per_mark_means[mark_name], dtype=float)
        valid = np.isfinite(means) & np.isfinite(gene_grohmm) & np.isfinite(gene_danko)
        if valid.sum() < 3:
            print(f"{mark_name:<30}  {'N/A':>12}  {'N/A':>10}  {'N/A':>11}  {'N/A':>10}")
            continue
        r_g, p_g = scipy.stats.pearsonr(means[valid], np.array(gene_grohmm)[valid])
        r_d, p_d = scipy.stats.pearsonr(means[valid], np.array(gene_danko)[valid])
        print(f"{mark_name:<30}  {r_g:>12.4f}  {p_g:>10.4f}  {r_d:>11.4f}  {p_d:>10.4f}")
        # store in index
        for mark_name2, d in epi_index.items():
            if mark_name2 == mark_name:
                epi_index[mark_name]["_correlations"] = {
                    "pearson_r_grohmm": r_g,
                    "p_grohmm":         p_g,
                    "pearson_r_danko":  r_d,
                    "p_danko":          p_d,
                    "n_genes":          int(valid.sum()),
                }
    print("=" * 70)

    # --- write summary index ------------------------------------------
    index_path = OUT_DIR / "_epi_index.json"
    with open(index_path, "w") as f:
        json.dump(epi_index, f, indent=2)
    print(f"\nSummary index written to {index_path}")
    print("Done.")


if __name__ == "__main__":
    main()
