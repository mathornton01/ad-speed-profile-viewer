"""
Script 13: RNA Folding Free Energy (MFE) Along Nascent Transcripts
==================================================================
Purpose: For each gene, extract the genomic sequence of the nascent transcript
at increasing lengths and compute the minimum free energy (MFE) of RNA folding
using ViennaRNA (RNA.fold).

The MFE profile ΔG(p) tells us how thermodynamically stable the nascent RNA is
as a function of the distance p from the TSS.  Comparing ΔG(p) to v(p) (from
script 07) gives insight into whether RNA secondary structure influences RNAPII
elongation speed.

Algorithm
---------
For each gene g:
  For p in range(1000, wave_end_40m, step=1000):
    seq = genome[TSS : TSS+p]   (sense strand, 5'→3' w.r.t transcript)
    seq = seq[:-12]             (remove 12 nt exit-channel = protected region)
    ΔG = RNA.fold(seq.replace('T','U'))[1]
    append (p, ΔG)

Checkpoint: each gene's output is written immediately; if the file already
exists and contains a valid result the gene is skipped on re-run.

Dependencies
------------
  Required:  biopython (Bio.SeqIO / Bio.Seq), tqdm
  Optional:  ViennaRNA Python bindings (`import RNA`)

If ViennaRNA is absent the script exits with a clear install message.
If biopython is absent the script exits similarly.
The hg19 genome FASTA must be available; the script searches several likely
locations (see GENOME_SEARCH_PATHS below).

Outputs
-------
    results/folding/<gene_id>_mfe.json:
        {
          "gene_id":    "...",
          "chrom":      "chrX",
          "strand":     "+",
          "tss":        123456,
          "step_bp":    1000,
          "positions_bp": [1000, 2000, ...],
          "mfe_kcal_mol": [-5.2, -8.7, ...]
        }

Usage
-----
    python scripts/13_rna_folding.py
(run from project root)

Performance: ~1–5 s per position per gene.  Use --max-genes N for testing.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DATA_ROOT       = PROJECT_ROOT / "data"
GENE_INDEX      = PROJECT_ROOT / "results" / "speed_profiles" / "_gene_index.json"
SPEED_PROF_DIR  = PROJECT_ROOT / "results" / "speed_profiles"
OUT_DIR         = PROJECT_ROOT / "results" / "folding"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEP_BP           = 1000     # sample every 1 kb
EXIT_CHANNEL_NT   = 12       # remove last 12 nt (RNA exit channel)
MIN_SEQ_LEN       = 50       # don't fold sequences shorter than this

# Candidate paths for hg19 FASTA
GENOME_SEARCH_PATHS = [
    DATA_ROOT / "hg19.fa",
    DATA_ROOT / "hg19.fasta",
    DATA_ROOT / "genome" / "hg19.fa",
    DATA_ROOT / "genome" / "hg19.fasta",
    Path("/home/mathornton") / "hg19.fa",
    Path("/home/mathornton") / "hg19.fasta",
    Path("/data/genomes/hg19/hg19.fa"),
    Path("/ref/hg19/hg19.fa"),
]


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
def require_viennarna():
    try:
        import RNA
        return RNA
    except ImportError:
        sys.exit(
            "\nViennaRNA Python bindings not found.\n"
            "Install with:\n"
            "  conda install -c bioconda viennarna\n"
            "or\n"
            "  pip install ViennaRNA\n"
            "(May require a build from source if no wheel is available for your platform.)\n"
        )


def require_biopython():
    try:
        from Bio import SeqIO
        from Bio.Seq import Seq
        return SeqIO, Seq
    except ImportError:
        sys.exit(
            "\nBiopython not found.\n"
            "Install with:  pip install biopython\n"
        )


# ---------------------------------------------------------------------------
# Genome loading
# ---------------------------------------------------------------------------
def find_genome_fasta() -> Path | None:
    for p in GENOME_SEARCH_PATHS:
        if p.exists():
            return p
    return None


def load_genome(SeqIO, fasta_path: Path) -> dict:
    """
    Load all chromosomes into a dict { chrom_name: str }.
    For hg19 the keys are 'chr1', 'chr2', …, 'chrX', 'chrY', 'chrM'.
    """
    print(f"Loading genome from {fasta_path}  (this may take 1-2 minutes)…")
    genome = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        chrom = rec.id.split()[0]  # strip anything after whitespace
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom
        genome[chrom] = str(rec.seq).upper()
    print(f"  Loaded {len(genome)} chromosomes.")
    return genome


# ---------------------------------------------------------------------------
# Sequence extraction
# ---------------------------------------------------------------------------
def extract_transcript_seq(genome: dict, chrom: str, tss: int, end: int,
                            strand: str, length_bp: int) -> str | None:
    """
    Return the RNA sequence (as DNA string, T not U) of the nascent transcript
    from the TSS up to TSS+length_bp, on the sense strand.

    Raises ValueError if coordinates are out of range.
    Returns None if chromosome not found.
    """
    if chrom not in genome:
        return None
    chrom_seq = genome[chrom]

    if strand == "+":
        seq_start = tss
        seq_end   = tss + length_bp
        seq = chrom_seq[seq_start:seq_end]
    else:
        # TSS is the 3′ end of the minus-strand gene (i.e., gene["end"] in 0-based)
        # The nascent transcript runs from TSS downstream in the 3′→5′ genomic direction
        # length_bp bases downstream of TSS on minus strand = genomic positions (tss - length_bp, tss)
        seq_end   = tss          # genomic coord of TSS (end of gene in 0-based BED)
        seq_start = tss - length_bp
        if seq_start < 0:
            return None
        seq = chrom_seq[seq_start:seq_end]
        # reverse complement to get 5'→3' RNA sense
        seq = seq[::-1].translate(str.maketrans("ACGT", "TGCA"))

    return seq if seq else None


# ---------------------------------------------------------------------------
# MFE computation
# ---------------------------------------------------------------------------
def compute_mfe(RNA, seq_dna: str) -> float:
    """Fold sequence, return MFE in kcal/mol."""
    seq_rna = seq_dna.replace("T", "U").replace("t", "u")
    _structure, mfe = RNA.fold(seq_rna)
    return float(mfe)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute RNA MFE profiles per gene.")
    parser.add_argument("--max-genes", type=int, default=None,
                        help="Limit to first N genes (for testing).")
    parser.add_argument("--genome", type=str, default=None,
                        help="Path to hg19 genome FASTA (overrides auto-search).")
    args = parser.parse_args()

    # --- dependency checks ------------------------------------------
    RNA        = require_viennarna()
    SeqIO, Seq = require_biopython()

    # --- genome FASTA -----------------------------------------------
    if args.genome:
        genome_path = Path(args.genome)
        if not genome_path.exists():
            sys.exit(f"Genome FASTA not found: {genome_path}")
    else:
        genome_path = find_genome_fasta()
        if genome_path is None:
            sys.exit(
                "hg19 genome FASTA not found in any of the expected locations:\n  "
                + "\n  ".join(str(p) for p in GENOME_SEARCH_PATHS)
                + "\nProvide the path with --genome /path/to/hg19.fa"
            )

    genome = load_genome(SeqIO, genome_path)

    # --- gene index -------------------------------------------------
    print(f"Loading gene index from {GENE_INDEX}")
    with open(GENE_INDEX) as f:
        genes = json.load(f)

    if args.max_genes:
        genes = genes[:args.max_genes]
        print(f"  Limiting to {len(genes)} genes (--max-genes).")
    else:
        print(f"  {len(genes)} genes.")

    n_done = 0
    n_skip = 0
    n_miss = 0

    for gene in tqdm(genes, desc="MFE profiles", unit="gene"):
        gene_id = gene["gene_id"]
        chrom   = gene["chrom"]
        strand  = gene["strand"]

        out_path = OUT_DIR / f"{gene_id}_mfe.json"

        # checkpoint: skip if already complete
        if out_path.exists():
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                if "positions_bp" in existing and len(existing["positions_bp"]) > 0:
                    n_skip += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                pass  # corrupted — recompute

        # --- determine TSS and wave end ----------------------------
        if strand == "+":
            tss = int(gene["start"])
        else:
            tss = int(gene["end"])

        # load wave_40m from speed profile
        prof_path = SPEED_PROF_DIR / gene["file"]
        wave_end = None
        if prof_path.exists():
            with open(prof_path) as f:
                prof_data = json.load(f)
            wave_end = prof_data.get("wave_40m")

        if wave_end is None or wave_end <= 0:
            # fall back to gene length
            gene_len = int(gene["end"]) - int(gene["start"])
            wave_end = min(gene_len, 200_000)
            warnings.warn(
                f"{gene_id}: no wave_40m in profile; using gene length {wave_end} bp."
            )

        wave_end = int(wave_end)

        # --- compute MFE at each step ------------------------------
        positions: list[int]   = []
        mfe_vals: list[float]  = []

        for p in range(STEP_BP, wave_end + 1, STEP_BP):
            seq = extract_transcript_seq(genome, chrom, tss, int(gene["end"]),
                                         strand, p)
            if seq is None:
                warnings.warn(f"{gene_id}: chromosome {chrom} not in genome — skipping gene.")
                n_miss += 1
                break

            # trim exit channel
            usable = seq[: max(0, len(seq) - EXIT_CHANNEL_NT)]
            if len(usable) < MIN_SEQ_LEN:
                continue

            mfe = compute_mfe(RNA, usable)
            positions.append(p)
            mfe_vals.append(mfe)

        if not positions:
            continue

        # --- write result ------------------------------------------
        result = {
            "gene_id":      gene_id,
            "chrom":        chrom,
            "strand":       strand,
            "tss":          tss,
            "step_bp":      STEP_BP,
            "wave_40m_bp":  wave_end,
            "exit_channel_nt": EXIT_CHANNEL_NT,
            "positions_bp":   positions,
            "mfe_kcal_mol":   mfe_vals,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        n_done += 1

    print(f"\nDone.  Computed: {n_done}, Skipped (cached): {n_skip}, "
          f"Missing chrom: {n_miss}.")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
