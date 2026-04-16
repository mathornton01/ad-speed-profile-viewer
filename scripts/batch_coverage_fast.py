#!/usr/bin/env python3
"""
batch_coverage_fast.py - Read each BED file ONCE, extract all 81 genes in one pass.
Reduces 440 file reads to 11. With 121 GB RAM this is comfortably in-memory.
"""
import numpy as np
import pandas as pd
import gzip, json, pickle
from pathlib import Path
from collections import defaultdict
import time

BASE      = Path("/home/mathornton/herald-workspace/simon/Projects/AD Experiments/read accumulation rate experiments")
RAW       = BASE / "data" / "raw"
RATES_DIR = BASE / "data" / "danko_rates"
PROC      = BASE / "data" / "processed"

with open(PROC / "library_sizes.json") as f:
    LIB_SIZES = json.load(f)

SAMPLE_FILES = {
    "0m_R1": "GSM1014637_E2.0m_R1.noAdapt.bed.gz",
    "0m_R2": "GSM1014638_E2.0m_R2.noAdapt.bed.gz",
    "0m_R3": "GSM1014639_E2.0m_R3.noAdapt.bed.gz",
    "10m_R1":"GSM1014640_E2.10m_R1.noAdapt.bed.gz",
    "10m_R2":"GSM1014641_E2.10m_R2.noAdapt.bed.gz",
    "10m_R3":"GSM1014642_E2.10m_R3.noAdapt.bed.gz",
    "25m_R1":"GSM1014643_E2.25m_R1.noAdapt.bed.gz",
    "25m_R3":"GSM1014644_E2.25m_R3.noAdapt.bed.gz",
    "40m_R1":"GSM1014645_E2.40m_R1.noAdapt.bed.gz",
    "40m_R2":"GSM1014646_E2.40m_R2.noAdapt.bed.gz",
    "40m_R3":"GSM1014647_E2.40m_R3.noAdapt.bed.gz",
}
TIMEPOINTS = {
    "0m":  ["0m_R1","0m_R2","0m_R3"],
    "10m": ["10m_R1","10m_R2","10m_R3"],
    "25m": ["25m_R1","25m_R3"],
    "40m": ["40m_R1","40m_R2","40m_R3"],
}

rates = pd.read_csv(RATES_DIR / "MCF7.10-40m.regressionRate.tsv", sep="\t")
print(f"Genes: {len(rates)}", flush=True)

# Build gene lookup: (chrom, strand) -> list of (uniqueID, start, end)
gene_index = defaultdict(list)
for _, r in rates.iterrows():
    gene_index[(r.chrom, r.strand)].append((r.uniqueID, r.chromStart, r.chromEnd))

# Precompute per-gene zero arrays indexed by uniqueID
gene_meta = {r.uniqueID: (r.chrom, r.chromStart, r.chromEnd, r.strand)
             for _, r in rates.iterrows()}

def load_sample_all_genes(path, lib_size, gene_index, gene_meta):
    """Single-pass read of entire BED file, extracting coverage for all genes."""
    # Initialize accumulators
    cov_arrays = {uid: np.zeros(end - start, np.float32)
                  for uid, (ch, start, end, st) in gene_meta.items()}
    norm = 1e6 / lib_size
    
    with gzip.open(str(path), "rb") as fh:
        for line in fh:
            parts = line.split(b"\t")
            chrom  = parts[0].decode()
            strand = parts[5].strip().decode()
            key = (chrom, strand)
            if key not in gene_index:
                continue
            pos = int(parts[1])
            for uid, gstart, gend in gene_index[key]:
                if gstart <= pos < gend:
                    cov_arrays[uid][pos - gstart] += 1
    # normalise
    for uid in cov_arrays:
        cov_arrays[uid] *= norm
    return cov_arrays

# Main computation
all_cov_per_rep = {}  # sample_label -> {uid -> cov_array}
for rep_label, fname in SAMPLE_FILES.items():
    t0 = time.time()
    print(f"  Loading {rep_label} ...", end=" ", flush=True)
    cov = load_sample_all_genes(RAW/fname, LIB_SIZES[rep_label], gene_index, gene_meta)
    print(f"{time.time()-t0:.1f}s", flush=True)
    all_cov_per_rep[rep_label] = cov

# AD average across replicates for each timepoint
coverages = {uid: {} for uid in gene_meta}
for tp, reps in TIMEPOINTS.items():
    for uid in gene_meta:
        stack = np.stack([all_cov_per_rep[r][uid] for r in reps])
        coverages[uid][tp] = stack.mean(axis=0)

with open(PROC / "gene_coverages.pkl", "wb") as f:
    pickle.dump(coverages, f)
print(f"\nSaved {len(coverages)} genes to gene_coverages.pkl", flush=True)

# Quick sanity
uid0 = rates.iloc[0].uniqueID
print(f"\nSanity check ({uid0}):")
for tp in ["0m","10m","25m","40m"]:
    c = coverages[uid0][tp]
    print(f"  {tp}: mean={c.mean():.5f} RPM/bp, n_nonzero={np.count_nonzero(c):,}")
