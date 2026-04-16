#!/usr/bin/env python3
"""
batch_coverage.py - Compute AD-averaged coverage for all 81 Danko genes.
Groups genes by chromosome for efficient I/O.
"""
import numpy as np
import pandas as pd
import gzip, json, pickle
from pathlib import Path
from collections import defaultdict

BASE     = Path("/home/mathornton/herald-workspace/simon/Projects/AD Experiments/read accumulation rate experiments")
RAW      = BASE / "data" / "raw"
RATES_DIR= BASE / "data" / "danko_rates"
PROC     = BASE / "data" / "processed"

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

def load_chrom_strand(path, lib_size, chrom, strand):
    rows = []
    cb = chrom.encode(); sb = strand.encode()
    with gzip.open(str(path), "rb") as fh:
        for line in fh:
            if line[:len(cb)] == cb and line[len(cb):len(cb)+1] == b"\t":
                parts = line.split(b"\t")
                if parts[5].strip() == sb:
                    rows.append(int(parts[1]))
    return np.array(rows, np.int32), 1e6 / lib_size

def make_cov(positions, norm, start, end):
    L = end - start
    cov = np.zeros(L, np.float32)
    mask = (positions >= start) & (positions < end)
    np.add.at(cov, positions[mask] - start, 1)
    return cov * norm

def ad_average(cov_list):
    return np.stack(cov_list).mean(axis=0), 10*np.log10(len(cov_list))

rates = pd.read_csv(RATES_DIR / "MCF7.10-40m.regressionRate.tsv", sep="\t")
print(f"Processing {len(rates)} genes...", flush=True)

gene_groups = defaultdict(list)
for _, row in rates.iterrows():
    gene_groups[(row.chrom, row.strand)].append(row)

coverages = {}
for (chrom, strand), gene_rows in sorted(gene_groups.items()):
    print(f"  {chrom} {strand}: {len(gene_rows)} gene(s)", flush=True)
    for tp, reps in TIMEPOINTS.items():
        cov_lists = defaultdict(list)
        for rep in reps:
            positions, norm = load_chrom_strand(RAW/SAMPLE_FILES[rep], LIB_SIZES[rep], chrom, strand)
            for gr in gene_rows:
                cov_lists[gr.uniqueID].append(make_cov(positions, norm, gr.chromStart, gr.chromEnd))
        for gr in gene_rows:
            uid = gr.uniqueID
            if uid not in coverages:
                coverages[uid] = {}
            coverages[uid][tp], _ = ad_average(cov_lists[uid])

with open(PROC / "gene_coverages.pkl", "wb") as f:
    pickle.dump(coverages, f)
print(f"Saved {len(coverages)} genes to gene_coverages.pkl", flush=True)
