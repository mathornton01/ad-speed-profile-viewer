"""
coverage.py - Strand-specific 1bp coverage from GRO-Seq BED files.

BED format (GSE41324): chrom, start, end, name, score, strand
start==end for single-base read positions (5' end of RNAPII-associated RNA).

No bedtools required - pure numpy/pandas implementation.
"""
import numpy as np
import pandas as pd
import gzip
import os
from pathlib import Path


SAMPLE_FILES = {
    "0m_R1":  "GSM1014637_E2.0m_R1.noAdapt.bed.gz",
    "0m_R2":  "GSM1014638_E2.0m_R2.noAdapt.bed.gz",
    "0m_R3":  "GSM1014639_E2.0m_R3.noAdapt.bed.gz",
    "10m_R1": "GSM1014640_E2.10m_R1.noAdapt.bed.gz",
    "10m_R2": "GSM1014641_E2.10m_R2.noAdapt.bed.gz",
    "10m_R3": "GSM1014642_E2.10m_R3.noAdapt.bed.gz",
    "25m_R1": "GSM1014643_E2.25m_R1.noAdapt.bed.gz",
    "25m_R3": "GSM1014644_E2.25m_R3.noAdapt.bed.gz",
    "40m_R1": "GSM1014645_E2.40m_R1.noAdapt.bed.gz",
    "40m_R2": "GSM1014646_E2.40m_R2.noAdapt.bed.gz",
    "40m_R3": "GSM1014647_E2.40m_R3.noAdapt.bed.gz",
}

TIMEPOINTS = {
    "0m":  ["0m_R1", "0m_R2", "0m_R3"],
    "10m": ["10m_R1", "10m_R2", "10m_R3"],
    "25m": ["25m_R1", "25m_R3"],
    "40m": ["40m_R1", "40m_R2", "40m_R3"],
}


def load_bed(path: str) -> pd.DataFrame:
    """Load a GRO-Seq BED.gz file into a DataFrame.
    
    Returns columns: chrom, pos (0-based), strand
    pos is the 5' end position of the read (col2 == col3 in this dataset).
    """
    with gzip.open(path, 'rb') as fh:
        df = pd.read_csv(
            fh, sep="\t", header=None, usecols=[0, 1, 5],
            names=["chrom", "pos", "strand"],
            dtype={"chrom": "category", "pos": np.int32, "strand": "category"},
        )
    return df


def library_size(df: pd.DataFrame) -> int:
    """Total number of reads in BED file."""
    return len(df)


def gene_coverage(
    df: pd.DataFrame,
    chrom: str,
    start: int,
    end: int,
    strand: str,
    lib_size: int,
    rpm_scale: bool = True,
) -> np.ndarray:
    """
    Extract 1bp strand-specific coverage over a gene locus.

    Parameters
    ----------
    df        : loaded BED DataFrame (chrom, pos, strand)
    chrom     : chromosome (e.g. 'chr1')
    start     : locus start (0-based, inclusive)
    end       : locus end (0-based, exclusive)
    strand    : '+' or '-'
    lib_size  : total reads in library (for RPM)
    rpm_scale : if True, return reads-per-million; else raw counts

    Returns
    -------
    cov : np.ndarray of shape (end - start,) with 1bp coverage
    """
    mask = (df.chrom == chrom) & (df.strand == strand) & \
           (df.pos >= start) & (df.pos < end)
    sub = df.loc[mask, "pos"].values - start  # shift to 0-indexed within locus
    length = end - start
    cov = np.zeros(length, dtype=np.float32)
    np.add.at(cov, sub, 1)
    if rpm_scale and lib_size > 0:
        cov = cov * (1e6 / lib_size)
    return cov


def diff_coverage(cov_t: np.ndarray, cov_0: np.ndarray) -> np.ndarray:
    """
    Compute difference coverage (treatment - control), 
    clipped to 0 (signal only where treatment > control).
    Used for wave-front detection.
    """
    return np.maximum(cov_t - cov_0, 0.0)
