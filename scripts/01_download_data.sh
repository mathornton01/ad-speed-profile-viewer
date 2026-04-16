#!/bin/bash
# 01_download_data.sh
# Download GSE41324 (Danko 2013, E2 time course GRO-Seq in MCF-7)
# from NCBI GEO FTP.
#
# GEO: GSE41324
# SRA: SRP015988
# FTP base: ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE41nnn/GSE41324/
#
# Usage: bash 01_download_data.sh
# Run from: read accumulation rate experiments/

set -euo pipefail

DATA_DIR="../data/raw/GSE41324"
mkdir -p "$DATA_DIR"

GEO_FTP="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE41nnn/GSE41324/suppl"

echo "Downloading GSE41324 supplemental files..."
echo "Target: $DATA_DIR"

# Download the full supplemental archive listing first
wget -q -O "$DATA_DIR/filelist.txt" \
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41324&form=text&view=full" \
    || echo "Warning: could not fetch filelist"

# Download all BED files (GRO-Seq coverage, strand-specific)
# Typical file naming: GSE41324_GRO-seq_{condition}_{rep}.bed.gz
wget -r -nd -np -A "*.bed.gz,*.bw,*.bigwig,*.bigWig" \
    --directory-prefix="$DATA_DIR" \
    "$GEO_FTP/" \
    || echo "Note: FTP recursive download may need manual inspection"

echo ""
echo "To download raw reads via SRA instead:"
echo "  prefetch SRP015988"
echo "  fastq-dump --split-files SRP015988"
echo ""
echo "Individual sample SRR accessions (from SRA study SRP015988):"
echo "  -- Look up at: https://www.ncbi.nlm.nih.gov/sra?term=SRP015988"
echo ""
echo "Done. Files in: $DATA_DIR"
ls -lh "$DATA_DIR/" 2>/dev/null || echo "(directory empty — check download above)"
