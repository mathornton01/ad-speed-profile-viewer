#!/bin/bash
# 02_compute_coverage.sh
# Compute 1 bp resolution strand-specific GRO-Seq coverage from BED files.
#
# Requires: bedtools, samtools (optional), hg19 chromosome sizes
#
# Input:  data/raw/GSE41324/*.bed.gz  (or .bam files)
# Output: data/processed/coverage/*.bedgraph  (strand-specific, 1 bp)
#
# Usage: bash 02_compute_coverage.sh
# Run from: read accumulation rate experiments/

set -euo pipefail

RAW_DIR="../data/raw/GSE41324"
PROC_DIR="../data/processed/coverage"
ANNO_DIR="../data/annotations"
CHROM_SIZES="$ANNO_DIR/hg19.chrom.sizes"

mkdir -p "$PROC_DIR" "$ANNO_DIR"

# ── Download hg19 chromosome sizes if not present ────────────────────────────
if [ ! -f "$CHROM_SIZES" ]; then
    echo "Fetching hg19 chromosome sizes..."
    wget -q -O "$CHROM_SIZES" \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.chrom.sizes"
fi

# ── Download hg19 RefSeq gene annotations if not present ────────────────────
REFSEQ_BED="$ANNO_DIR/hg19_refseq_genes.bed"
if [ ! -f "$REFSEQ_BED" ]; then
    echo "Fetching hg19 RefSeq gene models..."
    # Download from UCSC table browser (refGene)
    wget -q -O "$ANNO_DIR/refGene.txt.gz" \
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz"
    # Convert to BED12 format: chrom, txStart, txEnd, name, score, strand
    gunzip -c "$ANNO_DIR/refGene.txt.gz" | awk 'BEGIN{OFS="\t"} {
        print $3, $5, $6, $2, 0, $4
    }' | sort -k1,1 -k2,2n > "$REFSEQ_BED"
    echo "  Wrote: $REFSEQ_BED"
fi

# ── Process each BED file ────────────────────────────────────────────────────
# GEO BED files for GRO-Seq are typically already mapped reads in BED format
# Each line = one read, with strand in column 6

echo "Computing coverage from BED files in $RAW_DIR..."

for BED_GZ in "$RAW_DIR"/*.bed.gz; do
    [ -f "$BED_GZ" ] || { echo "No BED files found in $RAW_DIR"; break; }

    SAMPLE=$(basename "$BED_GZ" .bed.gz)
    echo "  Processing: $SAMPLE"

    # Count total reads for RPM normalization
    TOTAL=$(zcat "$BED_GZ" | wc -l)
    SCALE=$(echo "scale=10; 1000000 / $TOTAL" | bc)

    # Plus strand coverage
    zcat "$BED_GZ" | awk '$6=="+"' | \
        bedtools genomecov -i stdin -g "$CHROM_SIZES" -bg -scale "$SCALE" | \
        sort -k1,1 -k2,2n \
        > "$PROC_DIR/${SAMPLE}_plus.bedgraph"

    # Minus strand coverage (negate for visualization, keep positive for analysis)
    zcat "$BED_GZ" | awk '$6=="-"' | \
        bedtools genomecov -i stdin -g "$CHROM_SIZES" -bg -scale "$SCALE" | \
        sort -k1,1 -k2,2n \
        > "$PROC_DIR/${SAMPLE}_minus.bedgraph"

    echo "    Total reads: $TOTAL  RPM scale: $SCALE"
    echo "    Output: ${SAMPLE}_plus.bedgraph, ${SAMPLE}_minus.bedgraph"
done

echo ""
echo "Coverage files written to: $PROC_DIR"
echo ""
echo "Next: run 03_wave_front.py and 04_ad_estimator.py"
