#!/bin/bash
# 09_download_epigenomic.sh
# Download fresh MCF-7 epigenomic data from ENCODE for:
#   - WGBS DNA methylation (bigWig, fraction methylated)
#   - H3K36me3 ChIP-seq (fold change over control bigWig)
#   - H3K27ac  ChIP-seq
#   - H3K4me3  ChIP-seq
#   - H3K27me3 ChIP-seq
#   - CTCF     ChIP-seq (TF ChIP-seq)
#
# Usage: bash scripts/09_download_epigenomic.sh
# Run from: read accumulation rate experiments/

set -euo pipefail

ENCODE_BASE="https://www.encodeproject.org"
DATA_DIR="data/epigenomics"
ASSEMBLY="GRCh38"

# ── directories ────────────────────────────────────────────────────────────────
mkdir -p \
    "$DATA_DIR/methylation" \
    "$DATA_DIR/H3K36me3" \
    "$DATA_DIR/H3K27ac" \
    "$DATA_DIR/H3K4me3" \
    "$DATA_DIR/H3K27me3" \
    "$DATA_DIR/CTCF"

# ── check / install pyBigWig ───────────────────────────────────────────────────
echo "Checking pyBigWig..."
python3 -c "import pyBigWig; print('pyBigWig ok')" 2>/dev/null \
    || { echo "pyBigWig not found – installing via pip3..."; \
         pip3 install pyBigWig --quiet 2>/dev/null \
         || pip3 install pyBigWig --quiet --break-system-packages 2>/dev/null \
         || echo "Warning: pyBigWig install failed (optional, continuing)"; }

# ── helper: query ENCODE and pick best bigWig ──────────────────────────────────
# Usage: pick_bigwig <search_url> <output_subdir> <file_type_hint>
# Prints: accession TAB download_url
pick_bigwig() {
    local url="$1"
    local subdir="$2"
    local hint="${3:-fold_change_over_control}"

    echo "  Querying: $url" >&2

    # Fetch experiment list
    local resp
    resp=$(wget -q -O - "$url&limit=10" 2>/dev/null) || { echo "  WARN: query failed" >&2; return 1; }

    # Pull @graph accessions
    local accessions
    accessions=$(python3 - <<'PYEOF'
import sys, json
data = json.load(sys.stdin)
expts = data.get("@graph", [])
for e in expts:
    print(e.get("accession",""))
PYEOF
 <<< "$resp")

    if [ -z "$accessions" ]; then
        echo "  WARN: no experiments found" >&2
        return 1
    fi

    # Walk experiments until we find a usable file
    for acc in $accessions; do
        local expt_url="$ENCODE_BASE/experiments/$acc/?format=json"
        local expt
        expt=$(wget -q -O - "$expt_url" 2>/dev/null) || continue

        # Try to pick a pooled, GRCh38, fold_change bigWig
        local file_info
        file_info=$(python3 - "$hint" <<'PYEOF'
import sys, json
hint = sys.argv[1]
data = json.load(sys.stdin)
files = data.get("files", [])
candidates = []
for f in files:
    if f.get("file_format") != "bigWig":
        continue
    asm = f.get("assembly", "")
    if asm not in ("GRCh38", "hg38"):
        continue
    out = f.get("output_type", "")
    bio_reps = f.get("biological_replicates", [])
    tech_reps = f.get("technical_replicates", [])
    pooled = len(bio_reps) > 1 or (bio_reps == [] and tech_reps == [])
    score = 0
    if hint in out:
        score += 10
    if "signal" in out:
        score += 5
    if pooled:
        score += 3
    if "released" == f.get("status", ""):
        score += 2
    candidates.append((score, f.get("accession",""), f.get("href",""), out, asm))

candidates.sort(key=lambda x: -x[0])
if candidates:
    s, a, href, ot, asm = candidates[0]
    print(f"{a}\t{href}\t{ot}\t{asm}")
PYEOF
 <<< "$expt")

        if [ -n "$file_info" ]; then
            echo "$acc $file_info"
            return 0
        fi
    done

    echo "  WARN: no suitable bigWig found in any experiment" >&2
    return 1
}

# ── metadata accumulator ───────────────────────────────────────────────────────
META_FILE="$DATA_DIR/metadata.json"
cat > "$META_FILE" <<'JSON'
{
  "description": "MCF-7 epigenomic data from ENCODE (GRCh38)",
  "downloaded": "",
  "tracks": {}
}
JSON

update_meta() {
    local mark="$1"
    local expt_acc="$2"
    local file_acc="$3"
    local output_type="$4"
    local assembly="$5"
    local filename="$6"
    python3 - "$mark" "$expt_acc" "$file_acc" "$output_type" "$assembly" "$filename" "$META_FILE" <<'PYEOF'
import sys, json
mark, expt, facc, otype, asm, fn, meta_path = sys.argv[1:]
with open(meta_path) as fh:
    d = json.load(fh)
import datetime
d["downloaded"] = datetime.datetime.utcnow().isoformat() + "Z"
d["tracks"][mark] = {
    "experiment_accession": expt,
    "file_accession": facc,
    "output_type": otype,
    "assembly": asm,
    "local_file": fn
}
with open(meta_path, "w") as fh:
    json.dump(d, fh, indent=2)
PYEOF
}

# ── download helper ────────────────────────────────────────────────────────────
download_mark() {
    local mark="$1"
    local search_url="$2"
    local out_dir="$3"
    local hint="${4:-fold_change_over_control}"

    echo ""
    echo "=== $mark ==="
    local info
    info=$(pick_bigwig "$search_url" "$out_dir" "$hint") || { echo "Skipping $mark"; return; }

    local expt_acc file_acc href output_type assembly filename
    expt_acc=$(echo "$info" | awk '{print $1}')
    file_acc=$(echo "$info" | awk '{print $2}')
    href=$(echo "$info"     | awk '{print $3}')
    output_type=$(echo "$info" | awk '{print $4}')
    assembly=$(echo "$info"    | awk '{print $5}')

    local full_url="${ENCODE_BASE}${href}"
    filename="${out_dir}/${file_acc}.bigWig"

    echo "  Experiment : $expt_acc"
    echo "  File       : $file_acc  ($output_type, $assembly)"
    echo "  URL        : $full_url"
    echo "  Saving to  : $filename"

    wget -q --show-progress -O "$filename" "$full_url" \
        && echo "  Downloaded OK" \
        || { echo "  WARN: download failed"; return; }

    update_meta "$mark" "$expt_acc" "$file_acc" "$output_type" "$assembly" "$filename"
}

# ── WGBS ───────────────────────────────────────────────────────────────────────
WGBS_URL="${ENCODE_BASE}/search/?type=Experiment\
&biosample_ontology.term_name=MCF-7\
&assay_title=WGBS\
&status=released\
&format=json"
download_mark "WGBS_methylation" "$WGBS_URL" "$DATA_DIR/methylation" "methylation"

# ── Histone ChIP-seq marks ─────────────────────────────────────────────────────
for mark in H3K36me3 H3K27ac H3K4me3 H3K27me3; do
    CHIP_URL="${ENCODE_BASE}/search/?type=Experiment\
&biosample_ontology.term_name=MCF-7\
&assay_title=Histone+ChIP-seq\
&target.label=${mark}\
&status=released\
&format=json"
    download_mark "$mark" "$CHIP_URL" "$DATA_DIR/$mark" "fold_change_over_control"
done

# ── CTCF TF ChIP-seq ───────────────────────────────────────────────────────────
CTCF_URL="${ENCODE_BASE}/search/?type=Experiment\
&biosample_ontology.term_name=MCF-7\
&assay_title=TF+ChIP-seq\
&target.label=CTCF\
&status=released\
&format=json"
download_mark "CTCF" "$CTCF_URL" "$DATA_DIR/CTCF" "fold_change_over_control"

echo ""
echo "=== Done ==="
echo "Metadata written to $META_FILE"
cat "$META_FILE"
