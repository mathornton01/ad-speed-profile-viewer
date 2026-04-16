#!/usr/bin/env python3
"""
09b_download_encode.py
Download fresh MCF-7 epigenomic bigWig files from ENCODE.

Marks:
  - WGBS / RRBS DNA methylation (bigWig, fraction methylated)
  - H3K36me3 / H3K27ac / H3K4me3 / H3K27me3  Histone ChIP-seq
  - CTCF TF ChIP-seq

Usage:
  python3 scripts/09b_download_encode.py
Run from: read accumulation rate experiments/
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────

ENCODE_BASE = "https://www.encodeproject.org"
ASSEMBLY_PREF = {"GRCh38", "hg38"}

BASE_DIR = Path(__file__).resolve().parent.parent   # project root
DATA_DIR = BASE_DIR / "data" / "epigenomics"
META_PATH = DATA_DIR / "metadata.json"

# Each entry: (label, subdir, list-of-(assay_title, target_label), preferred_output_type)
# Multiple (assay, target) tuples are tried in order as fallbacks.
MARKS = [
    (
        "WGBS_methylation",
        "methylation",
        [("WGBS", None), ("RRBS", None), ("DNAme array", None)],
        "methylation state at CpG",
    ),
    ("H3K36me3", "H3K36me3", [("Histone ChIP-seq", "H3K36me3")], "fold change over control"),
    ("H3K27ac",  "H3K27ac",  [("Histone ChIP-seq", "H3K27ac")],  "fold change over control"),
    ("H3K4me3",  "H3K4me3",  [("Histone ChIP-seq", "H3K4me3")],  "fold change over control"),
    ("H3K27me3", "H3K27me3", [("Histone ChIP-seq", "H3K27me3")], "fold change over control"),
    ("CTCF",     "CTCF",     [("TF ChIP-seq", "CTCF")],          "fold change over control"),
]

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "herald-epigenomics-downloader/1.0 (mathornton@lab)",
}

# ── pyBigWig check ─────────────────────────────────────────────────────────────

def ensure_pybigwig() -> None:
    try:
        import pyBigWig  # noqa: F401
        print("pyBigWig: ok")
    except ModuleNotFoundError:
        print("pyBigWig not found – installing via pip3 ...")
        # Try normal install first; fall back to --break-system-packages for
        # PEP 668 environments (Ubuntu 23.04+, Debian 12+).
        for extra in ([], ["--break-system-packages"]):
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "pyBigWig", "--quiet"] + extra,
                    stderr=subprocess.DEVNULL if extra else None,
                )
                print("pyBigWig installed.")
                return
            except subprocess.CalledProcessError:
                if extra:
                    print("WARNING: could not install pyBigWig – continuing anyway.")
                    return

# ── ENCODE API helpers ─────────────────────────────────────────────────────────

def encode_get(path: str, retries: int = 3) -> dict:
    """
    Fetch a JSON resource from ENCODE. Returns {} on HTTP 404 (no results).
    Raises on other HTTP errors after `retries` attempts.
    """
    url = path if path.startswith("http") else ENCODE_BASE + path
    if "format=json" not in url:
        sep = "&" if "?" in url else "?"
        url += sep + "format=json"

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # ENCODE returns 404 for empty search results – treat as [].
                return {}
            last_exc = exc
        except Exception as exc:
            last_exc = exc

        if attempt < retries - 1:
            wait = 2 ** attempt
            print(f"  Retry {attempt + 1}/{retries} (err: {last_exc}) – sleeping {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"encode_get failed after {retries} attempts: {last_exc}")


def search_experiments(assay_title: str, target_label: str | None) -> list[dict]:
    """Return list of experiment stubs from ENCODE search."""
    params: dict[str, str] = {
        "type": "Experiment",
        "biosample_ontology.term_name": "MCF-7",
        "assay_title": assay_title,
        "status": "released",
        "format": "json",
        "limit": "20",
    }
    if target_label:
        params["target.label"] = target_label

    qs = urllib.parse.urlencode(params)
    url = f"{ENCODE_BASE}/search/?{qs}"
    print(f"  Search: {url}")
    data = encode_get(url)
    results = data.get("@graph", [])
    print(f"  Found {len(results)} experiment(s).")
    return results


def score_file(f: dict, preferred_output: str) -> int:
    """Score a file record; higher = better candidate. Returns -1 to exclude."""
    if f.get("file_format") != "bigWig":
        return -1
    asm = f.get("assembly", "")
    if asm not in ASSEMBLY_PREF:
        return -1
    score = 0
    out = f.get("output_type", "")
    # Exact keyword match on preferred type
    if preferred_output and preferred_output.lower() in out.lower():
        score += 20
    if "signal" in out:
        score += 5
    if "fold" in out:
        score += 8
    if "methylation" in out:
        score += 8
    bio_reps = f.get("biological_replicates", [])
    if len(bio_reps) > 1:
        score += 6  # pooled
    if f.get("status") == "released":
        score += 3
    # Slightly prefer larger files (proxy for more data)
    fsize = f.get("file_size", 0) or 0
    score += min(fsize // (10 * 1024 * 1024), 5)
    return score


def best_file(experiment: dict, preferred_output: str) -> dict | None:
    """Return the highest-scoring bigWig file from an experiment detail dict."""
    files = experiment.get("files", [])
    scored = [(score_file(f, preferred_output), f) for f in files]
    scored = [(s, f) for s, f in scored if s >= 0]
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def find_best_for_mark(
    assay_attempts: list[tuple[str, str | None]],
    preferred_output: str,
) -> tuple[str | None, str | None, dict | None]:
    """
    Try each (assay_title, target_label) pair in order.
    Returns (experiment_accession, assay_used, file_dict) or (None, None, None).
    """
    for assay_title, target_label in assay_attempts:
        desc = assay_title + (f" / {target_label}" if target_label else "")
        print(f"  Trying assay: {desc}")
        try:
            stubs = search_experiments(assay_title, target_label)
        except Exception as exc:
            print(f"  Search failed: {exc}")
            continue

        if not stubs:
            print(f"  No experiments for {desc} – trying next fallback.")
            continue

        candidates: list[tuple[int, str, dict]] = []
        for stub in stubs:
            expt_acc = stub.get("accession", "")
            if not expt_acc:
                continue
            try:
                expt = encode_get(f"/experiments/{expt_acc}/")
            except Exception as exc:
                print(f"  Could not fetch experiment {expt_acc}: {exc}")
                continue
            f = best_file(expt, preferred_output)
            if f is None:
                continue
            s = score_file(f, preferred_output)
            candidates.append((s, expt_acc, f))
            time.sleep(0.3)  # be polite to the API

        if not candidates:
            print(f"  No suitable bigWig files in {desc} experiments – trying next fallback.")
            continue

        candidates.sort(key=lambda x: -x[0])
        best_score, best_expt, best_f = candidates[0]
        print(
            f"  Best experiment: {best_expt}  file: {best_f.get('accession')}  "
            f"output: {best_f.get('output_type')}  assembly: {best_f.get('assembly')}  "
            f"score: {best_score}  assay: {assay_title}"
        )
        return best_expt, assay_title, best_f

    print("  No suitable data found across all fallbacks.")
    return None, None, None

# ── download ───────────────────────────────────────────────────────────────────

def download_file(href: str, dest: Path) -> bool:
    url = ENCODE_BASE + href if href.startswith("/") else href
    print(f"  Downloading: {url}")
    print(f"  -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 512  # 512 KB
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                out.write(block)
                downloaded += len(block)
                if total:
                    pct = downloaded / total * 100
                    print(
                        f"\r  Progress: {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)",
                        end="", flush=True,
                    )
            print()
        print(f"  Saved {downloaded / 1e6:.2f} MB")
        return True
    except Exception as exc:
        print(f"  ERROR downloading: {exc}")
        if dest.exists():
            dest.unlink()
        return False

# ── metadata ───────────────────────────────────────────────────────────────────

def load_meta() -> dict:
    if META_PATH.exists():
        with open(META_PATH) as fh:
            return json.load(fh)
    return {
        "description": "MCF-7 epigenomic data from ENCODE (GRCh38)",
        "downloaded": None,
        "tracks": {},
    }


def save_meta(meta: dict) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta["downloaded"] = datetime.now(timezone.utc).isoformat()
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Metadata written to {META_PATH}")

# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("ENCODE MCF-7 epigenomic downloader")
    print("=" * 60)

    ensure_pybigwig()

    # Create output directories
    for _, subdir, *_ in MARKS:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)

    meta = load_meta()

    for label, subdir, assay_attempts, preferred_output in MARKS:
        print(f"\n{'=' * 60}")
        print(f"Mark: {label}")
        print(f"{'=' * 60}")

        expt_acc, assay_used, file_info = find_best_for_mark(assay_attempts, preferred_output)

        if file_info is None:
            print(f"  SKIPPING {label}: no suitable file found")
            meta["tracks"][label] = {"status": "not_found"}
            save_meta(meta)
            continue

        file_acc   = file_info.get("accession", "unknown")
        href       = file_info.get("href", "")
        output_type = file_info.get("output_type", "")
        assembly    = file_info.get("assembly", "")

        dest = DATA_DIR / subdir / f"{file_acc}.bigWig"

        ok = download_file(href, dest)

        meta["tracks"][label] = {
            "experiment_accession": expt_acc,
            "file_accession": file_acc,
            "assay_used": assay_used,
            "output_type": output_type,
            "assembly": assembly,
            "local_file": str(dest),
            "status": "downloaded" if ok else "download_failed",
        }
        save_meta(meta)
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("All done.")
    print("=" * 60)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
