# Read Accumulation Rate Experiments

Per-gene RNA Polymerase II instantaneous elongation speed estimation from GRO-Seq
coverage, viewed through the Algebraic Diversity (AD) spectral framework.

An interactive Flask web application visualizes three speed estimators side by side
and presents a research dashboard with strategy log, todos, epigenomic overlays,
and temporal rephasing results.

## Goal

Use GRO-Seq read coverage profiles as a proxy for RNAPII occupancy to estimate
instantaneous elongation speed at 1 bp resolution, then apply Algebraic Diversity
(AD) to characterize the spectral structure of those speed profiles and derive
per-gene summary statistics that correlate with wave-front elongation rates from
Danko et al. (2013).

## Biological Motivation

In GRO-Seq, a nascent RNA read at position p means an RNAPII molecule was
elongating through position p at the time of run-on. Read density x(p) is
therefore proportional to occupancy time -- how long polymerase dwells at each
position. Since dwell time = 1/speed:

    v(p) ~ 1 / x(p)

This gives an instantaneous speed profile at every base pair along the gene body.

## Dataset

Danko et al. (2013), Molecular Cell 50:212-222.
"Signaling Pathways Differentially Affect RNA Polymerase II Initiation, Pausing,
and Elongation Rate in Cells." PMID: 23523369

- GEO: GSE41324
- Cell line: MCF-7 (human breast cancer)
- Treatment: 17-beta-estradiol (E2)
- Time points: Vehicle, 10 min, 25 min, 40 min (3 reps each)
- Genome: hg19

## AD Framework Application

We treat the per-gene coverage vector x in R^M (M = gene length in bp) as a
single observation and apply the cyclic group Z_M to obtain a full-rank spectral
estimate:

    R_hat = (1/M) * sum_{k=0}^{M-1} (P^k x)(P^k x)^T

The eigenvalues are the DFT periodogram of x. Key derived quantities:

- psi = spectral concentration = (lambda_max - lambda_min) / trace(R_hat)
- f*  = dominant pausing frequency
- Spectral entropy of coverage profile

Per-gene speed summaries derived from v_hat(p) = C / x(p):

- Harmonic mean of coverage-inverse (primary candidate)
- Spectral centroid of v_hat

## Three Speed Estimators Compared

| Estimator | Formula | Interpretation |
|-----------|---------|----------------|
| A: Raw 40m | v(p) = C / rho_40m(p) | Steady-state inverse density |
| B: Differential | v(p) = C / (rho_40m - rho_0m)(p) | E2-induced change |
| C: Product group | Z_M x Z_3 average over timepoints | Temporally coherent estimate |

## Benchmarking

We compare our per-gene speed summaries to the wave-front elongation rates from
Danko et al. (2013), computed by tracking the leading edge of the Pol II signal
in difference maps (E2 - Vehicle) using an HMM and linear regression.

## Repository Layout

```
read accumulation rate experiments/
├── README.md                 (this file)
├── STRATEGY_LOG.md           detailed session-by-session research log
├── LICENSE                   MIT
├── requirements.txt          top-level python deps
├── .gitignore                excludes raw data and large caches
├── scripts/                  numbered pipeline scripts 01..14
│   ├── 01_download_data.sh       GEO download
│   ├── 02_compute_coverage.sh    BED -> per-gene coverage
│   ├── 03_wave_front.py          HMM wave-front tracker
│   ├── 04_ad_estimator.py        Z_M spectral estimate
│   ├── 07_ad_speed_profiles.py   three-estimator computation
│   ├── 09_download_epigenomic.sh ENCODE bigWig pulls
│   ├── 10_epigenomic_profiles.py per-gene H3K*, CTCF extract
│   ├── 11_temporal_rephase.py    Z_M x Z_3 rephasing
│   ├── 12_cross_spectrum_ad.py   cross-spectral AD
│   └── 13,14 RNA folding vs speed
├── webapp/
│   ├── app.py                Flask app (port 5050)
│   ├── templates/index.html  dashboard UI
│   ├── static/               assets
│   └── requirements.txt      flask deps
├── notebooks/                exploratory ipynb
├── references/               key papers (Danko 2013, etc.)
├── data/                     (excluded) raw BEDs, bigWigs, processed caches
├── results/                  CSVs, PNGs, small summaries
└── logs/                     pipeline run logs
```

## Getting Started

1. Clone the repo:

        git clone git@github.com:mathornton01/ad-speed-profile-viewer.git
        cd ad-speed-profile-viewer

2. Install deps (ideally in a venv):

        python3 -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt

3. Download the raw GRO-Seq data (approximately 9 GB):

        bash scripts/01_download_data.sh

4. Optionally pull ENCODE epigenomic tracks for MCF-7:

        bash scripts/09_download_epigenomic.sh

5. Run the pipeline:

        python3 scripts/07_ad_speed_profiles.py
        python3 scripts/10_epigenomic_profiles.py
        python3 scripts/11_temporal_rephase.py

6. Launch the viewer:

        cd webapp
        python3 app.py
        # open http://localhost:5050

## Webapp Dashboards

The web app serves several interactive views:

- Speed profile viewer: per-gene three-estimator comparison
- Strategy log: auto-parsed session history from STRATEGY_LOG.md
- TODO board: active / completed research tasks
- Epigenomics overlay: H3K27ac, H3K27me3, H3K36me3, H3K4me3, CTCF tracks
- Temporal rephasing: Z_M x Z_3 coherent speed estimate
- Pipeline status: which data artifacts exist locally

API endpoints:

    GET  /api/strategy
    GET  /api/todos
    GET  /api/status
    GET  /api/epigenomics/<gene_file>
    GET  /api/temporal/<gene_file>

## Running Log

See `STRATEGY_LOG.md` for detailed session-by-session strategy, decisions,
open questions, and the evolving TODO list.

## License

MIT (see `LICENSE`).

## Citation

If this code is useful to your work please cite:

    Thornton, M. A., AD Speed Profile Viewer:
    Algebraic-diversity RNA Pol II elongation speed estimation from GRO-Seq.
    GitHub repository, 2026.
