"""
05_correlation.py — Benchmark AD-derived speed summaries against Danko wave-front rates.

Computes Pearson and Spearman correlations between:
  - Our AD/reciprocal-coverage speed proxies
  - Danko 2013 wave-front elongation rates (ground truth)

Also produces scatter plots and a summary table.

Author: Simon (Herald AI) + M. Thornton
Date: 2026-04-04
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not available — plots will be skipped.")


def correlation_summary(
    danko_rates: np.ndarray,
    our_rates: np.ndarray,
    label: str = "Our Estimator",
) -> dict:
    """
    Compute Pearson and Spearman correlations between two rate vectors.

    Parameters
    ----------
    danko_rates : np.ndarray
        Reference wave-front rates from Danko 2013 (kb/min).
    our_rates : np.ndarray
        Our estimated rates or speed proxies (any proportional units).
    label : str
        Name of our estimator for reporting.

    Returns
    -------
    results : dict
        pearson_r, pearson_p, spearman_r, spearman_p, n_genes
    """
    mask = np.isfinite(danko_rates) & np.isfinite(our_rates)
    d = danko_rates[mask]
    o = our_rates[mask]

    if len(d) < 3:
        return {"label": label, "n_genes": len(d), "pearson_r": np.nan,
                "spearman_r": np.nan, "pearson_p": np.nan, "spearman_p": np.nan}

    pr, pp = stats.pearsonr(d, o)
    sr, sp = stats.spearmanr(d, o)

    return {
        "label": label,
        "n_genes": int(mask.sum()),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def plot_scatter(
    danko_rates: np.ndarray,
    our_rates: np.ndarray,
    gene_ids: list,
    label: str,
    out_path: str,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """
    Scatter plot: Danko rates (x) vs our rates (y), with regression line.
    """
    if not HAS_MPL:
        return

    mask = np.isfinite(danko_rates) & np.isfinite(our_rates)
    d = danko_rates[mask]
    o = our_rates[mask]

    fig, ax = plt.subplots(figsize=(6, 6))

    if log_x:
        d = np.log10(d + 1e-6)
        xlabel = "Danko 2013 rate (log10 kb/min)"
    else:
        xlabel = "Danko 2013 rate (kb/min)"

    if log_y:
        o = np.log10(o + 1e-6)
        ylabel = f"{label} (log10)"
    else:
        ylabel = label

    ax.scatter(d, o, alpha=0.5, s=30, color="#2196F3", edgecolors="none")

    # Regression line
    slope, intercept, r, p, se = stats.linregress(d, o)
    x_line = np.linspace(d.min(), d.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#E53935", linewidth=2,
            label=f"R={r:.3f}, p={p:.2e}")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"AD Speed Proxy vs Danko Rates\n{label} (n={mask.sum()})", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def run_all_correlations(
    results_df: pd.DataFrame,
    out_dir: str = "../results",
) -> pd.DataFrame:
    """
    Run correlation analysis for all speed proxy columns vs Danko rates.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns:
          - 'gene_id'
          - 'danko_rate_kb_per_min' (reference)
          - 'mean_speed_proxy'
          - 'median_speed_proxy'
          - 'psi'
          - 'spectral_entropy'
          - 'mean_coverage_rpm' (negative correlation expected)
    out_dir : str
        Output directory for figures and tables.

    Returns
    -------
    corr_df : pd.DataFrame
        Correlation summary table.
    """
    import os
    os.makedirs(out_dir + "/figures", exist_ok=True)
    os.makedirs(out_dir + "/tables", exist_ok=True)

    danko = results_df["danko_rate_kb_per_min"].values
    gene_ids = results_df["gene_id"].tolist()

    candidates = {
        "Mean Speed Proxy (1/coverage)": results_df["mean_speed_proxy"].values,
        "Median Speed Proxy (1/coverage)": results_df["median_speed_proxy"].values,
        "Spectral Concentration (psi)": results_df["psi"].values,
        "Spectral Entropy (H)": results_df["spectral_entropy"].values,
        "Mean Coverage RPM (neg. corr expected)": results_df["mean_coverage_rpm"].values,
    }

    rows = []
    for label, our_rates in candidates.items():
        row = correlation_summary(danko, our_rates, label=label)
        rows.append(row)

        if HAS_MPL:
            safe_label = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            plot_scatter(
                danko, our_rates, gene_ids,
                label=label,
                out_path=f"{out_dir}/figures/scatter_{safe_label}.png",
                log_y=True,
            )

    corr_df = pd.DataFrame(rows)
    corr_df = corr_df.sort_values("spearman_r", ascending=False)

    # Save table
    table_path = f"{out_dir}/tables/correlation_summary.tsv"
    corr_df.to_csv(table_path, sep="\t", index=False, float_format="%.4f")
    print(f"Correlation table saved: {table_path}")

    return corr_df


if __name__ == "__main__":
    # Synthetic test: generate fake results and run correlation
    np.random.seed(42)
    n = 80

    # Simulate: true rates between 1-4 kb/min
    true_rates = np.random.uniform(1.0, 4.0, n)

    # Simulate our estimates as noisy versions of true rates
    # Mean speed proxy: inversely related to coverage, which relates to rate
    noise = np.random.normal(0, 0.2, n)
    mean_speed_proxy = 1.0 / (1.5 / true_rates + noise + 0.01)
    psi = 0.3 + 0.4 * (true_rates / 4.0) + np.random.normal(0, 0.05, n)
    spectral_entropy = 0.8 - 0.3 * (true_rates / 4.0) + np.random.normal(0, 0.05, n)
    mean_cov = 5.0 / true_rates + np.random.exponential(0.5, n)

    df = pd.DataFrame({
        "gene_id": [f"GENE_{i:03d}" for i in range(n)],
        "danko_rate_kb_per_min": true_rates,
        "mean_speed_proxy": mean_speed_proxy,
        "median_speed_proxy": mean_speed_proxy * np.random.uniform(0.9, 1.1, n),
        "psi": np.clip(psi, 0, 1),
        "spectral_entropy": np.clip(spectral_entropy, 0, 1),
        "mean_coverage_rpm": mean_cov,
    })

    print("Running correlation analysis on synthetic data (n=80 genes)...")
    corr_df = run_all_correlations(df, out_dir="../results")

    print("\nCorrelation Summary:")
    print(corr_df[["label", "n_genes", "pearson_r", "spearman_r"]].to_string(index=False))
