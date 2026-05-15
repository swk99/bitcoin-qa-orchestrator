"""
btc_calibrate.py
----------------
Fits MDP transition parameters from collected live Bitcoin data.

Fitted parameters (written to calibration_runs table + JSON):
  mu_mempool    : log-normal mu  for mempool_size_mb
  sigma_mempool : log-normal sigma
  lambda_inter  : exponential rate for inter_block_time_sec (1/mean)
  anomaly_rate  : empirical P(anomalous)
  p95_*         : 95th-percentile thresholds (updates anomaly oracle)

The fitted parameters can replace synthetic defaults in rade_train.py:
  --mu-mempool, --sigma-mempool, --lambda-inter

Usage:
    python btc_calibrate.py --days 90
    python btc_calibrate.py --days 30 --plot --output params.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Fitting functions (no scipy dependency)
# ──────────────────────────────────────────────

def fit_lognormal(values: list[float]) -> tuple[float, float]:
    """
    MLE for LogNormal(mu, sigma) from positive values.
    mu    = mean(log(x))
    sigma = std(log(x))
    """
    if not values:
        raise ValueError("empty sample")
    log_x = [math.log(v) for v in values if v > 0]
    n     = len(log_x)
    mu    = sum(log_x) / n
    var   = sum((x - mu) ** 2 for x in log_x) / max(n - 1, 1)
    return round(mu, 4), round(math.sqrt(var), 4)


def fit_exponential(values: list[float]) -> float:
    """
    MLE for Exponential(lambda) from positive values.
    lambda = 1 / mean(x)
    """
    if not values:
        raise ValueError("empty sample")
    mean_x = sum(values) / len(values)
    return round(1.0 / mean_x, 6)


def percentile(values: list[float], p: float) -> float:
    """p in [0,100]. Simple linear interpolation."""
    if not values:
        return 0.0
    sv = sorted(values)
    n  = len(sv)
    k  = (p / 100.0) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sv[lo] * (1 - frac) + sv[hi] * frac


# ──────────────────────────────────────────────
# Main calibration routine
# ──────────────────────────────────────────────

def calibrate(
    days:   int  = 90,
    source: Optional[str] = None,
    plot:   bool = False,
    output: Optional[str] = None,
    save_db: bool = True,
    notes:  Optional[str] = None,
) -> dict:
    """
    Load snapshots from PostgreSQL, fit distributions, return parameter dict.
    """
    from btc_live_db import LiveSnapshotDB

    db = LiveSnapshotDB.from_env()
    db.init_schema()

    df = db.to_dataframe(days=days, source=source)
    n  = len(df)
    log.info("Loaded %d snapshots (days=%d, source=%s)", n, days, source or "all")

    if n < 10:
        raise RuntimeError(
            f"Only {n} snapshots available. "
            "Run btc_live_runner.py for at least a few hours first."
        )

    # ── extract series ─────────────────────────
    mempool_vals    = df["mempool_size_mb"].tolist()
    inter_vals      = [v for v in df["inter_block_time_sec"].tolist() if v > 0]
    pending_vals    = df["pending_tx"].tolist()
    anomaly_vals    = df["is_anomalous"].tolist()

    # ── fit distributions ──────────────────────
    mu_m, sigma_m = fit_lognormal(mempool_vals)
    lambda_inter  = fit_exponential(inter_vals)
    anomaly_rate  = sum(1 for v in anomaly_vals if v) / len(anomaly_vals)

    # 95th percentile thresholds
    p95_mempool = percentile(mempool_vals, 95)
    p95_inter   = percentile(inter_vals,   95)
    p95_pending = int(percentile(pending_vals, 95))

    params = {
        "days_used":       days,
        "n_samples":       n,
        "mu_mempool":      mu_m,
        "sigma_mempool":   sigma_m,
        "lambda_inter":    lambda_inter,
        "mean_inter_sec":  round(1.0 / lambda_inter, 1),
        "anomaly_rate":    round(anomaly_rate, 4),
        "p95_mempool_mb":  round(p95_mempool, 2),
        "p95_inter_sec":   round(p95_inter,   1),
        "p95_pending":     p95_pending,
        "notes":           notes,
    }

    # ── summary ────────────────────────────────
    print("\n" + "=" * 55)
    print("  RADE MDP Calibration Results")
    print("=" * 55)
    print(f"  Samples   : {n:,}  ({days} days, source={source or 'all'})")
    print(f"\n  Mempool size (MB)")
    print(f"    Distribution : LogNormal(mu={mu_m}, sigma={sigma_m})")
    print(f"    Mean (approx): {math.exp(mu_m + sigma_m**2/2):.1f} MB")
    print(f"    P95          : {p95_mempool:.1f} MB  ← anomaly threshold")
    print(f"\n  Inter-block time (s)")
    print(f"    Distribution : Exponential(lambda={lambda_inter})")
    print(f"    Mean         : {1/lambda_inter:.1f} s  (paper: 592 s)")
    print(f"    P95          : {p95_inter:.0f} s  ← anomaly threshold")
    print(f"\n  Pending tx")
    print(f"    P95          : {p95_pending:,}  ← anomaly threshold")
    print(f"\n  Anomaly rate   : {anomaly_rate*100:.1f}%  (paper target: ~5%)")
    print("=" * 55)

    # ── optional plot ──────────────────────────
    if plot:
        _plot_distributions(mempool_vals, inter_vals, params)

    # ── save JSON ──────────────────────────────
    if output:
        Path(output).write_text(json.dumps(params, indent=2))
        print(f"\n  Saved: {output}")

    # ── save to DB ─────────────────────────────
    if save_db:
        cal_id = db.insert_calibration(params)
        print(f"  DB calibration_runs.id = {cal_id}")

    db.close()
    return params


def _plot_distributions(
    mempool_vals: list[float],
    inter_vals:   list[float],
    params:       dict,
) -> None:
    """Plot histograms + fitted PDFs. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Mempool histogram + LogNormal fit
    ax = axes[0]
    ax.hist(mempool_vals, bins=50, density=True, alpha=0.6, color="steelblue",
            label="observed")
    mu, sigma = params["mu_mempool"], params["sigma_mempool"]
    xs = np.linspace(0.1, max(mempool_vals), 300)
    pdf = (1 / (xs * sigma * np.sqrt(2 * np.pi))) * \
          np.exp(-((np.log(xs) - mu) ** 2) / (2 * sigma ** 2))
    ax.plot(xs, pdf, "r-", lw=2, label=f"LogNormal(μ={mu}, σ={sigma})")
    ax.axvline(params["p95_mempool_mb"], color="orange", ls="--",
               label=f"P95={params['p95_mempool_mb']:.1f} MB")
    ax.set_xlabel("Mempool size (MB)")
    ax.set_ylabel("Density")
    ax.set_title("Mempool Size Distribution")
    ax.legend(fontsize=8)

    # Inter-block histogram + Exponential fit
    ax = axes[1]
    ax.hist(inter_vals, bins=50, density=True, alpha=0.6, color="coral",
            label="observed")
    lam = params["lambda_inter"]
    xs2 = np.linspace(0, max(inter_vals), 300)
    ax.plot(xs2, lam * np.exp(-lam * xs2), "b-", lw=2,
            label=f"Exp(λ={lam:.5f})")
    ax.axvline(params["p95_inter_sec"], color="orange", ls="--",
               label=f"P95={params['p95_inter_sec']:.0f} s")
    ax.set_xlabel("Inter-block time (s)")
    ax.set_ylabel("Density")
    ax.set_title("Inter-Block Time Distribution")
    ax.legend(fontsize=8)

    plt.suptitle(
        f"RADE MDP Calibration  |  n={params['n_samples']:,} samples  |  "
        f"{params['days_used']} days",
        fontsize=12,
    )
    plt.tight_layout()

    out = "btc_calibration_plots.png"
    plt.savefig(out, dpi=150)
    print(f"  Plot saved: {out}")
    plt.close()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fit RADE MDP parameters from live data")
    p.add_argument("--days",   type=int,   default=90,
                   help="Number of days of history to use (default: 90)")
    p.add_argument("--source", type=str,   default=None,
                   help="Filter by source: 'live' or 'synthetic' (default: all)")
    p.add_argument("--plot",   action="store_true",
                   help="Save histogram + fitted PDF plot")
    p.add_argument("--output", type=str,   default="calibration_params.json",
                   help="Output JSON file for fitted parameters")
    p.add_argument("--no-db",  action="store_true",
                   help="Skip saving result to calibration_runs table")
    p.add_argument("--notes",  type=str,   default=None,
                   help="Optional notes for this calibration run")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    calibrate(
        days    = args.days,
        source  = args.source,
        plot    = args.plot,
        output  = args.output,
        save_db = not args.no_db,
        notes   = args.notes,
    )
