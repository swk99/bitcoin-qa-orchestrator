"""
btc_block_collector.py
----------------------
Block-event-driven Bitcoin network data collector.

Triggers a snapshot ONLY when a new block is confirmed.
This preserves the inter-block arrival distribution as a genuine
Poisson process, eliminating the observer-induced stochastic
distortion from time-driven polling (staircase artefact).

source = "block_event" (distinct from time-driven "live" snapshots)

Usage:
    python btc_block_collector.py
    python btc_block_collector.py --steps 10 --no-db   # test
    python btc_block_collector.py --qq-plot             # save QQ plot on exit
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger(__name__)
_STOP = False

MEMPOOL_BASE = "https://mempool.space/api"
SOURCE = "block_event"   # distinct from time-driven "live"

BANNER = """
╔══════════════════════════════════════════════════════╗
║   RADE — Block-Event Driven Collector                ║
║   source = "block_event"  (calibration-grade data)  ║
╚══════════════════════════════════════════════════════╝"""


def _handle_signal(sig, frame):
    global _STOP
    print(f"\n[RADE] Signal {sig} — stopping.")
    _STOP = True


# ──────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────

def _get(path: str, timeout: int = 10):
    url = f"{MEMPOOL_BASE}{path}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    ct = resp.headers.get("content-type", "")
    if "json" in ct:
        return resp.json()
    try:
        return int(resp.text.strip())
    except ValueError:
        return resp.text.strip()


def fetch_tip_height() -> int:
    return int(_get("/blocks/tip/height"))


def fetch_mempool() -> tuple[float, int]:
    data = _get("/mempool")
    return float(data["vsize"]) / 1_000_000.0, int(data["count"])


def fetch_block_info(height: int) -> dict:
    """Returns {timestamp, hash} for block at given height."""
    block_hash = _get(f"/block-height/{height}")
    block_info = _get(f"/block/{block_hash}")
    return {"timestamp": int(block_info["timestamp"]), "hash": str(block_hash)}


def fetch_fees() -> tuple[float, float]:
    data = _get("/v1/fees/recommended")
    fast = float(data.get("fastestFee", data.get("hourFee", 10.0)))
    med  = float(data.get("halfHourFee", data.get("hourFee", 5.0)))
    return fast, med


# ──────────────────────────────────────────────
# Distribution fitting (no scipy for main fits)
# ──────────────────────────────────────────────

def fit_distributions(values: list[float]) -> dict:
    """
    Fit Exponential, Gamma (method of moments), LogNormal.
    Compute AIC/BIC for model comparison.
    Note: Gamma uses method of moments, not MLE.
    """
    n = len(values)
    if n < 10:
        return {}

    mean_x = sum(values) / n
    var_x  = sum((x - mean_x)**2 for x in values) / (n - 1)

    results = {}

    # Exponential MLE
    lam = 1.0 / mean_x
    ll_exp = sum(math.log(lam) - lam * x for x in values if x > 0)
    results["exponential"] = {
        "params": {"lambda": round(lam, 6), "mean": round(mean_x, 1)},
        "loglik": round(ll_exp, 2),
        "aic":    round(2 * 1 - 2 * ll_exp, 2),
        "bic":    round(math.log(n) * 1 - 2 * ll_exp, 2),
        "method": "MLE",
    }

    # Gamma (method of moments)
    k = mean_x**2 / var_x
    theta = var_x / mean_x
    try:
        def lgamma_approx(z):
            return (z - 0.5) * math.log(z) - z + 0.5 * math.log(2 * math.pi)
        ll_g = sum(
            (k - 1) * math.log(x) - x / theta
            - k * math.log(theta) - lgamma_approx(k)
            for x in values if x > 0
        )
        results["gamma"] = {
            "params": {"k": round(k, 4), "theta": round(theta, 2)},
            "loglik": round(ll_g, 2),
            "aic":    round(2 * 2 - 2 * ll_g, 2),
            "bic":    round(math.log(n) * 2 - 2 * ll_g, 2),
            "method": "method-of-moments",
        }
    except Exception:
        pass

    # LogNormal MLE
    log_vals = [math.log(x) for x in values if x > 0]
    mu_ln = sum(log_vals) / len(log_vals)
    sigma_ln = math.sqrt(
        sum((v - mu_ln)**2 for v in log_vals) / max(len(log_vals) - 1, 1)
    )
    try:
        ll_ln = sum(
            -math.log(x) - math.log(sigma_ln)
            - 0.5 * math.log(2 * math.pi)
            - 0.5 * ((math.log(x) - mu_ln) / sigma_ln)**2
            for x in values if x > 0
        )
        results["lognormal"] = {
            "params": {"mu": round(mu_ln, 4), "sigma": round(sigma_ln, 4)},
            "loglik": round(ll_ln, 2),
            "aic":    round(2 * 2 - 2 * ll_ln, 2),
            "bic":    round(math.log(n) * 2 - 2 * ll_ln, 2),
            "method": "MLE",
        }
    except Exception:
        pass

    # KS statistic (Exponential)
    sv = sorted(values)
    ks = max(
        abs((i + 1) / n - (1 - math.exp(-lam * x)))
        for i, x in enumerate(sv)
    )
    results["ks_exponential"] = round(ks, 4)

    aic_map = {k: v["aic"] for k, v in results.items()
               if isinstance(v, dict) and "aic" in v}
    if aic_map:
        results["best_by_aic"] = min(aic_map, key=aic_map.get)

    return results


def print_distribution_report(inter_vals: list[float], mempool_vals: list[float]):
    print("\n" + "=" * 60)
    print("  Distribution Comparison (paper Table)")
    print("=" * 60)
    for label, vals in [("Inter-block time (s)", inter_vals),
                        ("Mempool size (MB)", mempool_vals)]:
        print(f"\n  {label} (n={len(vals):,})")
        if len(vals) >= 10:
            res = fit_distributions(vals)
            for model in ["exponential", "gamma", "lognormal"]:
                if model in res:
                    r = res[model]
                    print(f"    {model:12s} [{r['method']}]: "
                          f"AIC={r['aic']:8.1f}  BIC={r['bic']:8.1f}  "
                          f"params={r['params']}")
            if "best_by_aic" in res:
                print(f"    Best by AIC : {res['best_by_aic']}")
            if "ks_exponential" in res:
                print(f"    KS(Exp)     : {res['ks_exponential']}")
    print("=" * 60)


def save_qq_plot(inter_vals: list[float], out: str = "qq_interblock.png"):
    """QQ plot for paper. Requires matplotlib (scipy optional for Gamma)."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt

        n = len(inter_vals)
        sv = sorted(inter_vals)
        lam = n / sum(inter_vals)
        theoretical = [-math.log(1 - (i + 0.5) / n) / lam for i in range(n)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(theoretical, sv, alpha=0.4, s=10, color="steelblue")
        lim = max(max(theoretical), max(sv)) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="y=x (perfect fit)")
        ax.set_xlabel("Theoretical Exponential Quantiles (s)")
        ax.set_ylabel("Observed Inter-block Time (s)")
        ax.set_title(f"QQ Plot: Inter-block Time vs Exp\n"
                     f"(n={n:,}, source=block_event, λ={lam:.5f})")
        ax.legend(fontsize=9)

        ax = axes[1]
        xs = np.linspace(0, max(inter_vals), 300)
        ax.hist(inter_vals, bins=40, density=True, alpha=0.5,
                color="steelblue", label="observed")
        ax.plot(xs, lam * np.exp(-lam * xs), "r-", lw=2,
                label=f"Exponential(λ={lam:.5f})")

        # Gamma (method of moments) — no scipy needed
        mean_x = sum(inter_vals) / n
        var_x  = sum((x - mean_x)**2 for x in inter_vals) / (n - 1)
        k = mean_x**2 / var_x
        theta = var_x / mean_x
        gamma_pdf = (xs**(k - 1) * np.exp(-xs / theta)
                     / (theta**k * np.array([math.exp(
                         (k - 0.5) * math.log(k) - k + 0.5 * math.log(2 * math.pi)
                     )]*len(xs))))
        ax.plot(xs, gamma_pdf, "g--", lw=2,
                label=f"Gamma(k={k:.2f}, θ={theta:.1f}) [MOM]")

        ax.set_xlabel("Inter-block Time (s)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution Fit Comparison")
        ax.legend(fontsize=9)

        plt.suptitle("RADE: Block-Event Driven Inter-block Time Analysis",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"[QQ] Saved: {out}")
        plt.close()
    except Exception as e:
        print(f"[QQ] Skipped: {e}")


# ──────────────────────────────────────────────
# Main collector
# ──────────────────────────────────────────────

def run(
    poll_interval: int  = 10,
    no_db:         bool = False,
    steps:         int  = 0,
    report_every:  int  = 10,
    qq_plot:       bool = False,
):
    print(BANNER)
    print(f"\n[설정] poll={poll_interval}s  no_db={no_db}  "
          f"steps={steps or '무제한'}  source={SOURCE}")

    db = None
    if not no_db:
        from btc_live_db import LiveSnapshotDB
        db = LiveSnapshotDB.from_env()
        db.init_schema()

        # Ensure extra columns exist
        with db.conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE live_snapshots
                ADD COLUMN IF NOT EXISTS is_fraud BOOLEAN NOT NULL DEFAULT FALSE;
            """)
            cur.execute("""
                ALTER TABLE live_snapshots
                ADD COLUMN IF NOT EXISTS block_height INTEGER;
            """)
            cur.execute("""
                ALTER TABLE live_snapshots
                ADD COLUMN IF NOT EXISTS collector_type TEXT DEFAULT 'time_polling';
            """)
            # Unique index: prevent duplicate block snapshots
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_live_block_height
                ON live_snapshots(block_height)
                WHERE collector_type = 'block_event';
            """)
        db.conn.commit()
        existing = db.count(source=SOURCE)
        print(f"[DB] 연결 완료. 기존 block_event 스냅샷: {existing:,}개\n")

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    last_height:   Optional[int] = None
    last_block_ts: Optional[int] = None  # only updated from real block timestamps
    step           = 0
    total_saved    = 0
    inter_vals:    list[float] = []
    mempool_vals:  list[float] = []

    print("─" * 72)
    print(f"  {'Block':>8}  {'Mempool':>10}  {'Pending':>10}  "
          f"{'Inter(s)':>9}  {'Fee':>6}  Flags")
    print("─" * 72)

    while not _STOP:
        if steps and step >= steps:
            break

        # Poll tip height
        try:
            tip = fetch_tip_height()
        except Exception as e:
            log.warning("tip poll failed: %s", e)
            time.sleep(poll_interval)
            continue

        if tip == last_height:
            time.sleep(poll_interval)
            continue

        # ── New block! ────────────────────────
        step += 1

        try:
            block = fetch_block_info(tip)
            block_ts = block["timestamp"]

            # Compute real inter-block time
            if last_block_ts is None:
                # First observation: record timestamp but DO NOT save
                # (no valid inter_block available)
                last_height   = tip
                last_block_ts = block_ts
                print(f"  [{tip:>8,}] First block recorded — "
                      f"inter_block pending, snapshot skipped.")
                continue

            inter_block = max(1.0, float(block_ts - last_block_ts))

            # Fetch mempool state at block arrival
            mempool_mb, pending_tx = fetch_mempool()

            try:
                fee_fast, fee_med = fetch_fees()
            except Exception:
                fee_fast, fee_med = 10.0, 5.0

            # Fraud pattern
            is_fraud = (
                (pending_tx > 50_000 and fee_fast < 2.0 and mempool_mb < 35.0)
                or fee_fast > 100.0
                or (mempool_mb > 50.0 and fee_fast < 1.5)
            )
            is_anomalous = (
                mempool_mb > 100.0
                or inter_block > 1_100.0
                or pending_tx > 15_000
            )

            # Track for distribution analysis
            inter_vals.append(inter_block)
            mempool_vals.append(mempool_mb)
            total_saved += 1

            # Update state ONLY from real block timestamps
            last_height   = tip
            last_block_ts = block_ts

            flag = "⚠" if is_anomalous else " "
            frd  = "🚨" if is_fraud    else "  "
            print(
                f"  {tip:>8,}  {mempool_mb:>9.2f}MB  "
                f"{pending_tx:>10,}  {inter_block:>9.0f}s  "
                f"{fee_fast:>5.1f}  {frd}{flag}",
                flush=True,
            )

            # DB insert with block_height and collector_type
            if db:
                try:
                    sql = """
                        INSERT INTO live_snapshots
                          (collected_at, mempool_size_mb, pending_tx,
                           inter_block_time_sec, fee_rate_fast_sat_vb,
                           fee_rate_med_sat_vb, hashrate_eh_s, difficulty,
                           source, is_anomalous, is_fraud, error,
                           block_height, collector_type)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT DO NOTHING
                    """
                    with db.conn.cursor() as cur:
                        cur.execute(sql, (
                            datetime.fromtimestamp(block_ts, tz=timezone.utc),
                            round(mempool_mb, 4), pending_tx,
                            round(inter_block, 1), round(fee_fast, 2),
                            round(fee_med, 2), 0.0, 0.0,
                            SOURCE, is_anomalous, is_fraud, None,
                            tip, "block_event"
                        ))
                    db.conn.commit()
                except Exception as e:
                    log.warning("DB insert failed: %s", e)

            # Periodic distribution report
            if total_saved % report_every == 0 and len(inter_vals) >= 10:
                print_distribution_report(inter_vals, mempool_vals)

        except Exception as e:
            log.warning("Block %s collection error: %s", tip, e)
            # Advance height to avoid infinite retry
            # BUT do NOT update last_block_ts (preserve valid timestamp)
            last_height = tip

    # ── Final report ──────────────────────────
    print(f"\n[완료] 저장: {total_saved:,}개  (source={SOURCE})")
    if len(inter_vals) >= 10:
        mean_ib = sum(inter_vals) / len(inter_vals)
        print(f"  Inter-block 평균: {mean_ib:.1f}s  (이론: 600s)")
        print_distribution_report(inter_vals, mempool_vals)
        if qq_plot:
            save_qq_plot(inter_vals)

    if db:
        db.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Block-event driven Bitcoin data collector"
    )
    p.add_argument("--poll-interval", type=int, default=10)
    p.add_argument("--no-db",         action="store_true")
    p.add_argument("--steps",         type=int, default=0)
    p.add_argument("--report-every",  type=int, default=10)
    p.add_argument("--qq-plot",       action="store_true")
    p.add_argument("--status",        action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    if args.status:
        from btc_live_runner import print_status
        print_status()
        sys.exit(0)
    run(
        poll_interval = args.poll_interval,
        no_db         = args.no_db,
        steps         = args.steps,
        report_every  = args.report_every,
        qq_plot       = args.qq_plot,
    )
