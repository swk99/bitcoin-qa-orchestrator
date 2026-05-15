"""
rade_train_live.py  (v3 — NeurIPS/ICLR standard fixes)
-------------------------------------------------------
Fixes applied vs v2:
  [1] value_loss 0.5 multiplier (Schulman 2017 PPO standard)
  [2] advantage normalize applied only to policy loss; returns kept raw
  [3] RunningNorm: update() only during collection, not at eval
  [4] entropy coefficient raised to 0.05 (sparse-reward standard)
  [5] KDE rolling window capped at MAX_KDE_SAMPLES=500
  [6] scipy.special.erf replaces np.vectorize(erf)
  [7] Training/eval seed separated (NeurIPS checklist)
  [8] Checkpoint every 500 steps
  [9] DR, mean_cost, efficiency tracked in real time
  [10] Explicit train/eval mode toggle on model

Usage:
    python rade_train_live.py --steps 2000 --ablation-type mabse_full --use-live
    python rade_train_live.py --steps 200  --ablation-type mabse_full --no-live --no-db
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from btc_live_collector import LiveNetworkSnapshot
from rade_belief import MABSE, MABSEState

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

ACTIONS     = ["probe", "skip", "escalate"]
ACTION_COST = {"probe": 0.08, "skip": 0.01, "escalate": 0.12}
ACTION_ENC  = {"probe": 1.0,  "skip": 0.0,  "escalate": -1.0}
STATE_DIM_MABSE = 9
STATE_DIM_BASE  = 6

# KDE rolling window cap
MAX_KDE_SAMPLES = 500


# ──────────────────────────────────────────────
# Running state normaliser (Welford)
# FIX [3]: update() and normalize() are separate.
#          Call update() only during rollout collection.
#          Call normalize() for both rollout + next_state without update.
# ──────────────────────────────────────────────

class RunningNorm:
    def __init__(self, dim: int, eps: float = 1e-8):
        self.n    = 0
        self.mean = torch.zeros(dim)
        self.m2   = torch.zeros(dim)
        self.eps  = eps

    def update(self, x: torch.Tensor):
        """Update running statistics. Call only during training collection."""
        self.n += 1
        d        = x - self.mean
        self.mean = self.mean + d / self.n
        self.m2   = self.m2 + d * (x - self.mean)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize without updating statistics. Safe to call at eval time."""
        if self.n < 2:
            return x
        var = self.m2 / max(self.n - 1, 1)
        return (x - self.mean) / torch.sqrt(var + self.eps)

    def update_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: update then normalize (use during collection only)."""
        self.update(x)
        return self.normalize(x)


# ──────────────────────────────────────────────
# Actor-Critic network
# ──────────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.body   = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, len(ACTIONS))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


# ──────────────────────────────────────────────
# Reward
# ──────────────────────────────────────────────

def utility(anomalous: bool, action: str) -> float:
    if action == "escalate" and anomalous:     return 1.0
    if action == "probe"    and anomalous:     return 0.5
    if action == "skip"     and not anomalous: return 0.4
    return -0.3


def compute_reward(
    anomalous:       bool,
    action:          str,
    ensemble_belief: float,
    terminal_miss:   bool,
    alpha:           float = 1.0,
    lambda_cost:     float = 0.15,
    beta:            float = 0.5,
    gamma_fn:        float = 0.3,
) -> float:
    u   = utility(anomalous, action)
    rag = beta * ensemble_belief if action == "skip" else 0.0
    fn  = gamma_fn if terminal_miss else 0.0
    return alpha * u - lambda_cost * ACTION_COST[action] - rag - fn


# ──────────────────────────────────────────────
# GAE
# FIX [2]: returns are computed from raw (unnormalized) advantages + values.
#          advantage normalisation happens ONLY inside the policy loss.
# ──────────────────────────────────────────────

def compute_gae(
    rewards:    List[float],
    values:     List[float],
    last_value: float,
    gamma:      float = 0.97,
    lam:        float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        advantages : raw GAE advantages (NOT normalized)
        returns    : TD-lambda returns = advantages + values (for critic target)
    """
    adv, gae = [], 0.0
    for t in reversed(range(len(rewards))):
        v_next = last_value if t == len(rewards) - 1 else values[t + 1]
        delta  = rewards[t] + gamma * v_next - values[t]
        gae    = delta + gamma * lam * gae
        adv.append(gae)
    adv.reverse()
    advantages = torch.tensor(adv,    dtype=torch.float32)
    vals_t     = torch.tensor(values, dtype=torch.float32)
    returns    = advantages + vals_t   # critic target (raw, not normalized)
    return advantages, returns


# ──────────────────────────────────────────────
# Metrics tracker (FIX [10])
# ──────────────────────────────────────────────

@dataclass
class MetricsTracker:
    total:     int   = 0
    detected:  int   = 0
    anomalous: int   = 0
    cost_sum:  float = 0.0
    rewards:   List[float] = field(default_factory=list)

    def update(self, anomalous: bool, action: str, reward: float):
        self.total    += 1
        self.cost_sum += ACTION_COST[action]
        self.rewards.append(reward)
        if anomalous:
            self.anomalous += 1
            if action != "skip":
                self.detected += 1

    @property
    def detection_rate(self) -> float:
        return self.detected / max(1, self.anomalous)

    @property
    def mean_cost(self) -> float:
        return self.cost_sum / max(1, self.total)

    @property
    def efficiency(self) -> float:
        return self.detection_rate / max(1e-8, self.mean_cost)

    @property
    def mean_r100(self) -> float:
        window = self.rewards[-100:]
        return sum(window) / max(1, len(window))


# ──────────────────────────────────────────────
# DB Replay Environment
# ──────────────────────────────────────────────

class DBReplayEnv:
    """
    Replays block_event snapshots from PostgreSQL in chronological order.
    No API calls during training → reproducible, no rate limiting.
    """
    def __init__(
        self,
        dsn:     str,
        source:  str = "block_event",
        days:    int = 30,
        seed:    int = 42,
        shuffle: bool = False,
    ):
        self._dsn     = dsn
        self._source  = source
        self._days    = days
        self._shuffle = shuffle
        self._rng     = random.Random(seed)
        self._snaps:  List[LiveNetworkSnapshot] = []
        self._idx     = 0
        self._load()

    def _load(self):
        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(self._dsn)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT collected_at, mempool_size_mb, pending_tx,
                           inter_block_time_sec, fee_rate_fast_sat_vb,
                           fee_rate_med_sat_vb, hashrate_eh_s, difficulty,
                           source
                    FROM live_snapshots
                    WHERE source = %s
                      AND collected_at >= now() - INTERVAL '%s days'
                    ORDER BY collected_at ASC
                """, (self._source, self._days))
                rows = cur.fetchall()
            conn.close()
            self._snaps = [
                LiveNetworkSnapshot(
                    collected_at         = r["collected_at"],
                    mempool_size_mb      = float(r["mempool_size_mb"]),
                    pending_tx           = int(r["pending_tx"]),
                    inter_block_time_sec = float(r["inter_block_time_sec"]),
                    fee_rate_fast_sat_vb = float(r["fee_rate_fast_sat_vb"]),
                    fee_rate_med_sat_vb  = float(r["fee_rate_med_sat_vb"]),
                    hashrate_eh_s        = float(r["hashrate_eh_s"]),
                    difficulty           = float(r["difficulty"]),
                    source               = r["source"],
                )
                for r in rows
            ]
            if self._shuffle:
                self._rng.shuffle(self._snaps)
            print(f"[DBReplay] {len(self._snaps):,} snapshots "
                  f"(source={self._source}, days={self._days})")
        except Exception as e:
            print(f"[DBReplay] Load failed: {e}")
            self._snaps = []

    def step(self) -> LiveNetworkSnapshot:
        if not self._snaps:
            raise RuntimeError("No snapshots in DB. Run btc_block_collector.py first.")
        snap = self._snaps[self._idx % len(self._snaps)]
        self._idx += 1
        if self._idx % len(self._snaps) == 0 and self._shuffle:
            self._rng.shuffle(self._snaps)
        return snap

    def n_snaps(self) -> int:
        return len(self._snaps)


# ──────────────────────────────────────────────
# Synthetic environment fallback
# ──────────────────────────────────────────────

class SyntheticBitcoinEnv:
    def __init__(
        self,
        seed:          int   = 42,
        mu_mempool:    float = 3.531,
        sigma_mempool: float = 0.084,
        lambda_inter:  float = 1 / 592,
    ):
        self.rng           = random.Random(seed)
        self.mu_mempool    = mu_mempool
        self.sigma_mempool = sigma_mempool
        self.lambda_inter  = lambda_inter
        self._m = math.exp(self.rng.gauss(mu_mempool, sigma_mempool))
        self._p = int(self.rng.uniform(300, 650_000))
        self._b = self.rng.expovariate(lambda_inter)

    def step(self) -> LiveNetworkSnapshot:
        new_m   = math.exp(self.rng.gauss(self.mu_mempool, self.sigma_mempool))
        self._m = self._m * 0.9 + new_m * 0.1
        self._p = max(100, int(self._p * 0.9 + self.rng.uniform(300, 650_000) * 0.1))
        self._b = self._b * 0.7 + self.rng.expovariate(self.lambda_inter) * 0.3
        return LiveNetworkSnapshot(
            collected_at         = datetime.now(timezone.utc),
            mempool_size_mb      = round(self._m, 2),
            pending_tx           = self._p,
            inter_block_time_sec = round(self._b, 1),
            fee_rate_fast_sat_vb = round(self.rng.uniform(1, 80), 1),
            fee_rate_med_sat_vb  = round(self.rng.uniform(1, 40), 1),
            hashrate_eh_s        = round(self.rng.uniform(600, 800), 1),
            difficulty           = self.rng.uniform(8e13, 1e14),
            source               = "synthetic",
        )


# ──────────────────────────────────────────────
# Rollout step record
# ──────────────────────────────────────────────

@dataclass
class StepRecord:
    state:      torch.Tensor
    action_idx: int
    value:      float
    reward:     float


# ──────────────────────────────────────────────
# External Baseline Policies
# ──────────────────────────────────────────────

class FixedThresholdPolicy:
    """Rule-based 2019 baseline. Escalate if any signal exceeds threshold."""
    def __init__(self, thresh_mem=100.0, thresh_inter=1100.0, thresh_pend=15_000):
        self.thresh_mem   = thresh_mem
        self.thresh_inter = thresh_inter
        self.thresh_pend  = thresh_pend
        self.metrics = MetricsTracker()

    def act(self, snap: LiveNetworkSnapshot) -> str:
        anomalous = (
            snap.mempool_size_mb > self.thresh_mem
            or snap.inter_block_time_sec > self.thresh_inter
            or snap.pending_tx > self.thresh_pend
        )
        action = "escalate" if anomalous else "skip"
        self.metrics.update(anomalous, action, 0.0)
        return action


class RollingQuantilePolicy:
    """Sliding-window P95 threshold. Most direct KDE competitor."""
    def __init__(self, window: int = 100, quantile: float = 0.95):
        self.window   = window
        self.quantile = quantile
        self._hist:   list[float] = []
        self.metrics  = MetricsTracker()

    def _p95(self) -> float:
        if not self._hist:
            return 100.0
        sv = sorted(self._hist)
        k  = self.quantile * (len(sv) - 1)
        lo, hi = int(k), min(int(k) + 1, len(sv) - 1)
        return sv[lo] * (1 - (k - lo)) + sv[hi] * (k - lo)

    def act(self, snap: LiveNetworkSnapshot) -> str:
        self._hist.append(snap.mempool_size_mb)
        if len(self._hist) > self.window:
            self._hist.pop(0)
        anomalous = (
            snap.mempool_size_mb > self._p95()
            or snap.inter_block_time_sec > 1_100.0
            or snap.pending_tx > 15_000
        )
        action = "escalate" if anomalous else "skip"
        self.metrics.update(anomalous, action, 0.0)
        return action


class EWMAZScorePolicy:
    """EWMA Z-score baseline for burst/spike detection."""
    def __init__(self, alpha: float = 0.1, z_thresh: float = 2.5):
        self.alpha    = alpha
        self.z_thresh = z_thresh
        self._mu      = None
        self._s2      = None
        self.metrics  = MetricsTracker()

    def act(self, snap: LiveNetworkSnapshot) -> str:
        m = snap.mempool_size_mb
        if self._mu is None:
            self._mu, self._s2 = m, 1.0
            action = "skip"
            anomalous = False
        else:
            diff     = m - self._mu
            self._s2 = (1 - self.alpha) * (self._s2 + self.alpha * diff**2)
            self._mu = (1 - self.alpha) * self._mu + self.alpha * m
            z        = abs(diff) / math.sqrt(max(self._s2, 1e-6))
            anomalous = z > self.z_thresh
            action   = "probe" if anomalous else "skip"
        self.metrics.update(anomalous, action, 0.0)
        return action


def run_baselines(snaps: List[LiveNetworkSnapshot], steps: int = 2000) -> List[dict]:
    """Run all external baselines on the same snapshot sequence."""
    pols = [FixedThresholdPolicy(), RollingQuantilePolicy(), EWMAZScorePolicy()]
    seq  = (snaps * (steps // len(snaps) + 1))[:steps]
    for snap in seq:
        for pol in pols:
            pol.act(snap)
    results = []
    print("\n" + "=" * 58)
    print("  External Baselines")
    print(f"  {'Policy':20s} {'DR':>7} {'Cost':>8} {'η':>8}")
    print("─" * 58)
    for pol in pols:
        m = pol.metrics
        r = {"policy": type(pol).__name__,
             "dr": m.detection_rate,
             "cost": m.mean_cost,
             "eta": m.efficiency}
        results.append(r)
        print(f"  {r['policy']:20s} {r['dr']:>7.3f} {r['cost']:>8.4f} {r['eta']:>8.2f}")
    print("=" * 58)
    return results


# ──────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RADE Actor-Critic training (v3)")
    p.add_argument("--steps",          type=int,   default=2000)
    p.add_argument("--rollout",        type=int,   default=64)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--lambda-cost",    type=float, default=0.15)
    p.add_argument("--alpha",          type=float, default=1.0)
    p.add_argument("--beta",           type=float, default=0.5)
    p.add_argument("--gamma-fn",       type=float, default=0.3)
    p.add_argument("--gamma",          type=float, default=0.97)
    p.add_argument("--gae-lambda",     type=float, default=0.95)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--c-v",            type=float, default=0.5)
    # FIX [4]: raised from 0.01 to 0.05 for sparse-reward stability
    p.add_argument("--c-e",            type=float, default=0.05)
    p.add_argument("--reward-clip",    type=float, default=5.0)
    p.add_argument("--hidden",         type=int,   default=128)
    p.add_argument("--use-live",       action="store_true")
    p.add_argument("--no-live",        action="store_true")
    p.add_argument("--db-source",      type=str,   default="block_event",
                   choices=["block_event", "live", "historical", "all"])
    p.add_argument("--db-days",        type=int,   default=30)
    p.add_argument("--db-shuffle",     action="store_true")
    p.add_argument("--ablation-type",
                   choices=["mabse_full", "mabse_no_multiscale",
                             "mabse_no_kde", "ac_base"],
                   default="mabse_full")
    p.add_argument("--no-db",          action="store_true")
    p.add_argument("--policy-name",    type=str,   default="rade_a2c_v3")
    p.add_argument("--notes",          type=str,   default="")
    p.add_argument("--ckpt-dir",       type=str,   default="checkpoints",
                   help="Directory for checkpoints (FIX [8])")
    p.add_argument("--ckpt-every",     type=int,   default=500)
    p.add_argument("--tau-short",      type=float, default=10.0)
    p.add_argument("--tau-mid",        type=float, default=60.0)
    p.add_argument("--tau-long",       type=float, default=1440.0)
    p.add_argument("--kde-bandwidth",  type=float, default=5.0)
    p.add_argument("--mu-mempool",     type=float, default=3.531)
    p.add_argument("--sigma-mempool",  type=float, default=0.084)
    p.add_argument("--lambda-inter",   type=float, default=1/592)
    return p.parse_args()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # FIX [7]: separate training and eval seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    _eval_rng = random.Random(args.seed + 1000)   # reserved for eval

    use_live = args.use_live and not args.no_live

    use_mabse      = args.ablation_type in ("mabse_full", "mabse_no_kde")
    use_kde        = args.ablation_type in ("mabse_full", "mabse_no_multiscale")
    use_multiscale = args.ablation_type == "mabse_full"
    state_dim      = STATE_DIM_MABSE if (use_mabse and use_live) else STATE_DIM_BASE

    dsn = os.environ.get("DATABASE_URL", "postgresql://btcqa:btcqa@localhost:5432/btcqa")

    # ── Environment ────────────────────────────
    if use_live:
        env = DBReplayEnv(
            dsn=dsn, source=args.db_source,
            days=args.db_days, seed=args.seed, shuffle=args.db_shuffle,
        )
        if env.n_snaps() == 0:
            print("[WARN] No snapshots — falling back to synthetic.")
            use_live = False

        mabse = MABSE(
            db_dsn        = dsn,
            tau_short     = args.tau_short,
            tau_mid       = args.tau_mid,
            tau_long      = args.tau_long if use_multiscale else args.tau_mid,
            k_neighbors   = 6,
            kde_bandwidth = args.kde_bandwidth,
        )
        mabse._refresh_cache(force=True)
        # FIX [5]: cap KDE samples
        for kde in [mabse.kde_mem, mabse.kde_int, mabse.kde_pend]:
            kde._max_samples = MAX_KDE_SAMPLES
        print(f"[RADE v3] DB replay. snaps={env.n_snaps():,}  "
              f"belief_eps={mabse.n_episodes():,}")
    else:
        env = SyntheticBitcoinEnv(
            seed=args.seed,
            mu_mempool=args.mu_mempool,
            sigma_mempool=args.sigma_mempool,
            lambda_inter=args.lambda_inter,
        )
        mabse = None
        print(f"[RADE v3] Synthetic mode (state_dim={state_dim})")

    # ── Model ──────────────────────────────────
    model     = ActorCritic(in_dim=state_dim, hidden=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    norm      = RunningNorm(dim=state_dim)

    # FIX [8]: checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # ── DB logging ─────────────────────────────
    db     = None
    run_id = None
    if not args.no_db:
        try:
            from db import EpisodeDB
            db = EpisodeDB.from_env()
            db.init_schema()
            run_id = db.create_experiment(
                policy        = args.policy_name,
                ablation_type = args.ablation_type,
                lambda_cost   = args.lambda_cost,
                seed          = args.seed,
                notes         = args.notes or None,
            )
            print(f"[DB] run_id={run_id}")
        except Exception as e:
            print(f"[DB] skipped: {e}")

    # ── Metrics ────────────────────────────────
    metrics = MetricsTracker()

    # ── State builder ──────────────────────────
    def build_state(
        snap:      LiveNetworkSnapshot,
        prev_snap: Optional[LiveNetworkSnapshot],
        prev_act:  str,
        step:      int,
        update_norm: bool = True,
    ) -> torch.Tensor:
        if use_mabse and mabse is not None:
            ms: MABSEState = mabse.compute(
                mempool_mb      = snap.mempool_size_mb,
                pending_tx      = snap.pending_tx,
                inter_block_sec = snap.inter_block_time_sec,
                prev_mempool_mb = prev_snap.mempool_size_mb if prev_snap else 0.0,
                prev_pending_tx = prev_snap.pending_tx      if prev_snap else 0,
                prev_action     = prev_act,
                current_step    = step,
            )
            raw = torch.tensor(ms.to_vector(), dtype=torch.float32)
        else:
            m  = min(snap.mempool_size_mb / 150.0, 1.0)
            p  = min(snap.pending_tx / 700_000.0,  1.0)
            b  = min(snap.inter_block_time_sec / 1200.0, 1.0)
            dm = (snap.mempool_size_mb - (prev_snap.mempool_size_mb if prev_snap else 0)) / 30.0
            dp = (snap.pending_tx - (prev_snap.pending_tx if prev_snap else 0)) / 50_000.0
            h  = ACTION_ENC.get(prev_act, 0.0)
            raw = torch.tensor([m, p, b, dm, dp, h], dtype=torch.float32)

        # FIX [3]: update running stats only during collection
        if update_norm:
            return norm.update_and_normalize(raw)
        else:
            return norm.normalize(raw)

    def check_anomalous(snap: LiveNetworkSnapshot) -> bool:
        if use_mabse and mabse is not None and use_kde:
            ms_state = mabse.compute(
                mempool_mb      = snap.mempool_size_mb,
                pending_tx      = snap.pending_tx,
                inter_block_sec = snap.inter_block_time_sec,
            )
            return mabse.is_anomalous(
                ms_state, snap.mempool_size_mb,
                snap.pending_tx, snap.inter_block_time_sec,
            )
        return (
            snap.mempool_size_mb > 100.0
            or snap.inter_block_time_sec > 1100.0
            or snap.pending_tx > 15_000
        )

    # ── Training loop ──────────────────────────
    prev_snap:   Optional[LiveNetworkSnapshot] = None
    prev_action: str = "skip"
    rollout_buf: List[StepRecord] = []
    curr_snap = env.step()

    model.train()

    for step in range(1, args.steps + 1):
        # FIX [3]: update_norm=True during rollout collection
        state_n = build_state(curr_snap, prev_snap, prev_action, step, update_norm=True)

        model.eval()
        with torch.no_grad():
            logits, value = model(state_n.unsqueeze(0))
        model.train()

        probs  = F.softmax(logits, dim=-1).squeeze(0)
        dist   = Categorical(probs=probs)
        a_idx  = int(dist.sample().item())
        action = ACTIONS[a_idx]

        anomalous  = check_anomalous(curr_snap)
        ens_belief = 0.0
        if use_mabse and mabse is not None:
            ms_state   = mabse.compute(
                mempool_mb      = curr_snap.mempool_size_mb,
                pending_tx      = curr_snap.pending_tx,
                inter_block_sec = curr_snap.inter_block_time_sec,
                current_step    = step,
            )
            ens_belief = ms_state.ensemble_belief

        terminal_miss = anomalous and action == "skip"
        reward = compute_reward(
            anomalous=anomalous, action=action,
            ensemble_belief=ens_belief, terminal_miss=terminal_miss,
            alpha=args.alpha, lambda_cost=args.lambda_cost,
            beta=args.beta, gamma_fn=args.gamma_fn,
        )
        reward = max(-args.reward_clip, min(args.reward_clip, reward))

        metrics.update(anomalous, action, reward)

        rollout_buf.append(StepRecord(
            state=state_n, action_idx=a_idx,
            value=float(value.item()), reward=float(reward),
        ))

        if db and run_id:
            try:
                from memory import Episode
                db.insert_episode(run_id, Episode(
                    mempool_size=curr_snap.mempool_size_mb,
                    pending_tx=curr_snap.pending_tx,
                    inter_block_time=curr_snap.inter_block_time_sec,
                    action=action, reward=reward,
                    cost=ACTION_COST[action],
                    detected=anomalous and action != "skip",
                    step=step,
                    timestamp=curr_snap.collected_at.timestamp()
                    if hasattr(curr_snap.collected_at, "timestamp") else 0.0,
                ), step=step)
            except Exception:
                pass

        prev_snap   = curr_snap
        prev_action = action
        curr_snap   = env.step()

        # ── GAE-A2C update ─────────────────────
        should_update = len(rollout_buf) >= args.rollout or step == args.steps
        if not should_update:
            continue

        # Bootstrap: eval mode, no norm update
        model.eval()
        next_vec = build_state(curr_snap, prev_snap, prev_action, step + 1, update_norm=False)
        with torch.no_grad():
            _, next_val = model(next_vec.unsqueeze(0))
        bootstrap = float(next_val.item())
        model.train()

        rewards = [r.reward for r in rollout_buf]
        values  = [r.value  for r in rollout_buf]
        # FIX [2]: raw advantages + raw returns
        adv_raw, ret = compute_gae(rewards, values, bootstrap, args.gamma, args.gae_lambda)

        # FIX [2]: normalize only for policy loss
        adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

        states  = torch.stack([r.state for r in rollout_buf])
        actions = torch.tensor([r.action_idx for r in rollout_buf], dtype=torch.long)

        logits_b, values_b = model(states)
        d         = Categorical(logits=logits_b)
        logp_b    = d.log_prob(actions)
        entropy_b = d.entropy().mean()

        policy_loss = -(logp_b * adv_norm.detach()).mean()
        # FIX [1]: 0.5 * c_v (Schulman 2017 standard: c_v=0.5 → total coeff=0.25)
        value_loss  = 0.5 * F.mse_loss(values_b, ret.detach())
        total_loss  = policy_loss + args.c_v * value_loss - args.c_e * entropy_b

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        rollout_buf = []

        # FIX [9]: log DR + cost + efficiency
        print(
            f"[step={step:04d}] "
            f"r100={metrics.mean_r100:+.4f}  "
            f"DR={metrics.detection_rate:.3f}  "
            f"cost={metrics.mean_cost:.4f}  "
            f"η={metrics.efficiency:.2f}  "
            f"belief={ens_belief:.3f}  "
            f"loss={float(total_loss.item()):.4f}  "
            f"{args.ablation_type}"
        )

        # FIX [8]: checkpoint
        if step % args.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"ckpt_{args.ablation_type}_s{args.seed}_t{step}.pt"
            torch.save({
                "step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "norm_mean": norm.mean, "norm_m2": norm.m2, "norm_n": norm.n,
                "metrics": {"dr": metrics.detection_rate,
                            "cost": metrics.mean_cost,
                            "eta": metrics.efficiency},
            }, ckpt_path)
            print(f"[CKPT] saved → {ckpt_path}")

    # ── Final summary ──────────────────────────
    print(f"\n[DONE] steps={args.steps}  ablation={args.ablation_type}  "
          f"seed={args.seed}")
    print(f"  DR={metrics.detection_rate:.4f}  "
          f"cost={metrics.mean_cost:.4f}  "
          f"η={metrics.efficiency:.2f}")

    # Run external baselines on same sequence
    if use_live and hasattr(env, '_snaps') and env._snaps:
        run_baselines(env._snaps, steps=args.steps)

    # DB rollup
    if db and run_id:
        try:
            db.insert_rollup(run_id, {
                "mean_return":    metrics.mean_r100,
                "detection_rate": metrics.detection_rate,
                "mean_cost":      metrics.mean_cost,
                "mean_ttd":       None,
                "efficiency_eta": metrics.efficiency,
                "total_episodes": metrics.total,
            })
            db.close()
            print(f"[DB] rollup saved run_id={run_id}")
        except Exception as e:
            print(f"[DB] rollup failed: {e}")


if __name__ == "__main__":
    main()
