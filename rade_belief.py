"""
mabse.py
--------
MABSE: Multi-scale Adaptive Belief State Estimator

Novel algorithm for POMDP belief state estimation from real Bitcoin data.

Key contributions vs prior work:
1. Multi-scale temporal belief (short/mid/long) vs single-scale in prior work
2. KDE-based adaptive anomaly threshold (belief-gated) vs fixed thresholds
3. PostgreSQL-backed causal retrieval with time-decay weighting
4. No LLM dependency -- fully data-driven

Architecture:
    PostgreSQL (live_snapshots)
        ↓
    Multi-Scale Belief Estimator
        ├── Short  τ_s = 10 min  → b̂_t^s
        ├── Mid    τ_m = 60 min  → b̂_t^m
        └── Long   τ_l = 1440 min → b̂_t^l
        ↓
    KDE Adaptive Threshold Updater
        → θ_t  (dynamic anomaly threshold)
        ↓
    9-dim state s_t = [m,p,b,Δm,Δp,h, b̂^s, b̂^m, b̂^l]
        ↓
    Actor-Critic (GAE-A2C)

References:
    - Krishnamurthy (2016) POMDP book: belief state foundations
    - Joseph et al. (2020) ICASSP: AC for anomaly detection
    - Lindstrom et al. (2020) Entropy: functional KDE for time series AD
    - iADCPS (2025): KDE-based dynamic threshold for CPS anomaly detection
    - Zhong et al. (2025) Sensors: SAC for controlled sensing with cost
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Time-scale constants (minutes)
# ──────────────────────────────────────────────

TAU_SHORT  =   10.0   # ~1 Bitcoin block interval
TAU_MID    =   60.0   # ~6 blocks (1 hour)
TAU_LONG   = 1440.0   # 1 day (144 blocks)

# KDE bandwidth (Silverman's rule applied offline; this is the default)
KDE_DEFAULT_BANDWIDTH = 5.0  # MB

# Adaptive threshold base false-alarm rate
ALPHA_0 = 0.05  # 5% base anomaly rate (95th percentile)

# Belief-gating sensitivity for threshold adaptation
LAMBDA_ALPHA = 2.0   # higher → more aggressive threshold reduction under high belief


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

@dataclass
class StateEmbedding:
    """
    5-dim normalised embedding for cosine similarity retrieval.
    Eq. (8) from RADE paper, extended with fee_rate.
    """
    mempool_mb:      float
    pending_tx:      int
    inter_block_sec: float
    action_enc:      float   # probe=1, skip=0, escalate=-1
    reward_clip:     float   # clipped to [-1, 1]

    def to_vector(self) -> list[float]:
        return [
            min(self.mempool_mb / 150.0,      1.0),
            min(self.pending_tx / 700_000.0,  1.0),
            min(self.inter_block_sec / 1200.0, 1.0),
            self.action_enc,
            max(-1.0, min(1.0, self.reward_clip)),
        ]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        dot  = sum(x * y for x, y in zip(a, b))
        na   = math.sqrt(sum(x * x for x in a)) + 1e-12
        nb   = math.sqrt(sum(x * x for x in b)) + 1e-12
        return max(0.0, dot / (na * nb))


# ──────────────────────────────────────────────
# Past episode record (loaded from PostgreSQL)
# ──────────────────────────────────────────────

@dataclass
class PastEpisode:
    collected_at:     datetime
    mempool_mb:       float
    pending_tx:       int
    inter_block_sec:  float
    action:           str
    reward:           float
    is_anomalous:     bool

    def embedding(self) -> StateEmbedding:
        enc = {"probe": 1.0, "escalate": -1.0}.get(self.action, 0.0)
        return StateEmbedding(
            mempool_mb      = self.mempool_mb,
            pending_tx      = self.pending_tx,
            inter_block_sec = self.inter_block_sec,
            action_enc      = enc,
            reward_clip     = self.reward,
        )

    def age_minutes(self, now: datetime) -> float:
        diff = now - self.collected_at.replace(tzinfo=timezone.utc) \
            if self.collected_at.tzinfo is None \
            else now - self.collected_at
        return max(0.0, diff.total_seconds() / 60.0)


# ──────────────────────────────────────────────
# KDE Adaptive Threshold
# ──────────────────────────────────────────────

class KDEThreshold:
    """
    Non-parametric adaptive anomaly threshold using Gaussian KDE.

    Given a rolling window of observed mempool sizes, estimates the
    (1 - alpha) quantile of the empirical distribution as the threshold.
    Alpha is reduced when belief is high, making detection more sensitive.

    Reference: Lindstrom et al. (2020) Entropy; iADCPS (2025).
    """

    def __init__(
        self,
        bandwidth:   float = KDE_DEFAULT_BANDWIDTH,
        alpha_0:     float = ALPHA_0,
        lambda_alpha: float = LAMBDA_ALPHA,
        n_grid:      int   = 200,
    ):
        self.bandwidth    = bandwidth
        self.alpha_0      = alpha_0
        self.lambda_alpha = lambda_alpha
        self.n_grid       = n_grid
        self._samples:    list[float] = []

    def update(self, value: float):
        """Add a new observation to the rolling KDE window (capped at 500)."""
        self._samples.append(value)
        if len(self._samples) > 500:
            self._samples.pop(0)

    def _kde_cdf_at(self, x: float) -> float:
        """
        Fast vectorized Gaussian KDE CDF using numpy.
        Uses analytic CDF of each Gaussian component (erf).
        No grid needed — exact integration.
        """
        if not self._samples:
            return 0.0
        try:
            import numpy as np
            from math import erf, sqrt
            samples = np.array(self._samples, dtype=np.float64)
            h = self.bandwidth
            # CDF of mixture = mean of normal CDFs
            z = (x - samples) / (h * sqrt(2))
            try:
                from scipy.special import erf as _erf
                cdfs = 0.5 * (1.0 + _erf(z))
            except ImportError:
                cdfs = 0.5 * (1.0 + np.vectorize(erf)(z))
            return float(np.mean(cdfs))
        except ImportError:
            # numpy not available fallback
            from math import erf, sqrt
            n = len(self._samples)
            h = self.bandwidth
            total = sum(
                0.5 * (1.0 + erf((x - xi) / (h * sqrt(2))))
                for xi in self._samples
            )
            return total / n

    # Keep _kde_cdf as alias for compatibility
    def _kde_cdf(self, x: float) -> float:
        return self._kde_cdf_at(x)

    def _kde_pdf(self, x: float) -> float:
        """Gaussian KDE density at x (kept for compatibility)."""
        n = len(self._samples)
        if n == 0:
            return 0.0
        h = self.bandwidth
        return sum(
            math.exp(-0.5 * ((x - xi) / h) ** 2) / (h * math.sqrt(2 * math.pi))
            for xi in self._samples
        ) / n

    def threshold(self, belief: float) -> float:
        """
        Compute adaptive threshold θ_t.

        Eq. (belief-gated alpha):
            α_t = α_0 · exp(-λ_α · b̄_t)
            θ_t = F̂^{-1}(1 - α_t)

        When belief is high (b̄ → 1):
            α_t → 0  →  threshold lowers  →  more sensitive detection.
        When belief is low (b̄ → 0):
            α_t → α_0  →  threshold = 95th percentile (base rate).
        """
        if len(self._samples) < 10:
            # Not enough data — fall back to paper default
            return 100.0

        alpha_t = self.alpha_0 * math.exp(-self.lambda_alpha * belief)
        target_cdf = 1.0 - alpha_t

        # Binary search for quantile
        lo = min(self._samples)
        hi = max(self._samples) + 3 * self.bandwidth
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if self._kde_cdf_at(mid) < target_cdf:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2.0, 2)

    def n_samples(self) -> int:
        return len(self._samples)


# ──────────────────────────────────────────────
# Multi-Scale Belief State Estimator
# ──────────────────────────────────────────────

@dataclass
class MABSEState:
    """
    9-dimensional augmented state for the MABSE algorithm.

    s_t = [m̃, p̃, b̃, Δm̃, Δp̃, h, b̂^s, b̂^m, b̂^l]
    """
    # Raw normalised features
    norm_mempool:    float   # m̃ = m / 150
    norm_pending:    float   # p̃ = p / 700k
    norm_inter:      float   # b̃ = b / 1200

    # One-step differences
    delta_mempool:   float   # Δm̃
    delta_pending:   float   # Δp̃

    # Previous action encoding
    prev_action_enc: float   # h ∈ {-1, 0, 1}

    # Multi-scale beliefs (the novelty)
    belief_short:    float   # b̂^s  (10-min scale)
    belief_mid:      float   # b̂^m  (60-min scale)
    belief_long:     float   # b̂^l  (1440-min scale)

    # Derived
    @property
    def ensemble_belief(self) -> float:
        """b̄_t = (b̂^s + b̂^m + b̂^l) / 3"""
        return (self.belief_short + self.belief_mid + self.belief_long) / 3.0

    def to_vector(self) -> list[float]:
        return [
            self.norm_mempool,
            self.norm_pending,
            self.norm_inter,
            self.delta_mempool,
            self.delta_pending,
            self.prev_action_enc,
            self.belief_short,
            self.belief_mid,
            self.belief_long,
        ]

    @staticmethod
    def dim() -> int:
        return 9


class MABSE:
    """
    Multi-scale Adaptive Belief State Estimator.

    Loads past episodes from PostgreSQL and computes:
      1. Time-decay weighted cosine similarity retrieval at 3 scales
      2. Scale-specific belief states b̂^s, b̂^m, b̂^l
      3. KDE adaptive anomaly threshold θ_t

    Parameters
    ----------
    db_dsn       : PostgreSQL DSN string
    tau_short    : Short-scale decay constant (minutes)
    tau_mid      : Mid-scale decay constant (minutes)
    tau_long     : Long-scale decay constant (minutes)
    k_neighbors  : Max neighbours per scale
    kde_bandwidth: KDE bandwidth for threshold estimation (MB)
    cache_minutes: How often to refresh the episode cache from DB
    """

    def __init__(
        self,
        db_dsn:        str,
        tau_short:     float = TAU_SHORT,
        tau_mid:       float = TAU_MID,
        tau_long:      float = TAU_LONG,
        k_neighbors:   int   = 6,
        kde_bandwidth: float = KDE_DEFAULT_BANDWIDTH,
        cache_minutes: float = 1.0,
    ):
        self.db_dsn        = db_dsn
        self.tau_short     = tau_short
        self.tau_mid       = tau_mid
        self.tau_long      = tau_long
        self.k_neighbors   = k_neighbors
        self.cache_minutes = cache_minutes

        self._episodes:    list[PastEpisode] = []
        self._last_refresh: Optional[datetime] = None

        self.kde_mem  = KDEThreshold(bandwidth=kde_bandwidth)
        self.kde_int  = KDEThreshold(bandwidth=60.0)   # seconds scale
        self.kde_pend = KDEThreshold(bandwidth=500.0)  # count scale

    # ── DB cache ─────────────────────────────────

    def _refresh_cache(self, force: bool = False):
        """Load recent episodes from PostgreSQL into memory cache."""
        if self._last_refresh is None:
            force = True  # Always force on first call
        now = datetime.now(timezone.utc)
        if not force and self._last_refresh is not None:
            elapsed = (now - self._last_refresh).total_seconds() / 60.0
            if elapsed < self.cache_minutes:
                return

        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(self.db_dsn)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Load last 2 days + training episodes
                cur.execute("""
                    SELECT
                        collected_at, mempool_size_mb, pending_tx,
                        inter_block_time_sec, is_anomalous,
                        source
                    FROM live_snapshots
                    WHERE collected_at >= now() - INTERVAL '30 days'
                    ORDER BY collected_at ASC
                    LIMIT 5000
                """)
                rows = cur.fetchall()
            conn.close()

            self._episodes = []
            for r in rows:
                ep = PastEpisode(
                    collected_at    = r["collected_at"],
                    mempool_mb      = float(r["mempool_size_mb"]),
                    pending_tx      = int(r["pending_tx"]),
                    inter_block_sec = float(r["inter_block_time_sec"]),
                    action          = "skip",    # live snapshots have no action
                    reward          = 0.0,
                    is_anomalous    = bool(r["is_anomalous"]),
                )
                self._episodes.append(ep)
                # Update KDE windows
                self.kde_mem.update(ep.mempool_mb)
                self.kde_int.update(ep.inter_block_sec)
                self.kde_pend.update(float(ep.pending_tx))

            self._last_refresh = now
            log.debug("MABSE cache refreshed: %d episodes", len(self._episodes))

        except Exception as exc:
            log.warning("MABSE DB refresh failed: %s", exc)

    # ── belief computation ────────────────────────

    def _time_weight(self, age_minutes: float, tau: float) -> float:
        """Exponential decay weight: w = exp(-age / tau)."""
        return math.exp(-age_minutes / tau)

    def _belief_at_scale(
        self,
        query_vec:    list[float],
        now:          datetime,
        tau:          float,
        current_step: Optional[int] = None,
    ) -> float:
        """
        Compute b̂_t^τ at a given time scale.

        b̂_t^τ = Σ_k [sim(s_t, s_k) · w_k^τ · 𝟙[z_k=1]] / Σ_k w_k^τ

        Only uses episodes strictly before current time (causal constraint).
        """
        weighted_anomaly = 0.0
        weight_total     = 0.0

        for ep in self._episodes:
            age = ep.age_minutes(now)
            if age <= 0:
                continue  # causal constraint: exclude future

            w   = self._time_weight(age, tau)
            if w < 1e-6:
                continue  # negligible weight → skip

            sim = StateEmbedding.cosine(query_vec, ep.embedding().to_vector())

            weighted_anomaly += sim * w * (1.0 if ep.is_anomalous else 0.0)
            weight_total     += w

        if weight_total < 1e-9:
            return 0.0
        return max(0.0, min(1.0, weighted_anomaly / weight_total))

    # ── public interface ──────────────────────────

    def compute(
        self,
        mempool_mb:      float,
        pending_tx:      int,
        inter_block_sec: float,
        prev_mempool_mb: float  = 0.0,
        prev_pending_tx: int    = 0,
        prev_action:     str    = "skip",
        now:             Optional[datetime] = None,
        current_step:    Optional[int]      = None,
    ) -> MABSEState:
        """
        Compute the 9-dimensional MABSE state.

        Parameters mirror the fields collected by LiveBitcoinCollector.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        self._refresh_cache()

        # Normalise
        nm  = min(mempool_mb      / 150.0,       1.0)
        np_ = min(pending_tx      / 700_000.0,   1.0)
        nb  = min(inter_block_sec / 1_200.0,     1.0)
        dm  = (mempool_mb - prev_mempool_mb)     / 30.0
        dp  = (pending_tx - prev_pending_tx)     / 50_000.0
        h   = {"probe": 1.0, "escalate": -1.0}.get(prev_action, 0.0)

        # Query embedding (action-neutral for retrieval)
        query = StateEmbedding(
            mempool_mb      = mempool_mb,
            pending_tx      = pending_tx,
            inter_block_sec = inter_block_sec,
            action_enc      = 0.0,   # neutral
            reward_clip     = 0.0,
        ).to_vector()

        # Multi-scale beliefs
        b_s = self._belief_at_scale(query, now, self.tau_short,  current_step)
        b_m = self._belief_at_scale(query, now, self.tau_mid,    current_step)
        b_l = self._belief_at_scale(query, now, self.tau_long,   current_step)

        return MABSEState(
            norm_mempool    = nm,
            norm_pending    = np_,
            norm_inter      = nb,
            delta_mempool   = dm,
            delta_pending   = dp,
            prev_action_enc = h,
            belief_short    = b_s,
            belief_mid      = b_m,
            belief_long     = b_l,
        )

    def adaptive_threshold(self, belief: float) -> tuple[float, float, float]:
        """
        Return (θ_mempool, θ_inter, θ_pending) adaptive thresholds.

        θ_t = F̂^{-1}(1 - α_t),  α_t = α_0 · exp(-λ_α · b̄_t)

        Returns paper-default fallback values if KDE has < 10 samples.
        """
        return (
            self.kde_mem.threshold(belief),
            self.kde_int.threshold(belief),
            self.kde_pend.threshold(belief),
        )

    def is_anomalous(
        self,
        state:   MABSEState,
        mempool_mb:      float,
        pending_tx:      int,
        inter_block_sec: float,
    ) -> bool:
        """
        Adaptive anomaly oracle using KDE thresholds.
        Falls back to paper fixed thresholds if data insufficient.
        """
        θ_m, θ_b, θ_p = self.adaptive_threshold(state.ensemble_belief)
        return (
            mempool_mb      > θ_m
            or inter_block_sec > θ_b
            or pending_tx      > θ_p
        )

    def n_episodes(self) -> int:
        return len(self._episodes)
