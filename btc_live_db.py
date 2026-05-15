"""
btc_live_db.py
--------------
PostgreSQL persistence for live Bitcoin network snapshots.
Extends the existing db.py schema with a dedicated live_snapshots table.

Schema additions:
    live_snapshots   — raw time-series from mempool.space
    calibration_runs — MDP parameter fit history

Usage:
    ldb = LiveSnapshotDB.from_env()
    ldb.init_schema()
    ldb.insert(snap)
    df = ldb.to_dataframe(days=7)
    params = ldb.latest_calibration_params()
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from btc_live_collector import LiveNetworkSnapshot

# ──────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────

DDL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Raw live snapshots from mempool.space
CREATE TABLE IF NOT EXISTS live_snapshots (
    id                    BIGSERIAL PRIMARY KEY,
    collected_at          TIMESTAMPTZ NOT NULL,
    mempool_size_mb       REAL        NOT NULL,
    pending_tx            INTEGER     NOT NULL,
    inter_block_time_sec  REAL        NOT NULL,
    fee_rate_fast_sat_vb  REAL        NOT NULL DEFAULT 0.0,
    fee_rate_med_sat_vb   REAL        NOT NULL DEFAULT 0.0,
    hashrate_eh_s         REAL        NOT NULL DEFAULT 0.0,
    difficulty            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    source                TEXT        NOT NULL DEFAULT 'live',
    is_anomalous          BOOLEAN     NOT NULL,
    error                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_live_collected_at
    ON live_snapshots(collected_at DESC);

CREATE INDEX IF NOT EXISTS idx_live_anomalous
    ON live_snapshots(is_anomalous, collected_at DESC);

-- MDP calibration parameter history
CREATE TABLE IF NOT EXISTS calibration_runs (
    id              BIGSERIAL PRIMARY KEY,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    days_used       INTEGER     NOT NULL,
    n_samples       INTEGER     NOT NULL,
    -- LogNormal(mu, sigma) fit for mempool_size_mb
    mu_mempool      REAL        NOT NULL,
    sigma_mempool   REAL        NOT NULL,
    -- Exponential(lambda) fit for inter_block_time_sec
    lambda_inter    REAL        NOT NULL,
    -- Empirical anomaly rate
    anomaly_rate    REAL        NOT NULL,
    -- 95th percentile thresholds actually observed
    p95_mempool_mb  REAL        NOT NULL,
    p95_inter_sec   REAL        NOT NULL,
    p95_pending     INTEGER     NOT NULL,
    notes           TEXT
);
"""

# ──────────────────────────────────────────────
# DB class
# ──────────────────────────────────────────────

class LiveSnapshotDB:
    """PostgreSQL store for live Bitcoin snapshots and calibration results."""

    def __init__(self, dsn: str):
        if not HAS_PSYCOPG2:
            raise ImportError("pip install psycopg2-binary")
        self.dsn   = dsn
        self._conn = None

    @classmethod
    def from_env(cls) -> "LiveSnapshotDB":
        dsn = os.environ.get(
            "DATABASE_URL",
            "postgresql://btcqa:btcqa@localhost:5432/btcqa",
        )
        return cls(dsn=dsn)

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = False
        return self._conn

    def init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(DDL)
        self.conn.commit()

    # ── insert ──────────────────────────────────

    def insert(self, snap: LiveNetworkSnapshot) -> int:
        """Insert one snapshot. Returns the new row id."""
        sql = """
            INSERT INTO live_snapshots
              (collected_at, mempool_size_mb, pending_tx,
               inter_block_time_sec, fee_rate_fast_sat_vb, fee_rate_med_sat_vb,
               hashrate_eh_s, difficulty, source, is_anomalous, error)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                snap.collected_at,
                snap.mempool_size_mb,
                snap.pending_tx,
                snap.inter_block_time_sec,
                snap.fee_rate_fast_sat_vb,
                snap.fee_rate_med_sat_vb,
                snap.hashrate_eh_s,
                snap.difficulty,
                snap.source,
                snap.is_anomalous(),
                snap.error,
            ))
            row_id = cur.fetchone()[0]
        self.conn.commit()
        return row_id

    # ── query ────────────────────────────────────

    def count(self, source: Optional[str] = None) -> int:
        sql = "SELECT COUNT(*) FROM live_snapshots"
        args: tuple = ()
        if source:
            sql += " WHERE source = %s"
            args = (source,)
        with self.conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchone()[0]

    def to_dataframe(self, days: int = 90, source: Optional[str] = None):
        """Return recent snapshots as a pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pip install pandas")
        sql = """
            SELECT * FROM live_snapshots
            WHERE collected_at >= now() - INTERVAL '%s days'
        """
        args: list = [days]
        if source:
            sql += " AND source = %s"
            args.append(source)
        sql += " ORDER BY collected_at ASC"
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def insert_calibration(self, params: dict) -> int:
        sql = """
            INSERT INTO calibration_runs
              (days_used, n_samples,
               mu_mempool, sigma_mempool, lambda_inter,
               anomaly_rate,
               p95_mempool_mb, p95_inter_sec, p95_pending,
               notes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                params["days_used"],
                params["n_samples"],
                params["mu_mempool"],
                params["sigma_mempool"],
                params["lambda_inter"],
                params["anomaly_rate"],
                params["p95_mempool_mb"],
                params["p95_inter_sec"],
                params["p95_pending"],
                params.get("notes"),
            ))
            row_id = cur.fetchone()[0]
        self.conn.commit()
        return row_id

    def latest_calibration_params(self) -> Optional[dict]:
        """Return the most recent calibration run, or None."""
        sql = """
            SELECT * FROM calibration_runs
            ORDER BY computed_at DESC LIMIT 1
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            row = cur.fetchone()
        return dict(row) if row else None

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
