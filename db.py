"""
btc_qa/db.py
------------
PostgreSQL persistence layer for Bitcoin node diagnostic agent.
Stores full episode logs and experiment results for structured
queries and aggregation (complement to ChromaDB vector store).

Schema:
    episodes       — per-episode telemetry + decision + reward
    experiments    — experiment metadata (lambda, seed, policy)
    rollup_stats   — aggregated metrics per experiment run

Usage:
    db = EpisodeDB.from_env()          # reads DATABASE_URL
    db = EpisodeDB(dsn="postgresql://user:pw@localhost/btcqa")
    run_id = db.create_experiment(lambda_cost=0.15, policy="signal_escalate", seed=42)
    db.insert_episode(run_id, episode, step=1)
    stats = db.get_experiment_stats(run_id)
    df = db.to_dataframe(run_id)       # pandas DataFrame
"""

import os
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
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

from memory import Episode


DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    policy       TEXT NOT NULL,
    lambda_cost  REAL NOT NULL,
    seed         INTEGER NOT NULL,
    episodes     INTEGER NOT NULL DEFAULT 0,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id              BIGSERIAL PRIMARY KEY,
    run_id          UUID NOT NULL REFERENCES experiments(run_id) ON DELETE CASCADE,
    step            INTEGER NOT NULL,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    mempool_size    REAL NOT NULL,
    pending_tx      INTEGER NOT NULL,
    inter_block_time REAL NOT NULL,
    action          TEXT NOT NULL,
    reward          REAL NOT NULL,
    cost            REAL NOT NULL,
    detected        BOOLEAN NOT NULL,
    chroma_id       TEXT
);

CREATE TABLE IF NOT EXISTS rollup_stats (
    run_id           UUID PRIMARY KEY REFERENCES experiments(run_id) ON DELETE CASCADE,
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    mean_return      REAL,
    detection_rate   REAL,
    mean_cost        REAL,
    mean_ttd         REAL,
    efficiency_eta   REAL,
    total_episodes   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id);
CREATE INDEX IF NOT EXISTS idx_episodes_action ON episodes(action);
CREATE INDEX IF NOT EXISTS idx_episodes_detected ON episodes(detected);
"""


class EpisodeDB:
    """PostgreSQL-backed episode store."""

    def __init__(self, dsn: str):
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2 not installed: pip install psycopg2-binary")
        self.dsn = dsn
        self._conn = None

    @classmethod
    def from_env(cls) -> "EpisodeDB":
        dsn = os.environ.get(
            "DATABASE_URL",
            "postgresql://btcqa:btcqa@localhost:5432/btcqa"
        )
        return cls(dsn=dsn)

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = False
        return self._conn

    def init_schema(self):
        """Create tables if they don't exist."""
        with self.conn.cursor() as cur:
            cur.execute(DDL)
        self.conn.commit()

    def create_experiment(
        self,
        policy: str,
        lambda_cost: float,
        seed: int,
        notes: Optional[str] = None,
    ) -> str:
        """Insert experiment metadata, return run_id (UUID string)."""
        sql = """
            INSERT INTO experiments (policy, lambda_cost, seed, notes)
            VALUES (%s, %s, %s, %s)
            RETURNING run_id::text
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (policy, lambda_cost, seed, notes))
            run_id = cur.fetchone()[0]
        self.conn.commit()
        return run_id

    def insert_episode(self, run_id: str, episode: Episode, step: int, chroma_id: Optional[str] = None):
        """Persist one episode record."""
        sql = """
            INSERT INTO episodes
                (run_id, step, mempool_size, pending_tx, inter_block_time,
                 action, reward, cost, detected, chroma_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                run_id, step,
                round(episode.mempool_size, 3),
                int(episode.pending_tx),
                round(episode.inter_block_time, 1),
                episode.action,
                round(episode.reward, 6),
                round(episode.cost, 6),
                bool(episode.detected),
                chroma_id,
            ))
            # update episode count on experiment
            cur.execute(
                "UPDATE experiments SET episodes = episodes + 1 WHERE run_id = %s",
                (run_id,)
            )
        self.conn.commit()

    def insert_rollup(self, run_id: str, stats: dict):
        """Upsert aggregated stats for a completed experiment."""
        sql = """
            INSERT INTO rollup_stats
                (run_id, mean_return, detection_rate, mean_cost,
                 mean_ttd, efficiency_eta, total_episodes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                computed_at    = now(),
                mean_return    = EXCLUDED.mean_return,
                detection_rate = EXCLUDED.detection_rate,
                mean_cost      = EXCLUDED.mean_cost,
                mean_ttd       = EXCLUDED.mean_ttd,
                efficiency_eta = EXCLUDED.efficiency_eta,
                total_episodes = EXCLUDED.total_episodes
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                run_id,
                stats.get("mean_return"),
                stats.get("detection_rate"),
                stats.get("mean_cost"),
                stats.get("mean_ttd"),
                stats.get("efficiency_eta"),
                stats.get("total_episodes"),
            ))
        self.conn.commit()

    def get_experiment_stats(self, run_id: str) -> dict:
        """Compute live stats from raw episodes table."""
        sql = """
            SELECT
                COUNT(*)                          AS total,
                AVG(reward)                       AS mean_return,
                AVG(cost)                         AS mean_cost,
                SUM(detected::int)::float / COUNT(*) AS detection_rate,
                AVG(CASE WHEN detected THEN step END) AS mean_ttd
            FROM episodes
            WHERE run_id = %s
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (run_id,))
            row = cur.fetchone()
        return dict(row) if row else {}

    def get_lambda_pareto(self) -> list[dict]:
        """
        Return cost vs detection_rate per lambda value across all experiments.
        Used to plot the Pareto frontier figure.
        """
        sql = """
            SELECT
                e.lambda_cost,
                e.policy,
                AVG(ep.reward)                        AS mean_return,
                AVG(ep.cost)                          AS mean_cost,
                SUM(ep.detected::int)::float/COUNT(*) AS detection_rate
            FROM experiments e
            JOIN episodes ep ON ep.run_id = e.run_id
            GROUP BY e.lambda_cost, e.policy
            ORDER BY e.lambda_cost
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]

    def to_dataframe(self, run_id: str):
        """Return all episodes for a run as a pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas not installed: pip install pandas")
        import pandas as pd
        sql = "SELECT * FROM episodes WHERE run_id = %s ORDER BY step"
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (run_id,))
            rows = cur.fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
