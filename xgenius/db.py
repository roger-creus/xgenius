"""SQLite database for xgenius operational state.

Single source of truth for all job and hypothesis tracking.
Automatically maintained by xgenius — the LLM only queries it
and adds hypotheses/comments.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager

from xgenius.config import XGeniusConfig, get_xgenius_dir, ensure_xgenius_dir


@contextmanager
def _connect(db_path: str):
    """Context manager for SQLite connections with WAL mode."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Allow concurrent reads
    conn.execute("PRAGMA busy_timeout=30000")  # Wait up to 30s on locks
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class XGeniusDB:
    """SQLite database for xgenius operational state.

    Tables:
    - jobs: tracks every submitted job (auto-maintained by xgenius)
    - hypotheses: tracks research hypotheses (LLM creates, xgenius + LLM update)
    """

    def __init__(self, config: XGeniusConfig):
        self.config = config
        xgenius_dir = ensure_xgenius_dir(config)
        self.db_path = os.path.join(xgenius_dir, "xgenius.db")
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with _connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    cluster TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    hypothesis_id TEXT DEFAULT '',
                    command TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'submitted',
                    exit_code INTEGER DEFAULT NULL,
                    submitted_at TEXT NOT NULL,
                    started_at TEXT DEFAULT NULL,
                    completed_at TEXT DEFAULT NULL,
                    walltime_seconds REAL DEFAULT 0,
                    gpu_hours REAL DEFAULT 0,
                    gpus INTEGER DEFAULT 1,
                    gpu_type TEXT DEFAULT '',
                    cpus INTEGER DEFAULT 8,
                    memory TEXT DEFAULT '',
                    walltime_requested TEXT DEFAULT '',
                    log_file TEXT DEFAULT '',
                    results_pulled INTEGER DEFAULT 0,
                    output_dir TEXT DEFAULT '',
                    error_message TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    motivation TEXT DEFAULT '',
                    expected_outcome TEXT DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'proposed',
                    conclusion TEXT DEFAULT '',
                    comment TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            # Index for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_hypothesis ON jobs(hypothesis_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status)")

    # =========================================================================
    # JOBS — automatically maintained by xgenius
    # =========================================================================

    def record_job(self, job_id: str, cluster: str, experiment_id: str,
                   hypothesis_id: str, command: str, log_file: str = "",
                   gpus: int = 1, gpu_type: str = "", cpus: int = 8,
                   memory: str = "", walltime: str = "") -> None:
        """Record a newly submitted job."""
        with _connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jobs
                (job_id, cluster, experiment_id, hypothesis_id, command, status,
                 submitted_at, log_file, gpus, gpu_type, cpus, memory, walltime_requested)
                VALUES (?, ?, ?, ?, ?, 'submitted', ?, ?, ?, ?, ?, ?, ?)
            """, (job_id, cluster, experiment_id, hypothesis_id, command,
                  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  log_file, gpus, gpu_type, cpus, memory, walltime))

    def update_job_status(self, job_id: str, status: str, **kwargs) -> None:
        """Update a job's status and optional fields.

        kwargs can include: exit_code, walltime_seconds, gpu_hours,
        completed_at, started_at, results_pulled, output_dir, error_message
        """
        fields = ["status=?"]
        values = [status]

        for key in ["exit_code", "walltime_seconds", "gpu_hours", "completed_at",
                     "started_at", "results_pulled", "output_dir", "error_message"]:
            if key in kwargs:
                fields.append(f"{key}=?")
                values.append(kwargs[key])

        values.append(job_id)
        with _connect(self.db_path) as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id=?", values)

    def mark_completed(self, job_id: str, exit_code: int, walltime_seconds: float,
                       completed_at: str, output_dir: str = "") -> None:
        """Mark a job as completed (success or failure based on exit_code)."""
        status = "completed" if exit_code == 0 else "failed"
        # Estimate GPU hours
        gpu_hours = 0
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT gpus FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row:
                gpu_hours = row["gpus"] * (walltime_seconds / 3600)
            conn.execute("""
                UPDATE jobs SET status=?, exit_code=?, walltime_seconds=?,
                gpu_hours=?, completed_at=?, output_dir=?
                WHERE job_id=?
            """, (status, exit_code, walltime_seconds, gpu_hours, completed_at, output_dir, job_id))

    def mark_results_pulled(self, job_id: str) -> None:
        """Mark that results have been pulled for a job."""
        with _connect(self.db_path) as conn:
            conn.execute("UPDATE jobs SET results_pulled=1 WHERE job_id=?", (job_id,))

    def get_job(self, job_id: str) -> dict | None:
        """Get a single job by ID."""
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def get_jobs_by_hypothesis(self, hypothesis_id: str) -> list[dict]:
        """Get all jobs for a hypothesis."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE hypothesis_id=? ORDER BY submitted_at", (hypothesis_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_jobs_by_status(self, status: str) -> list[dict]:
        """Get all jobs with a given status."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY submitted_at", (status,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_pending_jobs(self) -> list[dict]:
        """Get all jobs that are submitted or running."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('submitted', 'running') ORDER BY submitted_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_pending_job_ids(self) -> set[str]:
        """Get set of job IDs that are submitted or running."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE status IN ('submitted', 'running')"
            ).fetchall()
            return {r["job_id"] for r in rows}

    def get_completed_not_pulled(self) -> list[dict]:
        """Get completed jobs whose results haven't been pulled yet."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('completed', 'failed') AND results_pulled=0"
            ).fetchall()
            return [dict(r) for r in rows]

    def is_hypothesis_complete(self, hypothesis_id: str) -> bool:
        """Check if ALL jobs for a hypothesis have finished (completed/failed/cancelled)."""
        with _connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT COUNT(*) as pending FROM jobs
                WHERE hypothesis_id=? AND status IN ('submitted', 'running')
            """, (hypothesis_id,)).fetchone()
            return row["pending"] == 0

    def get_hypothesis_job_summary(self, hypothesis_id: str) -> dict:
        """Get a summary of job statuses for a hypothesis."""
        with _connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT status, COUNT(*) as count FROM jobs
                WHERE hypothesis_id=? GROUP BY status
            """, (hypothesis_id,)).fetchall()
            summary = {r["status"]: r["count"] for r in rows}
            total = sum(summary.values())
            summary["total"] = total
            summary["all_done"] = summary.get("submitted", 0) + summary.get("running", 0) == 0
            return summary

    def get_all_jobs(self, limit: int = 500) -> list[dict]:
        """Get all jobs, most recent first."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY submitted_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_full_status(self) -> dict:
        """Get a comprehensive status overview for Claude's wake-up prompt."""
        with _connect(self.db_path) as conn:
            # Overall counts
            total = conn.execute("SELECT COUNT(*) as n FROM jobs").fetchone()["n"]
            by_status = conn.execute(
                "SELECT status, COUNT(*) as n FROM jobs GROUP BY status"
            ).fetchall()
            status_counts = {r["status"]: r["n"] for r in by_status}

            # Per-hypothesis breakdown
            hyp_rows = conn.execute("""
                SELECT hypothesis_id, status, COUNT(*) as n
                FROM jobs WHERE hypothesis_id != ''
                GROUP BY hypothesis_id, status
                ORDER BY hypothesis_id
            """).fetchall()

            hypotheses = {}
            for r in hyp_rows:
                hid = r["hypothesis_id"]
                if hid not in hypotheses:
                    hypotheses[hid] = {}
                hypotheses[hid][r["status"]] = r["n"]

            # Add hypothesis metadata
            hyp_meta = conn.execute("SELECT * FROM hypotheses ORDER BY created_at").fetchall()
            hyp_info = {}
            for h in hyp_meta:
                hid = h["hypothesis_id"]
                job_counts = hypotheses.get(hid, {})
                pending = job_counts.get("submitted", 0) + job_counts.get("running", 0)
                hyp_info[hid] = {
                    "description": h["description"][:80],
                    "status": h["status"],
                    "jobs": job_counts,
                    "all_done": pending == 0,
                    "ready_for_analysis": pending == 0 and (job_counts.get("completed", 0) > 0),
                }

            return {
                "total_jobs": total,
                "job_status_counts": status_counts,
                "hypotheses": hyp_info,
            }

    # =========================================================================
    # HYPOTHESES — LLM creates, both LLM and xgenius update
    # =========================================================================

    def add_hypothesis(self, hypothesis_id: str, description: str,
                       motivation: str = "", expected_outcome: str = "") -> None:
        """Record a new hypothesis."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with _connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO hypotheses
                (hypothesis_id, description, motivation, expected_outcome,
                 status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'proposed', ?, ?)
            """, (hypothesis_id, description, motivation, expected_outcome, now, now))

    def update_hypothesis(self, hypothesis_id: str, status: str = "",
                          conclusion: str = "", comment: str = "") -> None:
        """Update a hypothesis status/conclusion/comment."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        fields = ["updated_at=?"]
        values = [now]

        if status:
            fields.append("status=?")
            values.append(status)
        if conclusion:
            fields.append("conclusion=?")
            values.append(conclusion)
        if comment:
            fields.append("comment=?")
            values.append(comment)

        values.append(hypothesis_id)
        with _connect(self.db_path) as conn:
            conn.execute(f"UPDATE hypotheses SET {', '.join(fields)} WHERE hypothesis_id=?", values)

    def get_hypothesis(self, hypothesis_id: str) -> dict | None:
        """Get a single hypothesis."""
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM hypotheses WHERE hypothesis_id=?", (hypothesis_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_hypotheses(self) -> list[dict]:
        """Get all hypotheses."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses ORDER BY created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_hypotheses_by_status(self, status: str) -> list[dict]:
        """Get hypotheses by status."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE status=? ORDER BY created_at", (status,)
            ).fetchall()
            return [dict(r) for r in rows]

    # =========================================================================
    # UTILITY
    # =========================================================================

    def reset(self) -> None:
        """Clear all data (called by xgenius reset)."""
        with _connect(self.db_path) as conn:
            conn.execute("DELETE FROM jobs")
            conn.execute("DELETE FROM hypotheses")

    def build_wakeup_prompt(self, completions: list = None, reconciled_ids: list = None) -> str:
        """Build the prompt sent to Claude when the watcher triggers it.

        This is a FRESH session — Claude has no prior context.
        The prompt must be fully self-contained with everything Claude needs.
        """
        status = self.get_full_status()
        parts = []

        parts.append("You are an autonomous research agent. Read CLAUDE.md and research_goal.md for full instructions.")
        parts.append("This is a fresh session — review all state before acting.\n")

        # What just happened
        if completions:
            parts.append("## What just happened:")
            for c in completions:
                result = "SUCCESS" if c.exit_code == 0 else f"FAILED (exit={c.exit_code})"
                parts.append(f"- {c.experiment_id} (job {c.job_id}, {c.cluster}): {result}")
            parts.append("")

        if reconciled_ids:
            parts.append(f"## {len(reconciled_ids)} job(s) disappeared from SLURM (preempted/killed/timeout):")
            for jid in reconciled_ids:
                parts.append(f"- Job {jid}")
            parts.append("")

        # Overall status
        sc = status["job_status_counts"]
        if sc:
            parts.append(f"## Job overview: {status['total_jobs']} total — " +
                          ", ".join(f"{v} {k}" for k, v in sorted(sc.items())))
        parts.append("")

        # Per-hypothesis breakdown
        if status["hypotheses"]:
            parts.append("## Hypothesis status:")
            for hid, info in sorted(status["hypotheses"].items()):
                jobs = info["jobs"]
                counts = ", ".join(f"{v} {k}" for k, v in sorted(jobs.items()))
                if info["ready_for_analysis"]:
                    tag = "READY FOR ANALYSIS"
                elif info["all_done"]:
                    tag = "ALL DONE (check for failures)"
                else:
                    pending = jobs.get("submitted", 0) + jobs.get("running", 0)
                    tag = f"WAITING ({pending} running)"

                parts.append(f"- **{hid}** [{info['status']}]: {counts} — **{tag}**")
                parts.append(f"  {info['description']}")
            parts.append("")

        # What to do
        parts.append("## Your tasks (in order):")
        parts.append("1. Read CLAUDE.md for full tool documentation and conventions")
        parts.append("2. Read research_goal.md for the research objective")
        parts.append("3. Check results bank: `cat results/experiments.csv` and `cat results/hypotheses.csv`")
        parts.append("4. For READY FOR ANALYSIS hypotheses: pull results (`xgenius pull --cluster NAME`), parse the CSVs from `results/CLUSTER/`, update the results bank, update hypothesis status")
        parts.append("5. For WAITING hypotheses: do NOT conclude — wait for all experiments to finish")
        parts.append("6. If all hypotheses are analyzed: formulate new hypotheses and submit new experiments")
        parts.append("7. Commit and push all changes before exiting")
        parts.append("8. Exit when done — the watcher will trigger you again when more jobs complete")

        return "\n".join(parts)
