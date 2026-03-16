"""Results bank utilities for xgenius.

Provides a simple API for the agent to query, append to, and analyze
the results CSV bank. Created by xgenius init in the project directory.

The agent should also build project-specific analysis tools on top of these.
"""

import csv
import os
from typing import Any


class ResultsBank:
    """Query and manage the CSV results bank.

    The results bank is a CSV file at results/all_results.csv with:
    - Required columns: experiment_id, hypothesis_id, command, comment, status
    - Metric columns: project-dependent, defined by the agent

    Usage:
        from xgenius.results import ResultsBank
        bank = ResultsBank("results/all_results.csv")
        bank.get_all()
        bank.get_by_hypothesis("h001")
        bank.get_by_experiment("ppo_baseline_s1")
        bank.append({...})
    """

    def __init__(self, path: str = "results/all_results.csv"):
        self.path = path

    def _read_all(self) -> list[dict]:
        """Read all rows from the CSV."""
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def get_all(self) -> list[dict]:
        """Get all results."""
        return self._read_all()

    def get_by_hypothesis(self, hypothesis_id: str) -> list[dict]:
        """Get all results for a specific hypothesis."""
        return [r for r in self._read_all() if r.get("hypothesis_id") == hypothesis_id]

    def get_by_experiment(self, experiment_id: str) -> list[dict]:
        """Get results for a specific experiment."""
        return [r for r in self._read_all() if r.get("experiment_id") == experiment_id]

    def get_by_status(self, status: str) -> list[dict]:
        """Get all results with a specific status (open, closed, promising)."""
        return [r for r in self._read_all() if r.get("status") == status]

    def get_open(self) -> list[dict]:
        """Get all hypotheses marked as open (worth revisiting)."""
        return self.get_by_status("open")

    def get_promising(self) -> list[dict]:
        """Get all hypotheses marked as promising (actively developed)."""
        return self.get_by_status("promising")

    def get_closed(self) -> list[dict]:
        """Get all hypotheses marked as closed (dead ends)."""
        return self.get_by_status("closed")

    def get_comments(self, hypothesis_id: str = "", experiment_id: str = "") -> list[dict]:
        """Get comments/notes filtered by hypothesis or experiment."""
        rows = self._read_all()
        if hypothesis_id:
            rows = [r for r in rows if r.get("hypothesis_id") == hypothesis_id]
        if experiment_id:
            rows = [r for r in rows if r.get("experiment_id") == experiment_id]
        return [{"experiment_id": r.get("experiment_id"), "hypothesis_id": r.get("hypothesis_id"),
                 "comment": r.get("comment", ""), "status": r.get("status", "")} for r in rows]

    def get_unique_hypotheses(self) -> list[str]:
        """Get list of all unique hypothesis IDs in the results bank."""
        return sorted(set(r.get("hypothesis_id", "") for r in self._read_all() if r.get("hypothesis_id")))

    def get_fieldnames(self) -> list[str]:
        """Get the current CSV column names."""
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            reader = csv.reader(f)
            try:
                return next(reader)
            except StopIteration:
                return []

    def append(self, row: dict) -> None:
        """Append a single result row to the bank.

        If the CSV doesn't exist yet, creates it with the row's keys as headers.
        If it exists, the row must match the existing columns.
        Missing required columns (experiment_id, hypothesis_id, command, comment, status)
        will be filled with empty strings.
        """
        required = ["experiment_id", "hypothesis_id", "command", "comment", "status"]
        for col in required:
            if col not in row:
                row[col] = ""

        file_exists = os.path.exists(self.path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        if file_exists:
            fieldnames = self.get_fieldnames()
            # Add any new columns from this row
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
            # Rewrite with updated fieldnames if new columns added
            if set(row.keys()) - set(self.get_fieldnames()):
                existing = self._read_all()
                with open(self.path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in existing:
                        writer.writerow(r)
                    writer.writerow(row)
            else:
                with open(self.path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(row)
        else:
            fieldnames = list(row.keys())
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

    def append_many(self, rows: list[dict]) -> None:
        """Append multiple result rows."""
        for row in rows:
            self.append(row)

    def summary(self) -> str:
        """Get a human-readable summary of the results bank."""
        rows = self._read_all()
        if not rows:
            return "Results bank is empty."

        hypotheses = {}
        for r in rows:
            hid = r.get("hypothesis_id", "unknown")
            if hid not in hypotheses:
                hypotheses[hid] = {"total": 0, "open": 0, "closed": 0, "promising": 0}
            hypotheses[hid]["total"] += 1
            status = r.get("status", "")
            if status in hypotheses[hid]:
                hypotheses[hid][status] += 1

        lines = [f"Results bank: {len(rows)} total results across {len(hypotheses)} hypotheses\n"]
        for hid in sorted(hypotheses.keys()):
            s = hypotheses[hid]
            lines.append(f"  {hid}: {s['total']} results ({s['promising']} promising, {s['open']} open, {s['closed']} closed)")

        return "\n".join(lines)
