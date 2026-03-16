"""Results bank utilities for xgenius.

Two CSV tables:
- results/experiments.csv — one row per experiment (metrics, per-experiment notes)
- results/hypotheses.csv — one row per hypothesis (status, conclusions, notes)

The agent defines project-specific metric columns. xgenius only requires
the structural columns (IDs, command, comment, status).
"""

import csv
import os
from typing import Any


class ExperimentsBank:
    """Query and manage the experiments CSV (one row per experiment).

    Required columns: experiment_id, hypothesis_id, command, comment
    Metric columns: project-dependent (agent defines these)
    """

    def __init__(self, path: str = "results/experiments.csv"):
        self.path = path

    def _read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))

    def get_all(self) -> list[dict]:
        return self._read_all()

    def get_by_hypothesis(self, hypothesis_id: str) -> list[dict]:
        return [r for r in self._read_all() if r.get("hypothesis_id") == hypothesis_id]

    def get_by_experiment(self, experiment_id: str) -> list[dict]:
        return [r for r in self._read_all() if r.get("experiment_id") == experiment_id]

    def get_fieldnames(self) -> list[str]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            reader = csv.reader(f)
            try:
                return next(reader)
            except StopIteration:
                return []

    def append(self, row: dict) -> None:
        """Append an experiment result row."""
        required = ["experiment_id", "hypothesis_id", "command", "comment"]
        for col in required:
            if col not in row:
                row[col] = ""

        file_exists = os.path.exists(self.path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        if file_exists:
            fieldnames = self.get_fieldnames()
            new_cols = [k for k in row.keys() if k not in fieldnames]
            if new_cols:
                fieldnames.extend(new_cols)
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
        for row in rows:
            self.append(row)


class HypothesesBank:
    """Query and manage the hypotheses CSV (one row per hypothesis).

    Required columns: hypothesis_id, description, motivation, status, comment
    Status: open (revisit later), closed (dead end), promising (active), proposed (not tested yet)
    """

    def __init__(self, path: str = "results/hypotheses.csv"):
        self.path = path

    def _read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))

    def get_all(self) -> list[dict]:
        return self._read_all()

    def get_by_id(self, hypothesis_id: str) -> dict | None:
        for r in self._read_all():
            if r.get("hypothesis_id") == hypothesis_id:
                return r
        return None

    def get_by_status(self, status: str) -> list[dict]:
        return [r for r in self._read_all() if r.get("status") == status]

    def get_open(self) -> list[dict]:
        return self.get_by_status("open")

    def get_promising(self) -> list[dict]:
        return self.get_by_status("promising")

    def get_closed(self) -> list[dict]:
        return self.get_by_status("closed")

    def get_fieldnames(self) -> list[str]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            reader = csv.reader(f)
            try:
                return next(reader)
            except StopIteration:
                return []

    def upsert(self, row: dict) -> None:
        """Insert or update a hypothesis row (matched by hypothesis_id)."""
        required = ["hypothesis_id", "description", "motivation", "status", "comment"]
        for col in required:
            if col not in row:
                row[col] = ""

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        existing = self._read_all()

        # Determine fieldnames
        if existing:
            fieldnames = self.get_fieldnames()
            new_cols = [k for k in row.keys() if k not in fieldnames]
            fieldnames.extend(new_cols)
        else:
            fieldnames = list(row.keys())

        # Update existing or append
        updated = False
        for i, r in enumerate(existing):
            if r.get("hypothesis_id") == row.get("hypothesis_id"):
                existing[i] = {**r, **row}
                updated = True
                break

        if not updated:
            existing.append(row)

        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in existing:
                writer.writerow(r)


class ResultsBank:
    """Unified interface to both experiments and hypotheses banks.

    Usage:
        from xgenius.results import ResultsBank
        bank = ResultsBank("results/")
        bank.experiments.get_all()
        bank.hypotheses.get_open()
        bank.summary()
    """

    def __init__(self, results_dir: str = "results/"):
        self.results_dir = results_dir
        self.experiments = ExperimentsBank(os.path.join(results_dir, "experiments.csv"))
        self.hypotheses = HypothesesBank(os.path.join(results_dir, "hypotheses.csv"))

    def summary(self) -> str:
        exp_rows = self.experiments.get_all()
        hyp_rows = self.hypotheses.get_all()

        if not exp_rows and not hyp_rows:
            return "Results bank is empty."

        lines = []
        lines.append(f"Results bank: {len(exp_rows)} experiments, {len(hyp_rows)} hypotheses\n")

        if hyp_rows:
            lines.append("Hypotheses:")
            for h in hyp_rows:
                hid = h.get("hypothesis_id", "?")
                status = h.get("status", "?")
                desc = h.get("description", "")[:60]
                n_exp = len(self.experiments.get_by_hypothesis(hid))
                lines.append(f"  {hid} [{status}] ({n_exp} experiments): {desc}")

        return "\n".join(lines)
