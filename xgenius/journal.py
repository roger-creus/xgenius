"""Research journal for xgenius.

Tracks hypotheses, experiments, and results in an append-only JSONL log.
Generates LLM-readable markdown summaries for Claude's research context.
"""

import json
import os
import time
from dataclasses import dataclass

from xgenius.config import XGeniusConfig, get_xgenius_dir, get_project_dir, ensure_xgenius_dir


def _next_id(prefix: str, existing_ids: list[str]) -> str:
    """Generate the next sequential ID like h001, e001, etc."""
    max_num = 0
    for id_ in existing_ids:
        if id_.startswith(prefix):
            try:
                num = int(id_[len(prefix):])
                max_num = max(max_num, num)
            except ValueError:
                pass
    return f"{prefix}{max_num + 1:03d}"


class ResearchJournal:
    """Structured tracking of hypotheses, experiments, and results.

    Storage: .xgenius/journal.jsonl (append-only JSONL)
    Summary: .xgenius/journal_summary.md (auto-generated markdown)
    """

    def __init__(self, config: XGeniusConfig):
        self.config = config
        self._xgenius_dir = get_xgenius_dir(config)
        self._journal_path = os.path.join(self._xgenius_dir, "journal.jsonl")
        self._summary_path = os.path.join(self._xgenius_dir, "journal_summary.md")

    def _read_entries(self) -> list[dict]:
        """Read all journal entries."""
        if not os.path.exists(self._journal_path):
            return []
        entries = []
        with open(self._journal_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _append_entry(self, entry: dict) -> None:
        """Append a single entry to the journal."""
        ensure_xgenius_dir(self.config)
        with open(self._journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def add_hypothesis(
        self, text: str, motivation: str = "", expected_outcome: str = ""
    ) -> str:
        """Record a new hypothesis.

        Returns:
            The generated hypothesis ID (e.g., 'h001').
        """
        entries = self._read_entries()
        existing_ids = [e["id"] for e in entries if e.get("type") == "hypothesis"]
        new_id = _next_id("h", existing_ids)

        entry = {
            "type": "hypothesis",
            "id": new_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "text": text,
            "motivation": motivation,
            "expected_outcome": expected_outcome,
            "status": "proposed",
        }
        self._append_entry(entry)
        self._regenerate_summary()
        return new_id

    def add_experiment(
        self,
        hypothesis_id: str,
        cluster: str,
        job_id: str,
        command: str,
        config: dict | None = None,
    ) -> str:
        """Record an experiment submission.

        Returns:
            The generated experiment ID (e.g., 'e001').
        """
        entries = self._read_entries()
        existing_ids = [e["id"] for e in entries if e.get("type") == "experiment"]
        new_id = _next_id("e", existing_ids)

        entry = {
            "type": "experiment",
            "id": new_id,
            "hypothesis_id": hypothesis_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cluster": cluster,
            "job_id": job_id,
            "command": command,
            "config": config or {},
            "status": "submitted",
        }
        self._append_entry(entry)
        self._regenerate_summary()
        return new_id

    def add_result(
        self, experiment_id: str, metrics: dict, analysis: str = ""
    ) -> None:
        """Record experiment results."""
        entry = {
            "type": "result",
            "experiment_id": experiment_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": metrics,
            "analysis": analysis,
        }
        self._append_entry(entry)
        self._regenerate_summary()

    def update_hypothesis(
        self, hypothesis_id: str, status: str, conclusion: str = ""
    ) -> None:
        """Update hypothesis status.

        Args:
            hypothesis_id: ID of the hypothesis to update.
            status: New status (proposed, testing, confirmed, rejected, partially_confirmed).
            conclusion: Summary of findings.
        """
        entry = {
            "type": "hypothesis_update",
            "hypothesis_id": hypothesis_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "conclusion": conclusion,
        }
        self._append_entry(entry)
        self._regenerate_summary()

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        """Update an experiment's status (e.g., from submitted to completed)."""
        entry = {
            "type": "experiment_update",
            "experiment_id": experiment_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
        }
        self._append_entry(entry)

    def get_context(self) -> str:
        """Generate full research context for Claude.

        Returns everything Claude needs to make the next research decision:
        - Research goal
        - All hypotheses with current status
        - All experiments and results
        - What has been tried and what worked
        """
        entries = self._read_entries()

        # Load research goal
        project_dir = get_project_dir(self.config)
        goal_path = os.path.join(project_dir, self.config.project.research_goal)
        goal_text = ""
        if os.path.exists(goal_path):
            with open(goal_path) as f:
                goal_text = f.read()

        # Build hypothesis state (apply updates)
        hypotheses = {}
        for e in entries:
            if e["type"] == "hypothesis":
                hypotheses[e["id"]] = {
                    "id": e["id"],
                    "text": e["text"],
                    "motivation": e.get("motivation", ""),
                    "expected_outcome": e.get("expected_outcome", ""),
                    "status": e["status"],
                    "conclusion": "",
                    "timestamp": e["timestamp"],
                }
            elif e["type"] == "hypothesis_update":
                hid = e["hypothesis_id"]
                if hid in hypotheses:
                    hypotheses[hid]["status"] = e["status"]
                    hypotheses[hid]["conclusion"] = e.get("conclusion", "")

        # Build experiment state (apply updates)
        experiments = {}
        for e in entries:
            if e["type"] == "experiment":
                experiments[e["id"]] = {
                    "id": e["id"],
                    "hypothesis_id": e["hypothesis_id"],
                    "cluster": e["cluster"],
                    "job_id": e["job_id"],
                    "command": e["command"],
                    "status": e["status"],
                    "timestamp": e["timestamp"],
                }
            elif e["type"] == "experiment_update":
                eid = e["experiment_id"]
                if eid in experiments:
                    experiments[eid]["status"] = e["status"]

        # Build results map
        results = {}
        for e in entries:
            if e["type"] == "result":
                eid = e["experiment_id"]
                if eid not in results:
                    results[eid] = []
                results[eid].append({
                    "metrics": e["metrics"],
                    "analysis": e.get("analysis", ""),
                    "timestamp": e["timestamp"],
                })
                # Mark experiment as completed
                if eid in experiments:
                    experiments[eid]["status"] = "completed"

        # Generate markdown context
        lines = ["# Research Context\n"]

        if goal_text:
            lines.append("## Research Goal\n")
            lines.append(goal_text)
            lines.append("")

        # Hypotheses table
        if hypotheses:
            lines.append("## Hypotheses\n")
            lines.append("| ID | Hypothesis | Status | Conclusion |")
            lines.append("|---|---|---|---|")
            for h in sorted(hypotheses.values(), key=lambda x: x["id"]):
                lines.append(
                    f"| {h['id']} | {h['text'][:80]} | {h['status']} | {h['conclusion'][:60] or '-'} |"
                )
            lines.append("")

        # Experiments and results by hypothesis
        if experiments:
            lines.append("## Experiments & Results\n")
            hyp_groups = {}
            for exp in experiments.values():
                hid = exp["hypothesis_id"]
                if hid not in hyp_groups:
                    hyp_groups[hid] = []
                hyp_groups[hid].append(exp)

            for hid in sorted(hyp_groups.keys()):
                h = hypotheses.get(hid, {})
                lines.append(f"### {hid}: {h.get('text', 'Unknown hypothesis')[:80]}\n")
                for exp in hyp_groups[hid]:
                    status_icon = {
                        "submitted": "PENDING",
                        "running": "RUNNING",
                        "completed": "DONE",
                        "failed": "FAILED",
                        "cancelled": "CANCELLED",
                    }.get(exp["status"], exp["status"])

                    lines.append(
                        f"- **{exp['id']}** ({exp['cluster']}, job {exp['job_id']}): "
                        f"`{exp['command'][:60]}` [{status_icon}]"
                    )

                    # Add results if available
                    if exp["id"] in results:
                        for r in results[exp["id"]]:
                            metrics_str = ", ".join(
                                f"{k}={v}" for k, v in r["metrics"].items()
                            )
                            lines.append(f"  - Results: {metrics_str}")
                            if r["analysis"]:
                                lines.append(f"  - Analysis: {r['analysis']}")
                lines.append("")

        # What has been tried summary
        completed_hypotheses = [
            h for h in hypotheses.values()
            if h["status"] in ("confirmed", "rejected", "partially_confirmed")
        ]
        if completed_hypotheses:
            lines.append("## What Has Been Tried\n")
            for i, h in enumerate(completed_hypotheses, 1):
                lines.append(f"{i}. **{h['text'][:80]}** → {h['status']}: {h['conclusion']}")
            lines.append("")

        # Active work
        active_experiments = [
            e for e in experiments.values()
            if e["status"] in ("submitted", "running")
        ]
        if active_experiments:
            lines.append("## Currently Active\n")
            for exp in active_experiments:
                lines.append(
                    f"- {exp['id']} on {exp['cluster']} (job {exp['job_id']}): {exp['status']}"
                )
            lines.append("")

        return "\n".join(lines)

    def get_summary(self) -> str:
        """Get a concise summary of research progress."""
        entries = self._read_entries()

        total_hypotheses = len([e for e in entries if e["type"] == "hypothesis"])
        total_experiments = len([e for e in entries if e["type"] == "experiment"])
        total_results = len([e for e in entries if e["type"] == "result"])

        # Count hypothesis statuses
        statuses = {}
        for e in entries:
            if e["type"] == "hypothesis":
                statuses[e["id"]] = e["status"]
            elif e["type"] == "hypothesis_update":
                statuses[e["hypothesis_id"]] = e["status"]

        confirmed = sum(1 for s in statuses.values() if s == "confirmed")
        rejected = sum(1 for s in statuses.values() if s == "rejected")
        testing = sum(1 for s in statuses.values() if s in ("testing", "proposed"))

        return (
            f"Research Progress: {total_hypotheses} hypotheses "
            f"({confirmed} confirmed, {rejected} rejected, {testing} active), "
            f"{total_experiments} experiments, {total_results} results collected."
        )

    def get_active_experiments(self) -> list[dict]:
        """Return experiments with status submitted or running."""
        entries = self._read_entries()

        experiments = {}
        for e in entries:
            if e["type"] == "experiment":
                experiments[e["id"]] = e
            elif e["type"] == "experiment_update":
                if e["experiment_id"] in experiments:
                    experiments[e["experiment_id"]]["status"] = e["status"]
            elif e["type"] == "result":
                if e["experiment_id"] in experiments:
                    experiments[e["experiment_id"]]["status"] = "completed"

        return [
            e for e in experiments.values()
            if e["status"] in ("submitted", "running")
        ]

    def _regenerate_summary(self) -> None:
        """Regenerate the markdown summary file."""
        ensure_xgenius_dir(self.config)
        context = self.get_context()
        with open(self._summary_path, "w") as f:
            f.write(context)
