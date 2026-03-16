"""Research journal for xgenius.

The journal is Claude's persistent research memory — a simple markdown file
that records what was tried, what was learned, and what to do next.

NOT for operational tracking (that's the SQLite DB). The journal is for:
- Research narrative and reasoning
- Hypothesis motivations and conclusions
- Key findings and insights
- Ideas for future investigation
- Decisions and their rationale

Each entry is timestamped and appended. Claude reads this at the start
of every session to understand where the research stands.
"""

import os
import time

from xgenius.config import XGeniusConfig, get_xgenius_dir, get_project_dir, ensure_xgenius_dir


class ResearchJournal:
    """Simple append-only research journal (markdown file).

    Usage by Claude:
        journal = ResearchJournal(config)
        journal.write("## Hypothesis h001: Spectral norm on value net\\n\\nMotivation: ...")
        journal.read()  # Returns full journal content
    """

    def __init__(self, config: XGeniusConfig):
        self.config = config
        xgenius_dir = ensure_xgenius_dir(config)
        self.journal_path = os.path.join(xgenius_dir, "journal.md")

    def read(self) -> str:
        """Read the full journal. Claude reads this every session."""
        if not os.path.exists(self.journal_path):
            return ""
        with open(self.journal_path) as f:
            return f.read()

    def write(self, entry: str) -> None:
        """Append a timestamped entry to the journal."""
        ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        with open(self.journal_path, "a") as f:
            f.write(f"\n---\n**[{ts}]**\n\n{entry}\n")

    def add_experiment(self, hypothesis_id: str, cluster: str, job_id: str, command: str) -> None:
        """Auto-record an experiment submission in the journal."""
        self.write(f"Submitted job `{job_id}` on `{cluster}` for `{hypothesis_id}`: `{command}`")

    def clear(self) -> None:
        """Clear the journal (called by xgenius reset)."""
        if os.path.exists(self.journal_path):
            os.remove(self.journal_path)
