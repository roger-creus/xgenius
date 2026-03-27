# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

xgenius is an LLM-oriented autonomous research platform for SLURM clusters. It provides CLI tools that enable Claude Code to autonomously run experiments, track hypotheses, and iterate on research — with safety guarantees for shared infrastructure.

## Build & Install

```bash
pip install -e .           # editable install for development
pip install .              # standard install
```

Dependencies: `rich`, `paramiko`, `scp`, `tomli_w`. Requires Python 3.11+ (`tomllib` in stdlib).

## Running Tests

```bash
python -m pytest tests/ -v                    # all tests
python -m pytest tests/test_safety.py -v      # safety tests only
```

## Architecture

**Config-driven**: Everything starts from `xgenius.toml` (TOML format). Created by `xgenius init` in any project. Contains cluster definitions, SLURM parameters, safety limits, and watcher settings.

**Two state systems**:
- **SQLite DB** (`.xgenius/xgenius.db`) — automated operational state (job statuses, walltimes, exit codes). Updated by the watcher every cycle.
- **Research Journal** (`.xgenius/journal.md`) — Claude's persistent research memory. Written by Claude, read at the start of every session.

**Module layout:**
- `xgenius/cli.py` — Unified CLI (argparse with 25+ subcommands, all support `--json`)
- `xgenius/config.py` — TOML config loading, validation, dataclasses, run ID management
- `xgenius/db.py` — SQLite DB for operational state (jobs table, hypotheses table, state sync)
- `xgenius/safety.py` — `SafetyValidator`: resource limits, command allowlist, path containment, shell injection detection
- `xgenius/ssh.py` — `SSHClient`: structured SSH/SCP/rsync operations via subprocess, returns `SSHResult`
- `xgenius/jobs.py` — `JobManager`: job submission, status checking, cancellation, SLURM log pulling, completion detection
- `xgenius/journal.py` — Simple append-only markdown research journal
- `xgenius/results.py` — Results bank: two-table CSV system (experiments + hypotheses)
- `xgenius/container.py` — `ContainerManager`: step-by-step Docker→Singularity build with structured output
- `xgenius/watcher.py` — Background daemon: polls for `.done` markers, syncs DB from squeue, pulls results + logs, triggers fresh Claude session
- `xgenius/templates.py` — SBATCH template loading, `{{PLACEHOLDER}}` rendering, trap-based completion epilog
- `xgenius/dashboard.py` — Web-based DB browser for human inspection

**Safety enforcement**: Three layers — Python-level `SafetyValidator` (command allowlist, resource limits, path containment), Singularity containerization (filesystem isolation), SLURM scheduler limits.

**Per-project state**: `xgenius init` creates `xgenius.toml`, `research_goal.md`, and `.xgenius/` directory with:
- `xgenius.db` — SQLite operational database
- `journal.md` — research memory
- `DEBUG.md` — error log for human review
- `templates/` — customizable SBATCH templates
- `slurm_logs/` — locally pulled SLURM .out/.err files
- `batches/` — archived batch submission files
- `run_id` — unique run identifier for scoping jobs

## Key Patterns

- All CLI commands support `--json` for structured output (critical for LLM consumption)
- Safety validation happens before every remote operation in `jobs.py` — the LLM cannot bypass it
- Job IDs are captured from `sbatch` stdout and tracked in the SQLite DB
- SBATCH scripts get a trap-based completion epilog that writes `.done` marker files on the cluster
- The watcher daemon polls for markers, syncs DB from squeue, pulls results + SLURM logs locally, and triggers a fresh Claude session per completion batch
- Each run has a unique ID (xg-XXXXXX) that scopes SLURM job names and prevents old jobs from interfering
- SLURM logs are pulled to `.xgenius/slurm_logs/{hypothesis_id}/{experiment_id}/` for local inspection
- Project-local SBATCH templates in `.xgenius/templates/` take priority over package templates
- `xgenius compact` spawns a Claude agent to intelligently compact the research journal — reducing size while preserving all essential context (findings, hypothesis statuses, decisions, human directives, next steps). The original journal is backed up before replacement. Call this when the journal grows large and starts consuming too much context. Works with `--json` for programmatic use.
