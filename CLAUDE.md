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

**Module layout:**
- `xgenius/cli.py` — Unified CLI (argparse with 20 subcommands, all support `--json`)
- `xgenius/config.py` — TOML config loading, validation, dataclasses (`XGeniusConfig`, `ClusterConfig`, `SafetyConfig`)
- `xgenius/safety.py` — `SafetyValidator`: resource limits, command allowlist, path containment, shell injection detection, audit logging
- `xgenius/ssh.py` — `SSHClient`: structured SSH/SCP/rsync operations via subprocess, returns `SSHResult`
- `xgenius/jobs.py` — `JobManager`: job submission (captures SLURM job IDs), status checking, cancellation, log retrieval, completion detection
- `xgenius/journal.py` — `ResearchJournal`: JSONL-based hypothesis/experiment/result tracking, markdown context generation
- `xgenius/container.py` — `ContainerManager`: Singularity container build/push/verify
- `xgenius/watcher.py` — Background daemon: polls clusters for `.done` markers, triggers `claude --continue`
- `xgenius/templates.py` — SBATCH template loading, `{{PLACEHOLDER}}` rendering, completion epilog injection

**Safety enforcement**: Three layers — Python-level `SafetyValidator` (command allowlist, resource limits, path containment), Singularity containerization (filesystem isolation), SLURM scheduler limits.

**Per-project state**: `xgenius init` creates `xgenius.toml`, `research_goal.md`, and `.xgenius/` directory (journal, jobs, audit log, markers) in any project.

## Key Patterns

- All CLI commands support `--json` for structured output (critical for LLM consumption)
- Safety validation happens before every remote operation in `jobs.py` — the LLM cannot bypass it
- Job IDs are captured from `sbatch` stdout and tracked in `.xgenius/jobs.jsonl`
- SBATCH scripts get a completion epilog injected that writes `.done` marker files on the cluster
- The watcher daemon polls for these markers and triggers `claude --continue` to resume the research loop
