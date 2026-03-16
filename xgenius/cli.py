"""Unified CLI for xgenius.

All xgenius commands go through this single entry point.
Every command supports --json for structured output.
"""

import argparse
import json
import os
import sys

from rich.console import Console
from rich.table import Table

console = Console()


def _output(data: dict | list | str, use_json: bool) -> None:
    """Output data as JSON or human-readable format."""
    if use_json:
        if isinstance(data, str):
            print(json.dumps({"output": data}))
        else:
            print(json.dumps(data, indent=2))
    else:
        if isinstance(data, str):
            print(data)
        else:
            print(json.dumps(data, indent=2))


def _load_config(args):
    """Load config, handling errors."""
    from xgenius.config import load_config
    try:
        return load_config(args.config)
    except FileNotFoundError as e:
        _output({"error": str(e)}, getattr(args, "json", False))
        sys.exit(1)


# --- Init ---

def cmd_init(args):
    """Interactive project setup."""
    import tomli_w
    from xgenius.config import ensure_xgenius_dir, XGeniusConfig, ProjectConfig

    project_dir = os.getcwd()
    config_path = os.path.join(project_dir, "xgenius.toml")

    if os.path.exists(config_path) and not args.force:
        console.print("[yellow]xgenius.toml already exists. Use --force to overwrite.[/yellow]")
        return

    console.print("[bold]xgenius init[/bold] — Setting up autonomous research project\n")

    # Detect Dockerfile
    dockerfile = "Dockerfile"
    if not os.path.exists(os.path.join(project_dir, dockerfile)):
        console.print("[yellow]No Dockerfile found in current directory.[/yellow]")
        dockerfile = input("Path to Dockerfile (or press Enter to skip): ").strip() or ""

    # Project name
    default_name = os.path.basename(project_dir)
    project_name = input(f"Project name [{default_name}]: ").strip() or default_name

    # Container image name
    default_image = f"{project_name}.sif"
    container_image = input(f"Container image name [{default_image}]: ").strip() or default_image

    # Cluster setup
    clusters = {}
    console.print("\n[bold]Cluster Configuration[/bold]")
    console.print("Add clusters one at a time. Type 'done' when finished.\n")

    while True:
        cluster_name = input("Cluster name (or 'done'): ").strip()
        if cluster_name.lower() == "done":
            break
        if not cluster_name:
            continue

        hostname = input(f"  Hostname (SSH config name) [{cluster_name}]: ").strip() or cluster_name
        username = input("  Username: ").strip()
        project_path = input("  Project path on cluster (absolute): ").strip()
        scratch_path = input("  Scratch path on cluster (absolute): ").strip()
        image_path = input(f"  Image storage path [{scratch_path}/images]: ").strip() or f"{scratch_path}/images"

        # SLURM config
        console.print(f"\n  [bold]SLURM settings for {cluster_name}[/bold]")
        account = input("  Account (or press Enter for none): ").strip()
        partition = input("  Partition (or press Enter for none): ").strip()
        num_gpus = input("  GPUs per job [1]: ").strip() or "1"
        num_cpus = input("  CPUs per job [8]: ").strip() or "8"
        memory = input("  Memory per job [32G]: ").strip() or "32G"
        walltime = input("  Max walltime [12:00:00]: ").strip() or "12:00:00"
        modules = input("  Modules to load [singularity]: ").strip() or "singularity"
        sing_cmd = input("  Singularity command [singularity]: ").strip() or "singularity"
        output_dir = input(f"  Output dir on cluster [{scratch_path}/runs]: ").strip() or f"{scratch_path}/runs"

        sbatch_template = "slurm_account_template.sbatch" if account else "slurm_partition_template.sbatch"

        clusters[cluster_name] = {
            "hostname": hostname,
            "username": username,
            "project_path": project_path,
            "scratch_path": scratch_path,
            "image_path": image_path,
            "sbatch_template": sbatch_template,
            "slurm": {
                "account": account,
                "partition": partition,
                "num_gpus": int(num_gpus),
                "num_cpus": int(num_cpus),
                "memory": memory,
                "walltime": walltime,
                "modules": modules,
                "singularity_command": sing_cmd,
                "output_dir_cluster": output_dir,
                "output_dir_container": "/results",
            },
        }
        console.print(f"  [green]Added cluster: {cluster_name}[/green]\n")

    # Build TOML config
    config_data = {
        "project": {
            "name": project_name,
            "research_goal": "research_goal.md",
            "container_image": container_image,
            "dockerfile": dockerfile,
        },
        "safety": {
            "max_gpus_per_job": 1,
            "max_cpus_per_job": 16,
            "max_memory_per_job": "64G",
            "max_walltime": "24:00:00",
            "max_concurrent_jobs": 10,
            "max_total_gpu_hours": 500,
            "allowed_command_prefixes": ["python"],
            "forbidden_patterns": [
                "rm -rf", "sudo", "chmod", "chown", "wget", "curl",
                "mkfs", "dd ", "shutdown", "reboot", "kill -9",
            ],
            "require_singularity": True,
        },
        "watcher": {
            "poll_interval_seconds": 60,
            "trigger_command": "claude --dangerously-skip-permissions",
        },
        "clusters": clusters,
    }

    # Write xgenius.toml
    with open(config_path, "wb") as f:
        tomli_w.dump(config_data, f)
    console.print(f"\n[green]Created xgenius.toml[/green]")

    # Create research_goal.md template
    goal_path = os.path.join(project_dir, "research_goal.md")
    if not os.path.exists(goal_path):
        with open(goal_path, "w") as f:
            f.write(f"""# Research Goal

## Objective
Describe what you want the autonomous research agent to achieve.

## Baseline
Describe the current baseline methods/results to improve upon.

## What Counts as Success
Define measurable success criteria.

## Constraints
List any constraints on the research (compute budget, methods, etc.)

## Ideas to Explore
Optionally list initial ideas for the agent to consider.
""")
        console.print("[green]Created research_goal.md[/green] — Edit this to define your research objective.")

    # Create .xgenius directory
    from xgenius.config import load_config as _lc
    cfg = _lc(config_path)
    ensure_xgenius_dir(cfg)
    console.print("[green]Created .xgenius/ directory[/green]")

    # Create results directory
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    console.print("[green]Created results/ directory[/green] — results bank lives here.")

    # Copy SBATCH templates to project so Claude can customize them
    from xgenius.templates import copy_templates_to_project
    templates_dir = copy_templates_to_project(project_dir)
    console.print(f"[green]Copied SBATCH templates to {templates_dir}[/green] — Claude can customize these.")

    # Add .xgenius to .gitignore
    gitignore_path = os.path.join(project_dir, ".gitignore")
    # Gitignore: keep transient files out, but commit journal/jobs for research history
    xgenius_gitignore = """
# xgenius — transient state (don't commit)
.xgenius/markers/
.xgenius/watcher.log
.xgenius/watcher.lock

# Singularity/Apptainer containers (too large for git)
*.sif

# Common large/transient files
wandb/
runs/
*.tfevents.*
*.pt
*.pth
*.ckpt
"""
    if os.path.exists(gitignore_path):
        with open(gitignore_path) as f:
            content = f.read()
        if ".xgenius/markers/" not in content:
            with open(gitignore_path, "a") as f:
                f.write(xgenius_gitignore)
            console.print("[green]Updated .gitignore for xgenius[/green]")
    else:
        with open(gitignore_path, "w") as f:
            f.write(xgenius_gitignore)
        console.print("[green]Created .gitignore for xgenius[/green]")

    # Create/append CLAUDE.md with tool documentation
    _write_claude_md(project_dir)

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("Next steps:")
    console.print("  1. Edit research_goal.md to define your research objective")
    console.print("  2. Edit xgenius.toml to adjust safety limits and cluster settings")
    console.print("  3. Build your container: xgenius build")
    console.print("  4. Push to cluster: xgenius push-image --cluster <name>")
    console.print("  5. Start the watcher: xgenius watch")
    console.print("  6. Let Claude begin research!")


def _write_claude_md(project_dir: str):
    """Create or append CLAUDE.md with xgenius tool documentation."""
    claude_md_path = os.path.join(project_dir, "CLAUDE.md")

    xgenius_section = """
## xgenius — Autonomous Research Tools

This project uses xgenius for autonomous research on SLURM clusters.
Configuration is in `xgenius.toml`. Research goal is in `research_goal.md`.
Runtime state is in `.xgenius/` (journal, jobs, audit log).

### Git Conventions

You MUST commit and push your work regularly. Every meaningful change should be committed. Use these prefixes:

```
baseline: <description>        — Running/recording baseline experiments
hypothesis(ID): <description>  — Implementing a hypothesis (e.g., hypothesis(h003): add spectral norm to value net)
result(ID): <description>      — Recording results for a hypothesis
engineering: <description>     — Non-research improvements (better architecture, optimizer swap, etc.)
fix: <description>             — Bug fixes (broken training, container issues, etc.)
revert(ID): <description>     — Reverting a hypothesis that didn't work
ablation(ID): <description>    — Ablation study for a hypothesis
config: <description>          — Configuration changes (xgenius.toml, SBATCH templates, etc.)
container: <description>       — Dockerfile or container changes
docs: <description>            — Documentation updates
```

**Rules:**
- Commit BEFORE submitting experiments (so the cluster runs the committed code)
- Commit AFTER recording results (so the journal state is preserved)
- Push after every commit — this repo is your research record
- Never force push or rewrite history
- If a hypothesis breaks things, use `git revert` to undo it cleanly
- Keep .xgenius/ state files committed (journal, jobs) so the research history is preserved in git

### Available Commands

**Research Loop:**
- `xgenius journal read` — Read the full research journal (your persistent memory across sessions)
- `xgenius journal write "entry text"` — Append a timestamped entry to the journal

**Job Management:**
- `xgenius submit --cluster NAME --command "python script.py --args" [--experiment-id ID] [--hypothesis-id ID] [--gpus N] [--cpus N] [--memory "16G"] [--walltime "04:00:00"]` — Submit a job
- `xgenius batch-submit --file experiments.json` — Submit multiple jobs
- `xgenius status [--cluster NAME] [--json]` — Check job statuses
- `xgenius cancel --cluster NAME --job-ids ID1,ID2` — Cancel specific jobs
- `xgenius logs --experiment-id ID --json` — View job stdout (can also use --cluster NAME --job-id ID)
- `xgenius errors --experiment-id ID --json` — View job stderr/crashes/tracebacks (can also use --cluster NAME --job-id ID)
- `xgenius check-completions [--json]` — Check for newly completed jobs
- `xgenius reconcile --json` — Sync local job tracker with actual SLURM state (fixes stale jobs)

**Code & Data:**
- `xgenius sync --cluster NAME` — Rsync project code to cluster
- `xgenius pull --cluster NAME [--job-id ID]` — Pull results from cluster
- `xgenius ls --cluster NAME [--path PATH]` — List files on cluster

**Container (you must handle building intelligently):**
- `xgenius build --json` — Full pipeline: docker build → test → singularity convert. Returns structured step-by-step results.
- `xgenius build --step docker --json` — Docker build only. If it fails, read the error, fix the Dockerfile, and retry.
- `xgenius build --step test --json` — Run tests inside the Docker container. Use `--test-command "..."` for custom commands.
- `xgenius build --step singularity --json` — Convert Docker image to Singularity .sif.
- `xgenius build --skip-tests --json` — Full pipeline but skip tests.
- `xgenius push-image --cluster NAME [--image PATH] --json` — Push .sif to cluster, verify it runs.
- `xgenius verify-image --cluster NAME --json` — Verify container works on cluster.

**CRITICAL — Container design:**
The Docker/Singularity image must contain ONLY dependencies (CUDA, Python, pip packages), NOT the project code.
Code is mounted at runtime via Singularity --bind flags in the SBATCH template. This means:
- The Dockerfile should install all pip/system dependencies but should NOT COPY source code into the image.
- Removing `COPY ./src /src` or similar lines from the Dockerfile is correct — code goes via `xgenius sync`.
- Keep `COPY pyproject.toml` + `pip install` to install dependencies, but remove any final COPY of source code.
- Remove any ENTRYPOINT that expects code to be baked in — the SBATCH template handles command execution.
- After building, verify the image has all deps by running a test import, not by running the full code.

**IMPORTANT for container building:** When `xgenius build` fails, YOU are responsible for diagnosing the error.
Read the Dockerfile, understand the project structure, fix the issue, and re-run `xgenius build`.
You can run individual steps (`--step docker`, `--step test`, `--step singularity`) to iterate on specific failures.
Typical issues: outdated base images, missing dependencies, wrong Python version, broken COPY paths.

**Safety & Budget:**
- `xgenius budget` — Check remaining compute budget
- `xgenius validate --command "python script.py"` — Dry-run safety check
- `xgenius audit [--limit N]` — View audit log
- `xgenius job-history [--limit N] --json` — View past jobs with walltime, resources, log file paths, and status
- `xgenius reset` — Clear all state for a fresh research run

All commands support `--json` for structured output.

### Debugging Failed Jobs
When a job fails:
1. Run `xgenius errors --experiment-id EXPERIMENT_ID --json` to see tracebacks and error messages
2. Run `xgenius logs --experiment-id EXPERIMENT_ID --json` to see full stdout
3. Run `xgenius job-history --json` to see all jobs with their log file paths, statuses, and walltimes
4. Log files are stored at `{scratch}/.xgenius/logs/{experiment_id}_{job_id}.out` on the cluster

### Debug Log
When you encounter errors (cluster issues, submission failures, crashes, unexpected behavior), append a timestamped entry to `DEBUG.md` in the project root. Format:
```
## YYYY-MM-DD HH:MM — Brief title
Description of what went wrong, what you tried, and the outcome.
```
This file is for the HUMAN to review — it helps them see what infrastructure issues you're hitting. Commit it with your other changes.

### Results Bank
Two CSV tables in `results/`, both committed to git:

**`results/experiments.csv`** — one row per experiment
- Required: `experiment_id`, `hypothesis_id`, `command`, `comment` (per-experiment notes)
- Metric columns: project-dependent — you define what to track (the target metric from research_goal.md MUST be included)

**`results/hypotheses.csv`** — one row per hypothesis (upserted as status changes)
- Required: `hypothesis_id`, `description`, `motivation`, `status`, `comment` (high-level notes)
- Status: `proposed`, `open` (revisit later), `promising` (active), `closed` (dead end)

**CLI:** `xgenius results summary|experiments|hypotheses|hypothesis --id X|experiment --id X|open|promising|closed`

**Python API:**
```python
from xgenius.results import ResultsBank
bank = ResultsBank("results/")
bank.experiments.get_by_hypothesis("h001")
bank.experiments.append({...})
bank.hypotheses.upsert({"hypothesis_id": "h001", "status": "closed", "comment": "..."})
bank.summary()
```

**CRITICAL — Experiment CSV output convention:**
Every training script MUST save a CSV results file during/after execution. The file MUST:
- Be saved to the output directory (bound via Singularity `--bind` to the cluster output path)
- Follow this naming: `{hypothesis_id}__{experiment_id}.csv` (double underscore separator)
- Contain at minimum: the target metric(s) of interest as specified in research_goal.md
- Be parseable by you when you wake up — include column headers

Example: a job submitted as `--hypothesis-id h003 --experiment-id pqn_spectral_qbert_s1` should produce:
`/output_dir/h003__pqn_spectral_qbert_s1.csv`

If a training script does NOT save a CSV with this convention, you MUST add it before submitting experiments.
The watcher pulls these raw CSVs to `results/CLUSTER/` when jobs complete. You then parse them and append to the results bank.

**Your responsibilities:**
1. Ensure every training script saves a `{hypothesis_id}__{experiment_id}.csv` with the target metric — implement this if missing
2. When woken up, parse pulled CSVs from `results/CLUSTER/` and append rows to `results/experiments.csv`
3. Update `results/hypotheses.csv` with status and notes as hypotheses evolve
4. Build project-specific analysis tools to compare algorithms and track progress — commit these
5. Commit both CSVs to git after every update
6. Do NOT conclude on a hypothesis until ALL its experiments have completed — pilots verify correctness, NOT hypothesis validity
7. Leave detailed notes on BOTH tables: per-experiment observations + per-hypothesis conclusions

### Research and Knowledge
- **Search the web constantly** for related work, recent papers, implementation tricks, and state-of-the-art methods — not just at the start, but throughout the research
- **Install new dependencies** freely — add to Dockerfile, rebuild (`xgenius build`), push (`xgenius push-image`)
- **Read existing code** in the repository for inspiration
- Stay up to date — use web search to find papers, blog posts, and code repositories

### Resource Management
The xgenius.toml [safety] section defines MAXIMUM resource limits. You can request LESS:
- `--gpus 1` instead of the max 4
- `--walltime "02:00:00"` for a quick test instead of the max 24h
- `--memory "16G"` if the job doesn't need much RAM
- `--cpus 4` for a lightweight job

Use `xgenius job-history --json` to see how long past jobs took, then adjust walltime accordingly.
Use `xgenius status --json` to see pending/running jobs with their elapsed time, submit time, and pending reason.

### Two State Systems — DB + Journal

**SQLite DB** (`.xgenius/xgenius.db`) — AUTOMATED operational state. Updated by the watcher every cycle.
- Job statuses (submitted/pending/running/completed/failed/etc.)
- Walltimes, exit codes, timestamps, results_pulled flag
- Query with: `xgenius job-history --json`, `xgenius status --json`

**Research Journal** (`.xgenius/journal.md`) — YOUR persistent research memory. Written by you.
- What hypotheses were tried and why
- Key findings, insights, and conclusions
- Ideas for future investigation
- Decisions and rationale
- Read with: `xgenius journal read`
- Write with: `xgenius journal write "your entry here"`

**Every session, you MUST:**
1. Read the journal (`xgenius journal read`) to recall what previous sessions did
2. Check the DB (`xgenius job-history --json`) for current job states
3. Before exiting, write a journal entry summarizing what you did and what to do next

### Research Workflow
1. Read `xgenius journal read` for research memory from previous sessions
2. Check `xgenius job-history --json` for DB state (all job statuses, walltimes, etc.)
3. Formulate a hypothesis and record it
3. Modify code to test the hypothesis
4. Run `xgenius sync` to push code to cluster
5. Run `xgenius submit` to start experiments
6. Exit and wait — the `xgenius watch` daemon (managed by the human) will trigger you on job completion
7. Analyze results, record findings, iterate

## Project State Directory

The `.xgenius/` directory contains all xgenius runtime state:
- `.xgenius/templates/` — SBATCH job script templates (you can edit these to customize job behavior)
- `.xgenius/journal.md` — your persistent research memory (read/write every session)
- `.xgenius/xgenius.db` — SQLite DB with all job states (automated by watcher)
- `.xgenius/batches/` — archived batch submission files (auto-saved on every batch-submit)
- `.xgenius/watcher.log` — watcher daemon activity log

**SBATCH templates:** If you need to modify SBATCH job scripts (e.g., add `mkdir -p runs` before the command, change bind mounts), edit the templates in `.xgenius/templates/`. Do NOT modify the xgenius package. The project-local templates take priority.

### Container Build Workflow
When you need to build/rebuild the container:
1. Read the Dockerfile and understand what it does
2. Ensure the Dockerfile does NOT bake source code into the image (remove COPY of source dirs, remove ENTRYPOINT)
3. Ensure the Dockerfile installs all system + pip dependencies
4. Run `xgenius build --json` for the full pipeline
5. If docker build fails: read the error, fix the Dockerfile, retry with `xgenius build --step docker --json`
6. Verify deps work: `xgenius build --step test --test-command "python -c 'import torch; print(torch.cuda.is_available())'" --json`
7. If singularity conversion fails: check if apptainer/singularity is installed
8. Push to cluster: `xgenius push-image --cluster NAME --json`
9. You only need to rebuild the container when dependencies change. For code changes, use `xgenius sync`.

### Safety
- All commands are validated against limits in xgenius.toml [safety]
- Commands must start with allowed prefixes (e.g., "python")
- Resource requests are checked against max GPU/CPU/memory/walltime
- All job states tracked in .xgenius/xgenius.db (automated)
"""

    if os.path.exists(claude_md_path):
        with open(claude_md_path) as f:
            content = f.read()
        if "## xgenius — Autonomous Research Tools" not in content:
            with open(claude_md_path, "a") as f:
                f.write(xgenius_section)
            console.print("[green]Appended xgenius docs to CLAUDE.md[/green]")
    else:
        with open(claude_md_path, "w") as f:
            f.write("# CLAUDE.md\n\nThis file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.\n")
            f.write(xgenius_section)
        console.print("[green]Created CLAUDE.md with xgenius docs[/green]")


# --- Submit ---

def cmd_submit(args):
    """Submit a job to a SLURM cluster."""
    config = _load_config(args)
    from xgenius.jobs import JobManager
    from xgenius.journal import ResearchJournal

    manager = JobManager(config)
    result = manager.submit(
        cluster_name=args.cluster,
        command=args.command,
        experiment_id=args.experiment_id or "",
        hypothesis_id=args.hypothesis_id or "",
        num_gpus=args.gpus,
        gpu_type=args.gpu_type,
        num_cpus=args.cpus,
        memory=args.memory,
        walltime=args.walltime,
    )

    # Auto-record in journal if hypothesis_id provided
    if result.success and args.hypothesis_id:
        journal = ResearchJournal(config)
        journal.add_experiment(
            hypothesis_id=args.hypothesis_id,
            cluster=args.cluster,
            job_id=result.job_id,
            command=args.command,
        )

    _output(result.to_dict(), args.json)
    if not result.success:
        sys.exit(1)


# --- Batch Submit ---

def cmd_batch_submit(args):
    """Submit multiple jobs from a config file."""
    config = _load_config(args)
    from xgenius.jobs import JobManager
    from xgenius.journal import ResearchJournal
    from xgenius.config import get_xgenius_dir, ensure_xgenius_dir
    import shutil
    import time as _time

    manager = JobManager(config)
    journal = ResearchJournal(config)

    with open(args.file) as f:
        experiments = json.load(f)

    if isinstance(experiments, dict):
        experiments = experiments.get("experiments", [])

    # Archive batch file to .xgenius/batches/ for future reference
    xgenius_dir = ensure_xgenius_dir(config)
    batches_dir = os.path.join(xgenius_dir, "batches")
    os.makedirs(batches_dir, exist_ok=True)
    timestamp = _time.strftime("%Y%m%d_%H%M%S")
    batch_name = os.path.basename(args.file).replace(".json", "")
    archive_path = os.path.join(batches_dir, f"{timestamp}_{batch_name}.json")
    shutil.copy2(args.file, archive_path)

    results = []
    for exp in experiments:
        result = manager.submit(
            cluster_name=exp["cluster"],
            command=exp["command"],
            experiment_id=exp.get("experiment_id", ""),
            hypothesis_id=exp.get("hypothesis_id", ""),
            num_gpus=exp.get("gpus"),
            gpu_type=exp.get("gpu_type"),
            num_cpus=exp.get("cpus"),
            memory=exp.get("memory"),
            walltime=exp.get("walltime"),
        )
        if result.success and exp.get("hypothesis_id"):
            journal.add_experiment(
                hypothesis_id=exp["hypothesis_id"],
                cluster=exp["cluster"],
                job_id=result.job_id,
                command=exp["command"],
            )
        results.append(result.to_dict())

    _output(results, args.json)


# --- Status ---

def cmd_status(args):
    """Check job statuses."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)
    statuses = manager.status(cluster_name=args.cluster)

    if args.json:
        _output([s.to_dict() for s in statuses], True)
    else:
        if not statuses:
            console.print("No running jobs found.")
            return

        table = Table(title="Job Status")
        table.add_column("Job ID")
        table.add_column("Name")
        table.add_column("State")
        table.add_column("Elapsed")
        table.add_column("Time Limit")
        table.add_column("Cluster")

        for s in statuses:
            color = {"RUNNING": "green", "PENDING": "yellow", "FAILED": "red"}.get(s.state, "")
            table.add_row(s.job_id, s.name, f"[{color}]{s.state}[/{color}]" if color else s.state,
                         s.elapsed, s.time_limit, s.cluster)

        console.print(table)


# --- Cancel ---

def cmd_cancel(args):
    """Cancel specific jobs."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)
    job_ids = [j.strip() for j in args.job_ids.split(",")]
    result = manager.cancel(args.cluster, job_ids)
    _output(result, args.json)


# --- Logs ---

def cmd_logs(args):
    """Fetch job stdout logs."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)

    cluster = args.cluster
    job_id = args.job_id

    # Allow lookup by experiment ID
    if args.experiment_id:
        log_path, found_job_id, found_cluster = manager._find_log_path_by_experiment(args.experiment_id)
        if found_job_id:
            job_id = job_id or found_job_id
            cluster = cluster or found_cluster

    if not cluster or not job_id:
        _output({"error": "Must provide --cluster + --job-id, or --experiment-id"}, args.json)
        return

    output = manager.logs(cluster, job_id, lines=args.lines)
    _output(output, args.json)


# --- Errors ---

def cmd_errors(args):
    """Fetch job error logs."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)

    cluster = args.cluster
    job_id = args.job_id

    # Allow lookup by experiment ID
    if args.experiment_id:
        log_path, found_job_id, found_cluster = manager._find_log_path_by_experiment(args.experiment_id)
        if found_job_id:
            job_id = job_id or found_job_id
            cluster = cluster or found_cluster

    if not cluster or not job_id:
        _output({"error": "Must provide --cluster + --job-id, or --experiment-id"}, args.json)
        return

    output = manager.errors(cluster, job_id, lines=args.lines)
    _output(output, args.json)


# --- Check Completions ---

def cmd_check_completions(args):
    """Check for newly completed jobs."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)
    completions = manager.check_completions(cluster_name=args.cluster)
    _output([c.to_dict() for c in completions], args.json)


# --- Sync ---

def cmd_sync(args):
    """Sync code to cluster."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)
    clusters = [args.cluster] if args.cluster else list(config.clusters.keys())

    results = []
    for cluster_name in clusters:
        result = manager.sync_code(cluster_name)
        results.append(result)
        if not args.json:
            status = "[green]OK[/green]" if result["success"] else f"[red]FAILED: {result['error']}[/red]"
            console.print(f"Sync to {cluster_name}: {status}")

    if args.json:
        _output(results, True)


# --- Pull ---

def cmd_pull(args):
    """Pull results from cluster."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)
    result = manager.pull_results(
        cluster_name=args.cluster,
        job_id=args.job_id or "",
        local_dir=args.output or "",
    )
    _output(result, args.json)


# --- Ls ---

def cmd_ls(args):
    """List files on cluster."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)
    output = manager.list_remote_files(
        cluster_name=args.cluster,
        path=args.path or "",
        pattern=args.pattern or "",
    )
    _output(output, args.json)


# --- Build ---

def cmd_build(args):
    """Build Singularity container (full pipeline or individual steps)."""
    config = _load_config(args)
    from xgenius.container import ContainerManager

    manager = ContainerManager(config)

    if args.step == "docker":
        result = manager.docker_build(
            dockerfile=args.dockerfile or "",
            image_name=args.image_name or "",
            tag=args.tag or "latest",
            registry=args.registry or "",
        )
    elif args.step == "test":
        result = manager.docker_test(
            image_name=args.image_name or "",
            tag=args.tag or "latest",
            registry=args.registry or "",
            test_command=args.test_command or "",
        )
    elif args.step == "push-docker":
        result = manager.docker_push(
            image_name=args.image_name or "",
            tag=args.tag or "latest",
            registry=args.registry or "",
        )
    elif args.step == "singularity":
        result = manager.singularity_build(
            image_name=args.image_name or "",
            tag=args.tag or "latest",
            registry=args.registry or "",
        )
    else:
        # Default: full pipeline
        result = manager.build_all(
            dockerfile=args.dockerfile or "",
            image_name=args.image_name or "",
            tag=args.tag or "latest",
            registry=args.registry or "",
            skip_tests=args.skip_tests,
        )

    _output(result, args.json)
    if not result.get("success"):
        sys.exit(1)


# --- Push Image ---

def cmd_push_image(args):
    """Push container to cluster."""
    config = _load_config(args)
    from xgenius.container import ContainerManager

    manager = ContainerManager(config)
    clusters = [args.cluster] if args.cluster else list(config.clusters.keys())

    results = []
    for cluster_name in clusters:
        result = manager.push_to_cluster(
            cluster_name=cluster_name,
            image_path=args.image or "",
        )
        results.append(result)
        if not args.json:
            status = "[green]OK[/green]" if result["success"] else f"[red]FAILED: {result.get('error', '')}[/red]"
            console.print(f"Push to {cluster_name}: {status}")

    if args.json:
        _output(results, True)


# --- Verify Image ---

def cmd_verify_image(args):
    """Verify container on cluster."""
    config = _load_config(args)
    from xgenius.container import ContainerManager

    manager = ContainerManager(config)
    result = manager.verify_on_cluster(args.cluster)
    _output(result, args.json)


# --- Journal ---

def cmd_journal(args):
    """Journal operations."""
    config = _load_config(args)
    from xgenius.journal import ResearchJournal

    journal = ResearchJournal(config)

    if args.journal_command == "read":
        content = journal.read()
        _output(content if content else "Journal is empty.", args.json)
    elif args.journal_command == "write":
        journal.write(args.entry)
        _output({"status": "recorded"}, args.json)
    else:
        _output({"error": f"Unknown journal command: {args.journal_command}"}, args.json)


# --- Results Bank ---

def cmd_results(args):
    """Query the results bank."""
    from xgenius.results import ResultsBank
    from xgenius.config import get_project_dir

    config = _load_config(args)
    project_dir = get_project_dir(config)
    bank = ResultsBank(os.path.join(project_dir, "results"))

    if args.results_command == "summary":
        _output(bank.summary(), args.json)
    elif args.results_command == "experiments":
        _output(bank.experiments.get_all(), args.json)
    elif args.results_command == "hypotheses":
        _output(bank.hypotheses.get_all(), args.json)
    elif args.results_command == "hypothesis":
        hyp = bank.hypotheses.get_by_id(args.id)
        exps = bank.experiments.get_by_hypothesis(args.id)
        _output({"hypothesis": hyp, "experiments": exps}, args.json)
    elif args.results_command == "experiment":
        _output(bank.experiments.get_by_experiment(args.id), args.json)
    elif args.results_command == "open":
        _output(bank.hypotheses.get_open(), args.json)
    elif args.results_command == "promising":
        _output(bank.hypotheses.get_promising(), args.json)
    elif args.results_command == "closed":
        _output(bank.hypotheses.get_closed(), args.json)


# --- Report ---

def cmd_report(args):
    """Generate a full research report from journal history."""
    config = _load_config(args)
    from xgenius.journal import ResearchJournal
    from xgenius.config import get_project_dir

    journal = ResearchJournal(config)
    report = journal.generate_report()

    # Write to file
    project_dir = get_project_dir(config)
    output_path = args.output or os.path.join(project_dir, "research_report.md")
    with open(output_path, "w") as f:
        f.write(report)

    if args.json:
        _output({"output": output_path, "length": len(report)}, True)
    else:
        console.print(f"[green]Report written to {output_path}[/green]")
        console.print(f"Length: {len(report)} characters, {len(report.splitlines())} lines")


# --- Budget ---

def cmd_budget(args):
    """Show compute budget."""
    config = _load_config(args)
    from xgenius.safety import SafetyValidator

    validator = SafetyValidator(config)
    budget = validator.get_budget()
    _output(budget.to_dict(), args.json)


# --- Validate ---

def cmd_validate(args):
    """Validate a command against safety rules."""
    config = _load_config(args)
    from xgenius.safety import SafetyValidator

    validator = SafetyValidator(config)
    result = validator.validate_command(args.command)
    _output(result.to_dict(), args.json)


# --- Audit ---

def cmd_audit(args):
    """Show audit log."""
    config = _load_config(args)
    from xgenius.safety import SafetyValidator

    validator = SafetyValidator(config)
    entries = validator.get_audit_log(limit=args.limit)
    _output(entries, args.json)


# --- Reset ---

def cmd_reset(args):
    """Reset xgenius state for a fresh research run."""
    config = _load_config(args)
    from xgenius.config import get_xgenius_dir

    xgenius_dir = get_xgenius_dir(config)

    if not os.path.isdir(xgenius_dir):
        console.print("Nothing to reset — .xgenius/ directory not found.")
        return

    # Reset SQLite DB
    from xgenius.db import XGeniusDB
    try:
        db = XGeniusDB(config)
        db.reset()
    except Exception:
        pass

    files_to_clear = ["journal.md", "watcher.log"]
    cleared = []
    for fname in files_to_clear:
        fpath = os.path.join(xgenius_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "w") as f:
                pass
            cleared.append(fname)

    # Clear markers
    markers_dir = os.path.join(xgenius_dir, "markers")
    markers_cleared = 0
    if os.path.isdir(markers_dir):
        for mfile in os.listdir(markers_dir):
            os.remove(os.path.join(markers_dir, mfile))
            markers_cleared += 1

    if args.json:
        _output({"cleared": cleared, "markers_cleared": markers_cleared}, True)
    else:
        console.print(f"[green]Reset complete.[/green] Cleared: {', '.join(cleared)}")
        if markers_cleared:
            console.print(f"Removed {markers_cleared} completion marker(s).")
        console.print("Ready for a fresh research run.")


# --- Reconcile ---

def cmd_reconcile(args):
    """Reconcile local job tracker with actual SLURM state."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)
    result = manager.reconcile()
    _output(result, args.json)
    if not args.json:
        if result["reconciled"] > 0:
            console.print(f"[yellow]Reconciled {result['reconciled']} stale job(s)[/yellow]")
        if result["completed_detected"] > 0:
            console.print(f"[green]Detected {result['completed_detected']} completion(s)[/green]")
        console.print(f"Active jobs: {result['still_active']}")


# --- Job History ---

def cmd_job_history(args):
    """Show history of tracked jobs with walltime and resources."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)
    history = manager.job_history(limit=args.limit)
    _output(history, args.json)


# --- Watch ---

def cmd_watch(args):
    """Start the background watcher daemon."""
    from xgenius.watcher import run_watcher
    run_watcher(config_path=args.config, verbose=not args.quiet)


# --- Main ---

def main():
    # Parent parser with shared flags that all subcommands inherit
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--config", default="xgenius.toml", help="Path to xgenius.toml")
    parent_parser.add_argument("--json", action="store_true", help="Output structured JSON")

    parser = argparse.ArgumentParser(
        prog="xgenius",
        description="LLM-oriented autonomous research platform for SLURM clusters",
        parents=[parent_parser],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p = subparsers.add_parser("init", parents=[parent_parser], help="Initialize xgenius in current project")
    p.add_argument("--force", action="store_true", help="Overwrite existing config")
    p.set_defaults(func=cmd_init)

    # submit
    p = subparsers.add_parser("submit", parents=[parent_parser], help="Submit a job to a SLURM cluster")
    p.add_argument("--cluster", required=True, help="Cluster name")
    p.add_argument("--command", required=True, help="Command to run in container")
    p.add_argument("--experiment-id", default="", help="Experiment identifier")
    p.add_argument("--hypothesis-id", default="", help="Associated hypothesis ID")
    p.add_argument("--gpus", type=int, default=None, help="GPUs (override, must be <= safety max)")
    p.add_argument("--gpu-type", default=None, help="GPU type e.g. 'h100', 'a100', '3g.20gb' (must be in allowed_gpu_types)")
    p.add_argument("--cpus", type=int, default=None, help="CPUs (override, must be <= safety max)")
    p.add_argument("--memory", default=None, help="Memory e.g. '16G' (override, must be <= safety max)")
    p.add_argument("--walltime", default=None, help="Walltime e.g. '04:00:00' (override, must be <= safety max)")
    p.set_defaults(func=cmd_submit)

    # batch-submit
    p = subparsers.add_parser("batch-submit", parents=[parent_parser], help="Submit multiple jobs from file")
    p.add_argument("--file", required=True, help="JSON experiments file")
    p.set_defaults(func=cmd_batch_submit)

    # status
    p = subparsers.add_parser("status", parents=[parent_parser], help="Check job statuses")
    p.add_argument("--cluster", default=None, help="Specific cluster (default: all)")
    p.set_defaults(func=cmd_status)

    # cancel
    p = subparsers.add_parser("cancel", parents=[parent_parser], help="Cancel specific jobs")
    p.add_argument("--cluster", required=True, help="Cluster name")
    p.add_argument("--job-ids", required=True, help="Comma-separated job IDs")
    p.set_defaults(func=cmd_cancel)

    # logs
    p = subparsers.add_parser("logs", parents=[parent_parser], help="Fetch job stdout log")
    p.add_argument("--cluster", default=None, help="Cluster name (optional if using --experiment-id)")
    p.add_argument("--job-id", default=None, help="SLURM job ID (optional if using --experiment-id)")
    p.add_argument("--experiment-id", default=None, help="Look up by experiment ID instead of job ID")
    p.add_argument("--lines", type=int, default=200, help="Number of lines")
    p.set_defaults(func=cmd_logs)

    # errors
    p = subparsers.add_parser("errors", parents=[parent_parser], help="Fetch job error/crash logs")
    p.add_argument("--cluster", default=None, help="Cluster name (optional if using --experiment-id)")
    p.add_argument("--job-id", default=None, help="SLURM job ID (optional if using --experiment-id)")
    p.add_argument("--experiment-id", default=None, help="Look up by experiment ID instead of job ID")
    p.add_argument("--lines", type=int, default=200, help="Number of lines")
    p.set_defaults(func=cmd_errors)

    # check-completions
    p = subparsers.add_parser("check-completions", parents=[parent_parser], help="Check for completed jobs")
    p.add_argument("--cluster", default=None, help="Specific cluster (default: all)")
    p.set_defaults(func=cmd_check_completions)

    # sync
    p = subparsers.add_parser("sync", parents=[parent_parser], help="Sync project code to cluster")
    p.add_argument("--cluster", default=None, help="Cluster name (default: all)")
    p.set_defaults(func=cmd_sync)

    # pull
    p = subparsers.add_parser("pull", parents=[parent_parser], help="Pull results from cluster")
    p.add_argument("--cluster", required=True, help="Cluster name")
    p.add_argument("--job-id", default=None, help="Specific job ID")
    p.add_argument("--output", default=None, help="Local output directory")
    p.set_defaults(func=cmd_pull)

    # ls
    p = subparsers.add_parser("ls", parents=[parent_parser], help="List files on cluster")
    p.add_argument("--cluster", required=True, help="Cluster name")
    p.add_argument("--path", default=None, help="Remote path")
    p.add_argument("--pattern", default=None, help="Filename pattern")
    p.set_defaults(func=cmd_ls)

    # build
    p = subparsers.add_parser("build", parents=[parent_parser], help="Build Singularity container")
    p.add_argument("--dockerfile", default=None, help="Path to Dockerfile")
    p.add_argument("--image-name", default=None, help="Output image name")
    p.add_argument("--tag", default="latest", help="Docker tag")
    p.add_argument("--registry", default=None, help="Docker registry")
    p.add_argument("--step", default=None, choices=["docker", "test", "push-docker", "singularity"],
                   help="Run a single step instead of full pipeline")
    p.add_argument("--test-command", default=None, help="Custom test command for the 'test' step")
    p.add_argument("--skip-tests", action="store_true", help="Skip the test step in full pipeline")
    p.set_defaults(func=cmd_build)

    # push-image
    p = subparsers.add_parser("push-image", parents=[parent_parser], help="Push container to cluster")
    p.add_argument("--cluster", default=None, help="Cluster name (default: all)")
    p.add_argument("--image", default=None, help="Path to .sif file")
    p.set_defaults(func=cmd_push_image)

    # verify-image
    p = subparsers.add_parser("verify-image", parents=[parent_parser], help="Verify container on cluster")
    p.add_argument("--cluster", required=True, help="Cluster name")
    p.set_defaults(func=cmd_verify_image)

    # journal (with sub-subcommands)
    p = subparsers.add_parser("journal", parents=[parent_parser], help="Research journal — persistent research memory")
    jp = p.add_subparsers(dest="journal_command", required=True)

    jp.add_parser("read", parents=[parent_parser], help="Read the full journal")

    jw = jp.add_parser("write", parents=[parent_parser], help="Append an entry to the journal")
    jw.add_argument("entry", help="Journal entry text (markdown)")

    p.set_defaults(func=cmd_journal)

    # budget
    p = subparsers.add_parser("budget", parents=[parent_parser], help="Show compute budget")
    p.set_defaults(func=cmd_budget)

    # report
    p = subparsers.add_parser("report", parents=[parent_parser], help="Generate full research report from journal history")
    p.add_argument("--output", default=None, help="Output file path (default: research_report.md)")
    p.set_defaults(func=cmd_report)

    # results (with sub-subcommands)
    p = subparsers.add_parser("results", parents=[parent_parser], help="Query the results bank")
    rp = p.add_subparsers(dest="results_command", required=True)
    rp.add_parser("summary", parents=[parent_parser], help="Results bank summary")
    rp.add_parser("experiments", parents=[parent_parser], help="All experiment results")
    rp.add_parser("hypotheses", parents=[parent_parser], help="All hypotheses with status/notes")
    rp.add_parser("open", parents=[parent_parser], help="Open hypotheses (worth revisiting)")
    rp.add_parser("promising", parents=[parent_parser], help="Promising hypotheses (active)")
    rp.add_parser("closed", parents=[parent_parser], help="Closed hypotheses (dead ends)")
    rh = rp.add_parser("hypothesis", parents=[parent_parser], help="A hypothesis + its experiments")
    rh.add_argument("--id", required=True, help="Hypothesis ID")
    re = rp.add_parser("experiment", parents=[parent_parser], help="Results for an experiment")
    re.add_argument("--id", required=True, help="Experiment ID")
    p.set_defaults(func=cmd_results)

    # validate
    p = subparsers.add_parser("validate", parents=[parent_parser], help="Validate command against safety rules")
    p.add_argument("--command", required=True, help="Command to validate")
    p.set_defaults(func=cmd_validate)

    # audit
    p = subparsers.add_parser("audit", parents=[parent_parser], help="Show audit log")
    p.add_argument("--limit", type=int, default=50, help="Number of entries")
    p.set_defaults(func=cmd_audit)

    # job-history
    p = subparsers.add_parser("job-history", parents=[parent_parser], help="Show tracked job history with walltime/resources")
    p.add_argument("--limit", type=int, default=50, help="Number of entries")
    p.set_defaults(func=cmd_job_history)

    # reconcile
    p = subparsers.add_parser("reconcile", parents=[parent_parser], help="Reconcile local job tracker with actual SLURM state")
    p.set_defaults(func=cmd_reconcile)

    # reset
    p = subparsers.add_parser("reset", parents=[parent_parser], help="Reset all xgenius state for a fresh research run")
    p.set_defaults(func=cmd_reset)

    # watch
    p = subparsers.add_parser("watch", parents=[parent_parser], help="Start background watcher daemon")
    p.add_argument("--quiet", action="store_true", help="Suppress output")
    p.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
