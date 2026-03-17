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
    """Set up autonomous research project with template config."""
    from xgenius.config import ensure_xgenius_dir

    project_dir = os.getcwd()
    config_path = os.path.join(project_dir, "xgenius.toml")

    if os.path.exists(config_path) and not args.force:
        console.print("[yellow]xgenius.toml already exists. Use --force to overwrite.[/yellow]")
        return

    default_name = os.path.basename(project_dir)

    # Write template xgenius.toml — user edits this
    template = f"""# xgenius.toml — Edit this file to configure your autonomous research project.
# See https://github.com/roger-creus/xgenius for documentation.

[project]
name = "{default_name}"
research_goal = "research_goal.md"       # Path to your research goal markdown file
container_image = "{default_name}.sif"   # Singularity image filename
dockerfile = "Dockerfile"                # Path to Dockerfile

[safety]
max_gpus_per_job = 1                     # Maximum GPUs per single job
max_cpus_per_job = 16                    # Maximum CPUs per single job
max_memory_per_job = "64G"               # Maximum RAM per job
max_walltime = "24:00:00"                # Maximum walltime per job
max_concurrent_jobs = 50                 # Maximum jobs running at once
max_total_gpu_hours = 10000              # Total GPU-hours budget
allowed_command_prefixes = ["python"]     # Commands the agent can run
forbidden_patterns = ["rm -rf", "sudo", "chmod", "chown", "wget", "curl", "mkfs", "dd ", "shutdown", "reboot", "kill -9"]
require_singularity = true

[watcher]
poll_interval_seconds = 60
trigger_command = "claude --dangerously-skip-permissions"

# --- Add your clusters below ---
# Copy and modify this template for each cluster.
# hostname must match an entry in your ~/.ssh/config

# [clusters.mycluster]
# hostname = "mycluster-robot"           # SSH config host name
# username = "myuser"                    # SSH username
# project_path = "/home/myuser/project"  # Absolute path to code on cluster
# scratch_path = "/scratch/myuser"       # Scratch space for outputs
# image_path = "/scratch/myuser/images"  # Where .sif containers are stored
# sbatch_template = "slurm_account_template.sbatch"  # or slurm_partition_template.sbatch
#
# [clusters.mycluster.slurm]
# account = "my-allocation"              # SLURM account (leave "" if using partition)
# partition = ""                         # SLURM partition (leave "" if using account)
# num_gpus = 1                           # Default GPUs per job
# gpu_type = ""                          # Default GPU type (e.g., "a100", "h100")
# available_gpu_types = []               # All GPU types available on this cluster
# num_cpus = 8                           # Default CPUs per job
# memory = "32G"                         # Default RAM per job
# walltime = "12:00:00"                  # Default walltime per job
# modules = "apptainer"                  # Modules to load
# singularity_command = "apptainer"      # "singularity" or "apptainer"
# output_dir_cluster = "/scratch/myuser/runs"  # Experiment output directory
# output_dir_container = "/results"      # Mount point inside container
"""

    with open(config_path, "w") as f:
        f.write(template)
    console.print(f"[green]Created xgenius.toml[/green] — Edit this to add your clusters.")

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

    # Create .xgenius directory structure
    xgenius_dir = os.path.join(project_dir, ".xgenius")
    for subdir in ["markers", "batches", "templates", "slurm_logs"]:
        os.makedirs(os.path.join(xgenius_dir, subdir), exist_ok=True)

    # Create standard files
    for fname in ["journal.md", "DEBUG.md"]:
        fpath = os.path.join(xgenius_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                if fname == "DEBUG.md":
                    f.write("# Debug Log\n\nErrors and issues encountered during autonomous research.\n")
    console.print("[green]Created .xgenius/ directory[/green]")

    # Create run ID
    from xgenius.config import create_run_id
    with open(os.path.join(xgenius_dir, "run_id"), "w") as f:
        run_id = create_run_id()
        f.write(run_id)
    console.print(f"[green]Run ID: {run_id}[/green]")

    # Create results directory
    os.makedirs(os.path.join(project_dir, "results"), exist_ok=True)
    console.print("[green]Created results/ directory[/green]")

    # Copy SBATCH templates
    from xgenius.templates import copy_templates_to_project
    copy_templates_to_project(project_dir)
    console.print("[green]Copied SBATCH templates to .xgenius/templates/[/green]")

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
- Push after every commit — always push to the default branch (usually `main`): `git push origin main`
- Check the default branch with `git remote show origin | grep HEAD` if unsure
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
- `xgenius logs --experiment-id ID --json` — Read job stdout from local slurm_logs/ (also: --job-id)
- `xgenius errors --experiment-id ID --json` — Read job errors from local slurm_logs/ (also: --job-id)
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
- `xgenius db summary --json` — Full status overview (jobs by status, per-hypothesis breakdown)
- `xgenius db jobs --json` — All jobs (filter: `--hypothesis-id H`, `--status running`)
- `xgenius db job --id JOBID --json` — Single job details
- `xgenius db active --json` — Currently running/submitted jobs
- `xgenius db hypothesis-check --id H --json` — Check if all jobs for a hypothesis are done
- `xgenius db hypotheses --json` — All hypotheses in DB
- `xgenius db hypothesis-update --id H --status S --description D --conclusion C --comment N` — Update hypothesis metadata
- `xgenius db job-update --id JOBID --comment "notes"` — Add notes to a specific job
- `xgenius reset` — Clear all state for a fresh research run

All commands support `--json` for structured output.

### Debugging Failed Jobs
When a job fails:
1. Run `xgenius errors --experiment-id EXPERIMENT_ID --json` to see tracebacks and error messages
2. Run `xgenius logs --experiment-id EXPERIMENT_ID --json` to see full stdout
3. Run `xgenius db jobs --json` to see all jobs with their statuses and walltimes
4. SLURM logs are automatically pulled to `.xgenius/slurm_logs/{hypothesis_id}/{experiment_id}/` when jobs complete

### Debug Log
When you encounter errors (cluster issues, submission failures, crashes, unexpected behavior), append a timestamped entry to `.xgenius/DEBUG.md`. Format:
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

The filename format is `{hypothesis_id}__{experiment_id}.csv` — the double underscore separates hypothesis from experiment for easy parsing.

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

Use `xgenius db jobs --status completed --json` to see how long past jobs took, then adjust walltime accordingly.
Use `xgenius status --json` to see pending/running jobs with their elapsed time, submit time, and pending reason.

### Two State Systems — DB + Journal

**SQLite DB** (`.xgenius/xgenius.db`) — AUTOMATED operational state. Updated by the watcher every cycle.
- Job statuses (submitted/pending/running/completed/failed/etc.)
- Walltimes, exit codes, timestamps, results_pulled flag
- Query with: `xgenius db summary --json`, `xgenius db jobs --json`, `xgenius db active --json`

**Research Journal** (`.xgenius/journal.md`) — YOUR persistent research memory. Written by you.
- What hypotheses were tried and why
- Key findings, insights, and conclusions
- Ideas for future investigation
- Decisions and rationale
- Read with: `xgenius journal read`
- Write with: `xgenius journal write "your entry here"`

**Every session, you MUST:**
1. Read the journal (`xgenius journal read`) to recall what previous sessions did
2. Check the DB (`xgenius db summary --json`) for current job states
3. Before exiting, write a journal entry summarizing what you did and what to do next

### Research Workflow
1. Read `xgenius journal read` for research memory from previous sessions
2. Check `xgenius db summary --json` for all job states and hypothesis status
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
- `.xgenius/slurm_logs/` — SLURM .out/.err files organized by hypothesis/experiment (pulled automatically by watcher)
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

    _output(result.to_dict(), args.json)
    if not result.success:
        sys.exit(1)


# --- Batch Submit ---

def cmd_batch_submit(args):
    """Submit multiple jobs from a config file."""
    config = _load_config(args)
    from xgenius.jobs import JobManager
    from xgenius.config import get_xgenius_dir, ensure_xgenius_dir
    import shutil
    import time as _time

    manager = JobManager(config)

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
    """Fetch job stdout logs from local .xgenius/slurm_logs/."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)
    output = manager.logs(job_id=args.job_id or "", experiment_id=args.experiment_id or "", lines=args.lines)
    _output(output, args.json)


# --- Errors ---

def cmd_errors(args):
    """Fetch job error logs from local .xgenius/slurm_logs/."""
    config = _load_config(args)
    from xgenius.jobs import JobManager

    manager = JobManager(config)
    output = manager.errors(job_id=args.job_id or "", experiment_id=args.experiment_id or "", lines=args.lines)
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
    """Spawn a Claude agent to generate a thorough research report."""
    import subprocess
    config = _load_config(args)
    from xgenius.config import get_project_dir

    project_dir = get_project_dir(config)
    report_dir = os.path.join(project_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    output_md = os.path.join(report_dir, "report.md")
    output_html = os.path.join(report_dir, "report.html")

    prompt = f"""You are a research report writer. Generate a thorough, publication-quality research report for a human audience.

## Your task
Read ALL available data and produce a comprehensive report in the `report/` directory.
This directory is overwritten each time — it's a standalone snapshot of the research.
Save:
- Markdown: `{output_md}`
- HTML: `{output_html}` (self-contained, styled, with embedded plots as base64 images)
- Plots: `report/plots/` (PNG files)

## Data sources to read (in order)
1. `research_goal.md` — the original research objective
2. `xgenius journal read` — the research narrative and decisions made
3. `xgenius db summary --json` — overview of all jobs and hypotheses
4. `xgenius db hypotheses --json` — all hypotheses with status and conclusions
5. `xgenius db jobs --json` — all experiments with walltimes and exit codes
6. `cat results/experiments.csv` — the results bank with metrics
7. `cat results/hypotheses.csv` — hypothesis status and notes
8. `cat .xgenius/DEBUG.md` — any errors encountered
9. `git log --oneline -50` — recent git history showing research progression

## Report structure
Write a markdown report with:
1. **Title and Abstract** — summarize the entire research campaign
2. **Research Goal** — what was the objective
3. **Methodology** — how experiments were run (clusters, compute, tools)
4. **Baselines** — baseline results with tables
5. **Investigations** — for each hypothesis: motivation, what was tried, results (with tables/numbers), analysis, conclusion
6. **Key Findings** — what worked, what didn't, and why
7. **Performance Progression** — show how performance improved over time, from baselines through each iteration
8. **Compute Statistics** — total jobs, GPU-hours, clusters used, success/failure rates
9. **Conclusions and Future Work** — what was achieved, what remains

## Requirements
- Include ACTUAL NUMBERS from the results bank — do not make up data
- Create comparison tables showing baseline vs best results
- Create Python plots using matplotlib and save them to `report/plots/` — embed them in the markdown as `![](plots/filename.png)`
- Plot learning curves, bar charts comparing algorithms, progression over time
- Be thorough — this is for a human who wants to understand everything that happened
- Write clearly and technically — this could go in a paper appendix

## Output
1. Clean the `report/` directory first: remove old plots and files
2. Save plots to `report/plots/`
3. Save the markdown report to `{output_md}` (reference plots as `![](plots/filename.png)`)
4. Convert to a self-contained HTML file at `{output_html}` using Python:
   - Use the `markdown` library (install with pip if needed)
   - Embed plots as base64 images so the HTML is fully self-contained (portable single file)
   - Add clean CSS styling (readable fonts, max-width, nice tables, code blocks)
   - The human will open this HTML in a browser
5. The entire `report/` directory should be downloadable as a standalone research report
"""

    console.print("[bold]Generating research report...[/bold]")
    console.print("This spawns a Claude agent to analyze all data and produce a thorough report.")

    result = subprocess.run(
        ["claude", "-p", prompt, "--dangerously-skip-permissions"],
        cwd=project_dir,
    )

    if result.returncode == 0:
        console.print(f"[green]Report saved to {report_dir}/[/green]")
        import webbrowser
        if os.path.exists(output_html):
            webbrowser.open(f"file://{os.path.abspath(output_html)}")
    else:
        console.print(f"[red]Report generation failed (exit code {result.returncode})[/red]")


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
    from xgenius.config import get_xgenius_dir, create_run_id

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

    # Generate new run ID
    new_run_id = create_run_id()
    with open(os.path.join(xgenius_dir, "run_id"), "w") as f:
        f.write(new_run_id)
    console.print(f"[green]New run ID: {new_run_id}[/green]")

    files_to_clear = ["journal.md", "watcher.log", "DEBUG.md"]
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

def cmd_db(args):
    """Query the xgenius DB."""
    config = _load_config(args)
    from xgenius.db import XGeniusDB

    db = XGeniusDB(config)

    if args.db_command == "jobs":
        if args.hypothesis_id:
            _output(db.get_jobs_by_hypothesis(args.hypothesis_id), args.json)
        elif args.status:
            _output(db.get_jobs_by_status(args.status), args.json)
        else:
            _output(db.get_all_jobs(limit=args.limit), args.json)
    elif args.db_command == "job":
        _output(db.get_job(args.id) or {"error": "Job not found"}, args.json)
    elif args.db_command == "hypotheses":
        _output(db.get_all_hypotheses(), args.json)
    elif args.db_command == "summary":
        status = db.get_full_status()
        _output(status, args.json)
    elif args.db_command == "active":
        jobs = db.get_pending_jobs()
        _output(jobs, args.json)
    elif args.db_command == "hypothesis-check":
        hid = args.id
        summary = db.get_hypothesis_job_summary(hid)
        complete = db.is_hypothesis_complete(hid)
        _output({"hypothesis_id": hid, "complete": complete, **summary}, args.json)
    elif args.db_command == "hypothesis-update":
        db.update_hypothesis(
            hypothesis_id=args.id,
            status=args.status or "",
            description=args.description or "",
            conclusion=args.conclusion or "",
            comment=args.comment or "",
        )
        _output({"status": "updated"}, args.json)
    elif args.db_command == "job-update":
        db.update_job_notes(args.id, error_message=args.comment or "")
        _output({"status": "updated"}, args.json)


# --- Dashboard ---

def cmd_dashboard(args):
    """Start the web dashboard for inspecting the DB."""
    from xgenius.dashboard import run_dashboard
    run_dashboard(config_path=args.config, port=args.port)


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
    p = subparsers.add_parser("logs", parents=[parent_parser], help="Read job stdout from local slurm_logs/")
    p.add_argument("--job-id", default=None, help="SLURM job ID")
    p.add_argument("--experiment-id", default=None, help="Experiment ID")
    p.add_argument("--lines", type=int, default=200, help="Number of lines from end")
    p.set_defaults(func=cmd_logs)

    # errors
    p = subparsers.add_parser("errors", parents=[parent_parser], help="Read job errors from local slurm_logs/")
    p.add_argument("--job-id", default=None, help="SLURM job ID")
    p.add_argument("--experiment-id", default=None, help="Experiment ID")
    p.add_argument("--lines", type=int, default=200, help="Number of lines from end")
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
    # db (query the operational database)
    p = subparsers.add_parser("db", parents=[parent_parser], help="Query the xgenius DB")
    dp = p.add_subparsers(dest="db_command", required=True)

    dj = dp.add_parser("jobs", parents=[parent_parser], help="List jobs (filterable)")
    dj.add_argument("--hypothesis-id", default=None, help="Filter by hypothesis")
    dj.add_argument("--status", default=None, help="Filter by status (submitted/running/completed/failed/etc.)")
    dj.add_argument("--limit", type=int, default=100, help="Max results")

    djob = dp.add_parser("job", parents=[parent_parser], help="Get a single job by ID")
    djob.add_argument("--id", required=True, help="Job ID")

    dp.add_parser("hypotheses", parents=[parent_parser], help="List all hypotheses in DB")
    dp.add_parser("summary", parents=[parent_parser], help="Full status overview")
    dp.add_parser("active", parents=[parent_parser], help="Currently active (submitted/running) jobs")

    dhc = dp.add_parser("hypothesis-check", parents=[parent_parser], help="Check if all jobs for a hypothesis are done")
    dhc.add_argument("--id", required=True, help="Hypothesis ID")

    dhu = dp.add_parser("hypothesis-update", parents=[parent_parser], help="Update hypothesis metadata")
    dhu.add_argument("--id", required=True, help="Hypothesis ID")
    dhu.add_argument("--status", default=None, help="New status (proposed/testing/confirmed/rejected/open/closed/promising)")
    dhu.add_argument("--description", default=None, help="Update description")
    dhu.add_argument("--conclusion", default=None, help="Conclusion text")
    dhu.add_argument("--comment", default=None, help="Additional notes")

    dju = dp.add_parser("job-update", parents=[parent_parser], help="Add notes/metadata to a job")
    dju.add_argument("--id", required=True, help="Job ID")
    dju.add_argument("--comment", default=None, help="Notes about this job/experiment")

    p.set_defaults(func=cmd_db)

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

    # dashboard
    p = subparsers.add_parser("dashboard", parents=[parent_parser], help="Open web dashboard to inspect DB")
    p.add_argument("--port", type=int, default=8765, help="Port number")
    p.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
