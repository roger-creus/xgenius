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
            "trigger_command": "claude --continue",
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

    # Add .xgenius to .gitignore
    gitignore_path = os.path.join(project_dir, ".gitignore")
    gitignore_entry = ".xgenius/"
    if os.path.exists(gitignore_path):
        with open(gitignore_path) as f:
            content = f.read()
        if gitignore_entry not in content:
            with open(gitignore_path, "a") as f:
                f.write(f"\n# xgenius state\n{gitignore_entry}\n")
            console.print("[green]Added .xgenius/ to .gitignore[/green]")
    else:
        with open(gitignore_path, "w") as f:
            f.write(f"# xgenius state\n{gitignore_entry}\n")
        console.print("[green]Created .gitignore with .xgenius/[/green]")

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

### Available Commands

**Research Loop:**
- `xgenius journal context` — Full research context: goal, hypotheses, experiments, results, what to try next
- `xgenius journal summary` — Concise progress summary
- `xgenius journal add-hypothesis "text" --motivation "why" --expected "outcome"` — Record a hypothesis
- `xgenius journal add-result --experiment-id ID --metrics '{"key": value}' --analysis "text"` — Record results
- `xgenius journal update-hypothesis --id ID --status confirmed|rejected|partially_confirmed --conclusion "text"`

**Job Management:**
- `xgenius submit --cluster NAME --command "python script.py --args" [--experiment-id ID] [--hypothesis-id ID] [--gpus N] [--cpus N] [--memory "16G"] [--walltime "04:00:00"]` — Submit a job
- `xgenius batch-submit --file experiments.json` — Submit multiple jobs
- `xgenius status [--cluster NAME] [--json]` — Check job statuses
- `xgenius cancel --cluster NAME --job-ids ID1,ID2` — Cancel specific jobs
- `xgenius logs --cluster NAME --job-id ID` — View job stdout
- `xgenius errors --cluster NAME --job-id ID` — View job stderr/crashes
- `xgenius check-completions [--json]` — Check for newly completed jobs

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
- `xgenius job-history [--limit N] --json` — View past jobs with walltime, resources, and status

All commands support `--json` for structured output.

### Resource Management
The xgenius.toml [safety] section defines MAXIMUM resource limits. You can request LESS:
- `--gpus 1` instead of the max 4
- `--walltime "02:00:00"` for a quick test instead of the max 24h
- `--memory "16G"` if the job doesn't need much RAM
- `--cpus 4` for a lightweight job

Use `xgenius job-history --json` to see how long past jobs took, then adjust walltime accordingly.
Use `xgenius status --json` to see pending/running jobs with their elapsed time, submit time, and pending reason.

### Research Workflow
1. Run `xgenius journal context` to understand current state
2. Formulate a hypothesis and record it
3. Modify code to test the hypothesis
4. Run `xgenius sync` to push code to cluster
5. Run `xgenius submit` to start experiments
6. Wait for `xgenius watch` daemon to trigger you on completion
7. Analyze results, record findings, iterate

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
- All actions are logged to .xgenius/audit.jsonl
"""

    if os.path.exists(claude_md_path):
        with open(claude_md_path) as f:
            content = f.read()
        if "xgenius" not in content.lower():
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

    manager = JobManager(config)
    journal = ResearchJournal(config)

    with open(args.file) as f:
        experiments = json.load(f)

    if isinstance(experiments, dict):
        experiments = experiments.get("experiments", [])

    results = []
    for exp in experiments:
        result = manager.submit(
            cluster_name=exp["cluster"],
            command=exp["command"],
            experiment_id=exp.get("experiment_id", ""),
            hypothesis_id=exp.get("hypothesis_id", ""),
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

    if args.journal_command == "context":
        _output(journal.get_context(), args.json)
    elif args.journal_command == "summary":
        _output(journal.get_summary(), args.json)
    elif args.journal_command == "add-hypothesis":
        hid = journal.add_hypothesis(
            text=args.text,
            motivation=args.motivation or "",
            expected_outcome=args.expected or "",
        )
        _output({"hypothesis_id": hid}, args.json)
    elif args.journal_command == "add-result":
        metrics = json.loads(args.metrics)
        journal.add_result(
            experiment_id=args.experiment_id,
            metrics=metrics,
            analysis=args.analysis or "",
        )
        _output({"status": "recorded"}, args.json)
    elif args.journal_command == "update-hypothesis":
        journal.update_hypothesis(
            hypothesis_id=args.id,
            status=args.status,
            conclusion=args.conclusion or "",
        )
        _output({"status": "updated"}, args.json)
    else:
        _output({"error": f"Unknown journal command: {args.journal_command}"}, args.json)


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

    files_to_clear = ["jobs.jsonl", "journal.jsonl", "journal_summary.md", "audit.jsonl"]
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
    p = subparsers.add_parser("journal", parents=[parent_parser], help="Research journal operations")
    jp = p.add_subparsers(dest="journal_command", required=True)

    jp.add_parser("context", parents=[parent_parser], help="Full research context dump")
    jp.add_parser("summary", parents=[parent_parser], help="Concise progress summary")

    ah = jp.add_parser("add-hypothesis", parents=[parent_parser], help="Record a new hypothesis")
    ah.add_argument("text", help="Hypothesis text")
    ah.add_argument("--motivation", default="", help="Why this hypothesis")
    ah.add_argument("--expected", default="", help="Expected outcome")

    ar = jp.add_parser("add-result", parents=[parent_parser], help="Record experiment results")
    ar.add_argument("--experiment-id", required=True, help="Experiment ID")
    ar.add_argument("--metrics", required=True, help="JSON metrics string")
    ar.add_argument("--analysis", default="", help="Analysis text")

    uh = jp.add_parser("update-hypothesis", parents=[parent_parser], help="Update hypothesis status")
    uh.add_argument("--id", required=True, help="Hypothesis ID")
    uh.add_argument("--status", required=True, choices=["proposed", "testing", "confirmed", "rejected", "partially_confirmed"])
    uh.add_argument("--conclusion", default="", help="Conclusion text")

    p.set_defaults(func=cmd_journal)

    # budget
    p = subparsers.add_parser("budget", parents=[parent_parser], help="Show compute budget")
    p.set_defaults(func=cmd_budget)

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
