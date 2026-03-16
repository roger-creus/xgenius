# xgenius

LLM-oriented autonomous research platform for SLURM clusters.

xgenius enables Claude Code to autonomously run experiments on SLURM clusters: formulate hypotheses, modify code, submit jobs, analyze results, and iterate — with safety guarantees for shared infrastructure.

## How it works

```
┌─────────────────────────────────────────────────────────────┐
│  Your dev machine                                           │
│                                                             │
│  Claude Code ←──── xgenius watch (wakes Claude on job done) │
│    ↓ calls                         ↑ polls clusters         │
│  xgenius submit / status / pull / journal / ...             │
└──────────────────────────┬──────────────────────────────────┘
                           │ SSH
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  SLURM Cluster (Singularity container, sandboxed)           │
│  sbatch → job runs → writes .done marker on completion      │
└─────────────────────────────────────────────────────────────┘
```

1. Claude submits experiments via `xgenius submit`
2. Jobs run on the cluster inside Singularity containers
3. `xgenius watch` daemon detects completions and triggers `claude --continue`
4. Claude wakes up, pulls results, analyzes, and iterates

## Prerequisites

- Python 3.11+
- [Claude Code](https://claude.ai/code) with an active subscription
- SSH access to at least one SLURM cluster
- Docker + Singularity/Apptainer (for container builds)

## Installation

```bash
pip install xgenius
```

Or from source:

```bash
git clone https://github.com/roger-creus/xgenius.git
cd xgenius
pip install -e .
```

## One-time setup

### 1. Set up Claude Code auth token

`xgenius watch` needs to invoke `claude --continue` non-interactively. This requires a long-lived auth token:

```bash
claude setup-token
```

**Important:** Make sure you do NOT have an `ANTHROPIC_API_KEY` environment variable set, as it overrides subscription auth and causes failures:

```bash
# Check if set:
echo $ANTHROPIC_API_KEY

# If set in conda:
conda env config vars unset ANTHROPIC_API_KEY

# If set in shell config, remove the export line from ~/.bashrc or ~/.zshrc
```

### 2. Set up SSH access to your clusters

xgenius connects to clusters via SSH. You need passwordless SSH key authentication.

**Basic setup** (if your cluster doesn't require MFA):

```
# ~/.ssh/config
Host mycluster
  HostName mycluster.example.com
  User myuser
  IdentityFile ~/.ssh/id_ed25519
```

## Quick start

### 1. Create a dedicated research repo

**Important:** Claude needs push access to the repo. Create a **new repo** for your research (do NOT fork — Claude might accidentally PR upstream).

```bash
# Option A: Start from an existing codebase
mkdir auto-myproject
cp -r original-project/* auto-myproject/
cd auto-myproject
git init
git add -A
git commit -m "initial: import codebase"
# Create repo on GitHub, then:
git remote add origin git@github.com:yourusername/auto-myproject.git
git push -u origin main

# Option B: Start fresh
mkdir auto-myproject
cd auto-myproject
git init
# ... add your code ...
```

**Requirements:**
- Claude must be able to `git push` — use SSH keys or `gh auth login`
- `gh` CLI should be installed (`brew install gh` / `sudo apt install gh`)
- The repo should be private if your research is pre-publication

### 2. Initialize xgenius

```bash
cd auto-myproject
xgenius init
```

This interactively creates:
- `xgenius.toml` — cluster config, SLURM settings, safety limits
- `research_goal.md` — describe what you want Claude to achieve
- `.xgenius/` — runtime state directory with templates, journal, job tracker
- `CLAUDE.md` — tool documentation and git conventions for Claude

### 2. Configure `xgenius.toml`

The config file has three sections. **Read the [Configuration Guide](#configuration-guide) below carefully** — getting this right is critical.

Key things to set:
- **`[safety]`** — maximum resource limits Claude cannot exceed
- **`[clusters.NAME]`** — SSH hostname (must match `~/.ssh/config`), paths on the cluster
- **`[clusters.NAME.slurm]`** — default SLURM parameters and available GPU types

See [`examples/xgenius.toml`](examples/xgenius.toml) for a fully commented example.

### 3. Edit your research goal

Open `research_goal.md` and describe your objective, baselines, success criteria, and constraints. Be specific — this is what Claude reads to decide what experiments to run.

### 4. Build and push the container

Open Claude Code in your project directory and tell it:

```
Build the Singularity container for this project. Make sure the code runs correctly inside it. Then push it to the cluster.
```

Claude will:
- Examine the Dockerfile and fix issues (outdated base images, missing deps)
- Run `xgenius build --json` (docker build → test → singularity convert)
- Run `xgenius push-image --cluster NAME --json` (push + verify on cluster)

**Important:** The container should contain only dependencies (CUDA, Python, pip packages), NOT source code. Code is synced separately via `xgenius sync` and mounted at runtime.

### 5. Start the autonomous research loop

Run in a single terminal (tmux recommended):

```bash
cd auto-myproject
claude -p "Start the autonomous research loop. Read CLAUDE.md and research_goal.md and begin." --dangerously-skip-permissions && xgenius watch
```

The agent runs first (`claude -p`), does its initial work (reads goal, submits baseline experiments), and exits. Then the watcher daemon starts automatically (`&&`), polls clusters for completed jobs, and triggers a **fresh** `claude -p "..."` session when results are ready. Each wake-up is a clean session — no stale context accumulation.

**Important:** The watcher MUST start after the initial agent exits. The `&&` ensures this. Do NOT run them in parallel — it causes duplicate Claude sessions.

**Safety:** The watcher will never trigger Claude if another Claude process is already running in the project directory. Completions are accumulated and delivered in the next cycle.

**Warning:** Do not start other Claude Code sessions in the same project directory while the research loop is running — the watcher detects Claude processes by directory and will skip polling cycles until they exit.

**Monitor progress:**
```bash
tail -f .xgenius/watcher.log           # watcher activity
xgenius journal summary                # research progress
xgenius status                         # running jobs
xgenius job-history --json             # past jobs with walltimes
```

### Resetting for a fresh run

```bash
cd auto-myproject
xgenius reset                          # clear journal, jobs, audit log
git add -A && git commit -m "reset: fresh research run"
git push
```

The agent will start fresh and begin the research loop from scratch.

## Configuration Guide

### `xgenius.toml` structure

The config has four sections. See [`examples/xgenius.toml`](examples/xgenius.toml) for a fully commented example.

#### `[project]` — Project metadata

```toml
[project]
name = "my-research"                 # Project name
research_goal = "research_goal.md"   # Path to research goal (Claude reads this)
container_image = "my-project.sif"   # Singularity image filename
dockerfile = "Dockerfile"            # Path to Dockerfile
```

#### `[safety]` — Hard limits Claude cannot exceed

These are **maximums**. Claude can request less per-job via `--gpus`, `--cpus`, `--memory`, `--walltime` flags. Set these to the most you'd ever want a single job to use.

```toml
[safety]
max_gpus_per_job = 4                 # Max GPUs per single job
max_cpus_per_job = 32                # Max CPUs per single job
max_memory_per_job = "128G"          # Max RAM per job
max_walltime = "48:00:00"            # Max walltime per job
max_concurrent_jobs = 50             # Max jobs running/pending at once
max_total_gpu_hours = 10000          # Total GPU-hours budget across all experiments
allowed_command_prefixes = ["python"] # Only allow running Python scripts
forbidden_patterns = [               # Always blocked (shell injection protection)
    "rm -rf", "sudo", "chmod", "chown", "wget", "curl",
    "mkfs", "dd ", "shutdown", "reboot", "kill -9",
]
require_singularity = true           # All jobs must run inside a container
```

#### `[watcher]` — Background daemon settings

```toml
[watcher]
poll_interval_seconds = 60           # How often to check for completed jobs
trigger_command = "claude --continue" # Command to wake Claude up
```

#### `[clusters.NAME]` — One section per SLURM cluster

You can define multiple clusters. Claude will submit jobs to whichever cluster you configure.

```toml
[clusters.mycluster]
hostname = "mycluster"               # MUST match a Host entry in ~/.ssh/config
username = "myuser"                  # SSH username on the cluster
project_path = "/home/myuser/project" # Where project code lives (absolute path)
scratch_path = "/scratch/myuser"     # Scratch space for outputs/state (absolute path)
image_path = "/scratch/myuser/images" # Where .sif container images are stored
sbatch_template = "slurm_account_template.sbatch"
# Use "slurm_account_template.sbatch" if your cluster uses --account
# Use "slurm_partition_template.sbatch" if your cluster uses --partition
```

#### `[clusters.NAME.slurm]` — Default SLURM parameters

These are **defaults** — used when Claude doesn't specify overrides. Claude can request different values per-job within the `[safety]` limits.

```toml
[clusters.mycluster.slurm]
account = "my-allocation"            # SLURM account (--account). Leave "" if using partition.
partition = ""                       # SLURM partition (--partition). Leave "" if using account.
num_gpus = 1                         # Default GPUs per job
gpu_type = "a100"                    # Default GPU type. Leave "" for any GPU.
available_gpu_types = [              # All GPU types Claude can pick from on this cluster
    "a100",                          # Full A100
    "a100_3g.20gb",                  # MIG: 3/8 of A100 (good for quick tests)
    "v100",                          # Older GPU (cheaper/faster to schedule)
]
num_cpus = 8                         # Default CPUs per job
memory = "32G"                       # Default RAM per job
walltime = "12:00:00"                # Default walltime per job
modules = "apptainer"                # Modules to load before running container
singularity_command = "apptainer"    # "singularity" or "apptainer"
output_dir_cluster = "/scratch/myuser/runs"  # Where experiment outputs go
output_dir_container = "/results"    # Mount point inside the container
```

### GPU types

Many modern clusters support [Multi-Instance GPU (MIG)](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/), which splits a single GPU into smaller virtual GPUs. This is useful for quick test runs that don't need a full GPU.

Configure `available_gpu_types` per-cluster with all GPU types available on that cluster. Claude can then choose the right GPU for each job:

```bash
# Quick test on a MIG slice (smaller, faster to schedule)
xgenius submit --gpu-type "a100_3g.20gb" --walltime "01:00:00" --command "python test.py"

# Full training run on a complete GPU
xgenius submit --gpu-type "a100" --walltime "12:00:00" --command "python train.py"
```

### Resource management

Claude can override defaults per-job using flags on `xgenius submit`:

| Flag | Description | Example |
|------|-------------|---------|
| `--gpus N` | Number of GPUs | `--gpus 1` |
| `--gpu-type TYPE` | GPU model/MIG slice | `--gpu-type "a100_3g.20gb"` |
| `--cpus N` | Number of CPUs | `--cpus 4` |
| `--memory SIZE` | RAM | `--memory "16G"` |
| `--walltime TIME` | Job duration | `--walltime "02:00:00"` |

Claude uses `xgenius job-history --json` to learn how long past jobs took and adjusts future requests accordingly. `xgenius status --json` shows pending times and queue reasons so Claude can make smart scheduling decisions.

### Multiple clusters

Define multiple clusters to let Claude distribute jobs across them:

```toml
[clusters.fast-cluster]
hostname = "fast-robot"
# ... (GPU cluster for training)

[clusters.test-cluster]
hostname = "test-robot"
# ... (smaller cluster for quick tests)
```

Claude will see all configured clusters and can choose which to submit to based on availability and GPU types.

## Safety

Safety is enforced in Python code — Claude cannot bypass it:

1. **Command validation**: Only allowed prefixes (e.g., `python`). Shell injection blocked.
2. **Resource limits**: Max GPUs, CPUs, memory, walltime per job (from `[safety]`).
3. **GPU type validation**: Only GPU types listed in `available_gpu_types` are allowed.
4. **Budget tracking**: Total GPU-hours cap across all experiments.
5. **Path containment**: Code changes restricted to project directory.
6. **Singularity sandboxing**: All code runs inside containers.
7. **Audit log**: Every action logged to `.xgenius/audit.jsonl`.

## Commands

All commands support `--json` for structured output.

| Command | Purpose |
|---------|---------|
| `xgenius init` | Initialize project (creates config, research goal, CLAUDE.md) |
| `xgenius build` | Build Singularity container (docker build → test → convert) |
| `xgenius push-image` | Push container to cluster and verify |
| `xgenius verify-image` | Verify container exists on cluster |
| `xgenius submit` | Submit a SLURM job (safety-validated, resource overrides) |
| `xgenius batch-submit` | Submit multiple jobs from JSON file |
| `xgenius status` | Check job statuses (elapsed time, pending reason, resources) |
| `xgenius cancel` | Cancel specific jobs by ID |
| `xgenius logs` | Fetch job stdout |
| `xgenius errors` | Fetch job stderr / crash logs |
| `xgenius check-completions` | Check for completed jobs |
| `xgenius sync` | Rsync project code to cluster |
| `xgenius pull` | Pull results from cluster |
| `xgenius ls` | List files on cluster |
| `xgenius journal context` | Full research context for Claude |
| `xgenius journal summary` | Concise progress summary |
| `xgenius journal add-hypothesis` | Record a hypothesis |
| `xgenius journal add-result` | Record experiment results |
| `xgenius journal update-hypothesis` | Update hypothesis status |
| `xgenius budget` | Show remaining compute budget |
| `xgenius validate` | Dry-run safety check on a command |
| `xgenius audit` | View audit log |
| `xgenius job-history` | View past jobs with walltime, resources, and log paths |
| `xgenius reconcile` | Sync local job tracker with actual SLURM state |
| `xgenius reset` | Clear all state for a fresh research run |
| `xgenius watch` | Background daemon (triggers Claude on job completion) |

## Citation

If you use xgenius in your research, please cite it:

```bibtex
@software{creus2026xgenius,
  title = {xgenius: LLM-Oriented Autonomous Research Platform for SLURM Clusters},
  author = {Creus Castanyer, Roger},
  year = {2026},
  url = {https://github.com/roger-creus/xgenius},
  doi = {10.5281/zenodo.19038735},
  version = {1.0.0}
}
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19038735.svg)](https://doi.org/10.5281/zenodo.19038735)

## License

MIT — see [LICENSE](LICENSE) for details.
