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

```bash
claude setup-token
```
### 2. Set up SSH access to your clusters

You need passwordless SSH access to your SLURM clusters. 

## Quick start

### 1. Initialize your project

Clone your research codebase and run:

```bash
cd my-project
xgenius init
```

This interactively creates:
- `xgenius.toml` — cluster config, SLURM settings, safety limits
- `research_goal.md` — describe what you want Claude to achieve
- `.xgenius/` — runtime state directory (auto-gitignored)
- `CLAUDE.md` — tool documentation for Claude

### 2. Edit your research goal

Open `research_goal.md` and describe your objective, baselines, success criteria, and constraints.

### 3. Build and push the container

Open Claude Code in your project directory. Claude will use `xgenius build` to build a Docker image, run tests, and convert to Singularity. Tell Claude:

```
Build the Singularity container for this project. Make sure the code runs correctly inside it. Then push it to the cluster.
```

Claude will run:
```bash
xgenius build --json          # docker build → test → singularity convert
xgenius push-image --cluster mycluster --json  # push + verify on cluster
```

If any step fails, Claude reads the error, fixes the issue, and retries.

### 4. Start the watcher daemon

In a separate terminal (tmux!):

```bash
cd my-project
xgenius watch
```

This runs forever, polling your clusters for completed jobs and triggering `claude --continue` to wake Claude up.

### 5. Start the research loop

Tell Claude:

```
Start the autonomous research loop. Read research_goal.md and begin.
```

Claude will:
1. Read `xgenius journal context` for research state
2. Formulate a hypothesis
3. Modify code
4. Sync to cluster and submit experiments
5. Exit and wait for `xgenius watch` to trigger it on completion
6. Analyze results, record findings, iterate

## Configuration

### `xgenius.toml`

```toml
[project]
name = "my-research"
research_goal = "research_goal.md"
container_image = "my-project.sif"
dockerfile = "Dockerfile"

[safety]
max_gpus_per_job = 1
max_cpus_per_job = 16
max_memory_per_job = "64G"
max_walltime = "24:00:00"
max_concurrent_jobs = 10
max_total_gpu_hours = 500
allowed_command_prefixes = ["python"]
forbidden_patterns = ["rm -rf", "sudo", "chmod", "chown"]
require_singularity = true

[watcher]
poll_interval_seconds = 60
trigger_command = "claude --continue"

[clusters.mycluster]
hostname = "mycluster.example.com"    # Must match SSH config
username = "myuser"
project_path = "/home/myuser/my-project"
scratch_path = "/scratch/myuser"
image_path = "/scratch/myuser/images"
sbatch_template = "slurm_account_template.sbatch"

[clusters.mycluster.slurm]
account = "my-allocation"
num_gpus = 1
num_cpus = 8
memory = "32G"
walltime = "12:00:00"
modules = "apptainer"
singularity_command = "apptainer"
output_dir_cluster = "/scratch/myuser/runs"
output_dir_container = "/results"
```

### Safety

Safety is enforced in Python code — Claude cannot bypass it:

1. **Command validation**: Only allowed prefixes (e.g., `python`). Shell injection blocked.
2. **Resource limits**: Max GPUs, CPUs, memory, walltime per job.
3. **Budget tracking**: Total GPU-hours cap across all experiments.
4. **Path containment**: Code changes restricted to project directory.
5. **Singularity sandboxing**: All code runs inside containers.
6. **Audit log**: Every action logged to `.xgenius/audit.jsonl`.

## Commands available to Claude

| Command | Purpose |
|---------|---------|
| `xgenius init` | Initialize project (creates config, research goal, CLAUDE.md) |
| `xgenius build` | Build Singularity container (docker build → test → convert) |
| `xgenius push-image` | Push container to cluster and verify |
| `xgenius verify-image` | Verify container works on cluster |
| `xgenius submit` | Submit a SLURM job (safety-validated) |
| `xgenius batch-submit` | Submit multiple jobs from JSON file |
| `xgenius status` | Check job statuses across clusters |
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
| `xgenius watch` | Background daemon (triggers Claude on job completion) |

## License

MIT
