"""Configuration management for xgenius.

Loads and validates xgenius.toml project configuration files.
"""

import os
import tomllib
from dataclasses import dataclass, field


@dataclass
class SlurmConfig:
    """Per-cluster SLURM job parameters."""
    account: str = ""
    partition: str = ""
    num_gpus: int = 1
    gpu_type: str = ""  # default GPU type, e.g., "h100" — empty means any GPU
    available_gpu_types: list[str] = field(default_factory=list)  # all GPU types Claude can pick from
    num_cpus: int = 8
    memory: str = "32G"
    walltime: str = "12:00:00"
    modules: str = ""
    singularity_command: str = "singularity"
    output_dir_cluster: str = ""
    output_dir_container: str = "/results"
    output_file: str = ""


@dataclass
class ClusterConfig:
    """Configuration for a single cluster."""
    name: str
    hostname: str
    username: str
    project_path: str
    scratch_path: str
    image_path: str
    sbatch_template: str = "slurm_partition_template.sbatch"
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


@dataclass
class SafetyConfig:
    """Safety limits enforced on all operations."""
    max_gpus_per_job: int = 1
    max_cpus_per_job: int = 16
    max_memory_per_job: str = "64G"
    max_walltime: str = "24:00:00"
    max_concurrent_jobs: int = 10
    max_total_gpu_hours: float = 500
    allowed_command_prefixes: list[str] = field(default_factory=lambda: ["python"])
    forbidden_patterns: list[str] = field(default_factory=lambda: [
        "rm -rf", "sudo", "chmod", "chown", "wget", "curl",
        "mkfs", "dd ", "shutdown", "reboot", "kill -9",
    ])
    require_singularity: bool = True


@dataclass
class WatcherConfig:
    """Configuration for the background completion watcher."""
    poll_interval_seconds: int = 60
    trigger_command: str = "claude --dangerously-skip-permissions"


@dataclass
class ProjectConfig:
    """Top-level project settings."""
    name: str = "my-research"
    research_goal: str = "research_goal.md"
    container_image: str = ""
    dockerfile: str = "Dockerfile"


@dataclass
class XGeniusConfig:
    """Complete xgenius configuration."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    clusters: dict[str, ClusterConfig] = field(default_factory=dict)
    config_path: str = ""  # Path to the loaded config file


def _parse_slurm(data: dict) -> SlurmConfig:
    """Parse a [clusters.X.slurm] section."""
    return SlurmConfig(
        account=str(data.get("account", "")),
        partition=str(data.get("partition", "")),
        num_gpus=int(data.get("num_gpus", 1)),
        gpu_type=str(data.get("gpu_type", "")),
        available_gpu_types=data.get("available_gpu_types", []),
        num_cpus=int(data.get("num_cpus", 8)),
        memory=str(data.get("memory", "32G")),
        walltime=str(data.get("walltime", "12:00:00")),
        modules=str(data.get("modules", "")),
        singularity_command=str(data.get("singularity_command", "singularity")),
        output_dir_cluster=str(data.get("output_dir_cluster", "")),
        output_dir_container=str(data.get("output_dir_container", "/results")),
        output_file=str(data.get("output_file", "")),
    )


def _parse_cluster(name: str, data: dict) -> ClusterConfig:
    """Parse a [clusters.X] section."""
    slurm_data = data.get("slurm", {})
    return ClusterConfig(
        name=name,
        hostname=data.get("hostname", name),
        username=data["username"],
        project_path=data["project_path"],
        scratch_path=data["scratch_path"],
        image_path=data.get("image_path", ""),
        sbatch_template=data.get("sbatch_template", "slurm_partition_template.sbatch"),
        slurm=_parse_slurm(slurm_data),
    )


def load_config(path: str = "xgenius.toml") -> XGeniusConfig:
    """Load and validate an xgenius.toml configuration file.

    Args:
        path: Path to the TOML config file. Defaults to 'xgenius.toml' in cwd.

    Returns:
        Validated XGeniusConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required fields are missing.
    """
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Parse project section
    proj_data = raw.get("project", {})
    project = ProjectConfig(
        name=proj_data.get("name", "my-research"),
        research_goal=proj_data.get("research_goal", "research_goal.md"),
        container_image=proj_data.get("container_image", ""),
        dockerfile=proj_data.get("dockerfile", "Dockerfile"),
    )

    # Parse safety section
    safety_data = raw.get("safety", {})
    safety = SafetyConfig(
        max_gpus_per_job=safety_data.get("max_gpus_per_job", 1),
        max_cpus_per_job=safety_data.get("max_cpus_per_job", 16),
        max_memory_per_job=safety_data.get("max_memory_per_job", "64G"),
        max_walltime=safety_data.get("max_walltime", "24:00:00"),
        max_concurrent_jobs=safety_data.get("max_concurrent_jobs", 10),
        max_total_gpu_hours=safety_data.get("max_total_gpu_hours", 500),
        allowed_command_prefixes=safety_data.get("allowed_command_prefixes", ["python"]),
        forbidden_patterns=safety_data.get("forbidden_patterns", SafetyConfig().forbidden_patterns),
        require_singularity=safety_data.get("require_singularity", True),
    )

    # Parse watcher section
    watcher_data = raw.get("watcher", {})
    watcher = WatcherConfig(
        poll_interval_seconds=watcher_data.get("poll_interval_seconds", 60),
        trigger_command=watcher_data.get("trigger_command", "claude --continue"),
    )

    # Parse clusters
    clusters = {}
    for cluster_name, cluster_data in raw.get("clusters", {}).items():
        clusters[cluster_name] = _parse_cluster(cluster_name, cluster_data)

    config = XGeniusConfig(
        project=project,
        safety=safety,
        watcher=watcher,
        clusters=clusters,
        config_path=path,
    )

    _validate_config(config)
    return config


def _validate_config(config: XGeniusConfig) -> None:
    """Validate configuration for required fields and consistency."""
    if not config.clusters:
        return  # Empty clusters is valid during init

    for name, cluster in config.clusters.items():
        if not cluster.username:
            raise ValueError(f"Cluster '{name}' missing required field: username")
        if not cluster.project_path:
            raise ValueError(f"Cluster '{name}' missing required field: project_path")
        if not cluster.scratch_path:
            raise ValueError(f"Cluster '{name}' missing required field: scratch_path")
        if not os.path.isabs(cluster.project_path):
            raise ValueError(f"Cluster '{name}' project_path must be absolute: {cluster.project_path}")
        if not os.path.isabs(cluster.scratch_path):
            raise ValueError(f"Cluster '{name}' scratch_path must be absolute: {cluster.scratch_path}")


def get_project_dir(config: XGeniusConfig) -> str:
    """Get the project directory (where xgenius.toml lives)."""
    return os.path.dirname(config.config_path)


def get_xgenius_dir(config: XGeniusConfig) -> str:
    """Get the .xgenius state directory path."""
    return os.path.join(get_project_dir(config), ".xgenius")


def ensure_xgenius_dir(config: XGeniusConfig) -> str:
    """Create .xgenius directory and all standard files. Returns the path."""
    xgenius_dir = get_xgenius_dir(config)
    os.makedirs(xgenius_dir, exist_ok=True)
    os.makedirs(os.path.join(xgenius_dir, "markers"), exist_ok=True)
    os.makedirs(os.path.join(xgenius_dir, "batches"), exist_ok=True)

    # Create standard files if missing
    for fname in ["journal.md"]:
        fpath = os.path.join(xgenius_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                pass

    # Create DEBUG.md in .xgenius/
    debug_path = os.path.join(xgenius_dir, "DEBUG.md")
    if not os.path.exists(debug_path):
        with open(debug_path, "w") as f:
            f.write("# Debug Log\n\nErrors and issues encountered during autonomous research.\n")

    return xgenius_dir


def parse_memory_string(mem_str: str) -> int:
    """Parse memory string like '64G' into megabytes."""
    mem_str = mem_str.strip().upper()
    if mem_str.endswith("G"):
        return int(float(mem_str[:-1]) * 1024)
    elif mem_str.endswith("M"):
        return int(float(mem_str[:-1]))
    elif mem_str.endswith("T"):
        return int(float(mem_str[:-1]) * 1024 * 1024)
    else:
        return int(mem_str)


def parse_walltime(walltime: str) -> int:
    """Parse walltime string like '24:00:00' into seconds."""
    parts = walltime.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    else:
        return int(parts[0])
