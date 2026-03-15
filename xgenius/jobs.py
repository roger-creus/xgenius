"""Job lifecycle management for xgenius.

Handles submission, tracking, status checking, cancellation, and log retrieval
for SLURM jobs across clusters.
"""

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field

from xgenius.config import XGeniusConfig, ClusterConfig, ensure_xgenius_dir, get_xgenius_dir
from xgenius.safety import SafetyValidator, ValidationResult
from xgenius.ssh import SSHClient
from xgenius.templates import (
    load_template,
    render_template,
    build_params_from_cluster,
)


@dataclass
class SubmitResult:
    """Result of a job submission."""
    success: bool
    job_id: str = ""
    cluster: str = ""
    experiment_id: str = ""
    command: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        d = {"success": self.success, "cluster": self.cluster}
        if self.success:
            d["job_id"] = self.job_id
            d["experiment_id"] = self.experiment_id
            d["command"] = self.command
        else:
            d["error"] = self.error
        return d


@dataclass
class JobStatus:
    """Status of a single SLURM job."""
    job_id: str
    name: str
    state: str  # RUNNING, PENDING, COMPLETED, FAILED, CANCELLED, TIMEOUT
    elapsed: str
    time_limit: str
    nodes: str
    cluster: str
    submit_time: str = ""
    start_time: str = ""
    gpus: str = ""
    cpus: str = ""
    memory: str = ""
    reason: str = ""  # Why pending (e.g., "Priority", "Resources")

    def to_dict(self) -> dict:
        d = {
            "job_id": self.job_id,
            "name": self.name,
            "state": self.state,
            "elapsed": self.elapsed,
            "time_limit": self.time_limit,
            "nodes": self.nodes,
            "cluster": self.cluster,
        }
        if self.submit_time:
            d["submit_time"] = self.submit_time
        if self.start_time:
            d["start_time"] = self.start_time
        if self.gpus:
            d["gpus"] = self.gpus
        if self.cpus:
            d["cpus"] = self.cpus
        if self.memory:
            d["memory"] = self.memory
        if self.reason:
            d["reason"] = self.reason
        return d


@dataclass
class CompletionEvent:
    """A job completion detected from a .done marker."""
    job_id: str
    experiment_id: str
    exit_code: int
    cluster: str
    completed_at: str
    output_dir: str
    walltime_seconds: int = 0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "experiment_id": self.experiment_id,
            "exit_code": self.exit_code,
            "cluster": self.cluster,
            "completed_at": self.completed_at,
            "output_dir": self.output_dir,
            "walltime_seconds": self.walltime_seconds,
        }


class JobManager:
    """Manages the full job lifecycle across SLURM clusters."""

    def __init__(self, config: XGeniusConfig):
        self.config = config
        self.safety = SafetyValidator(config)
        self._ssh_clients: dict[str, SSHClient] = {}
        self._xgenius_dir = get_xgenius_dir(config)

    def _get_ssh(self, cluster_name: str) -> SSHClient:
        """Get or create an SSH client for a cluster."""
        if cluster_name not in self._ssh_clients:
            cluster = self.config.clusters.get(cluster_name)
            if not cluster:
                raise ValueError(f"Unknown cluster: {cluster_name}. Available: {list(self.config.clusters.keys())}")
            self._ssh_clients[cluster_name] = SSHClient(cluster)
        return self._ssh_clients[cluster_name]

    def _get_cluster(self, cluster_name: str) -> ClusterConfig:
        """Get cluster config by name."""
        cluster = self.config.clusters.get(cluster_name)
        if not cluster:
            raise ValueError(f"Unknown cluster: {cluster_name}. Available: {list(self.config.clusters.keys())}")
        return cluster

    def submit(
        self,
        cluster_name: str,
        command: str,
        experiment_id: str = "",
        hypothesis_id: str = "",
        num_gpus: int | None = None,
        gpu_type: str | None = None,
        num_cpus: int | None = None,
        memory: str | None = None,
        walltime: str | None = None,
    ) -> SubmitResult:
        """Submit a job to a SLURM cluster.

        Safety validation is performed before submission.
        Job ID is captured from sbatch output.
        Job is recorded in the tracker.

        Resource overrides let Claude request different resources than the
        cluster defaults. The safety validator ensures they never exceed
        the max limits in xgenius.toml [safety].

        Args:
            cluster_name: Name of the cluster to submit to.
            command: Command to run inside the Singularity container.
            experiment_id: Unique experiment identifier.
            hypothesis_id: Associated hypothesis ID.
            num_gpus: GPU override (must be <= safety max). Defaults to cluster config.
            gpu_type: GPU type override e.g. "h100", "3g.20gb" (must be in allowed_gpu_types). Defaults to cluster config.
            num_cpus: CPU override (must be <= safety max). Defaults to cluster config.
            memory: Memory override e.g. "16G" (must be <= safety max). Defaults to cluster config.
            walltime: Walltime override e.g. "04:00:00" (must be <= safety max). Defaults to cluster config.

        Returns:
            SubmitResult with job ID on success.
        """
        cluster = self._get_cluster(cluster_name)
        ssh = self._get_ssh(cluster_name)
        slurm = cluster.slurm

        # Apply resource overrides (fall back to cluster defaults)
        effective_gpus = num_gpus if num_gpus is not None else slurm.num_gpus
        effective_gpu_type = gpu_type if gpu_type is not None else slurm.gpu_type
        effective_cpus = num_cpus if num_cpus is not None else slurm.num_cpus
        effective_memory = memory or slurm.memory
        effective_walltime = walltime or slurm.walltime

        # Auto-generate experiment_id if not provided
        if not experiment_id:
            experiment_id = f"exp_{int(time.time())}"

        # Safety: validate command
        cmd_result = self.safety.validate_command(command)
        self.safety.log_action("validate_command", {"command": command, "cluster": cluster_name}, cmd_result)
        if not cmd_result.allowed:
            return SubmitResult(
                success=False,
                cluster=cluster_name,
                error=f"Command rejected: {cmd_result.reason}",
            )

        # Safety: validate resource request against max limits
        job_result = self.safety.validate_job_submission(
            num_gpus=effective_gpus,
            num_cpus=effective_cpus,
            memory=effective_memory,
            walltime=effective_walltime,
            gpu_type=effective_gpu_type,
            cluster_name=cluster_name,
        )
        self.safety.log_action(
            "validate_job",
            {"cluster": cluster_name, "gpus": effective_gpus, "gpu_type": effective_gpu_type,
             "cpus": effective_cpus, "memory": effective_memory, "walltime": effective_walltime},
            job_result,
        )
        if not job_result.allowed:
            return SubmitResult(
                success=False,
                cluster=cluster_name,
                error=f"Job rejected: {job_result.reason}",
            )

        # Load and render template
        try:
            template = load_template(cluster.sbatch_template)
        except FileNotFoundError as e:
            return SubmitResult(success=False, cluster=cluster_name, error=str(e))

        # Build params with overrides applied
        params = build_params_from_cluster(cluster, container_image=self.config.project.container_image)
        params["NUM_GPUS"] = str(effective_gpus)
        params["GPU_TYPE"] = f"{effective_gpu_type}:" if effective_gpu_type else ""
        params["NUM_CPUS"] = str(effective_cpus)
        params["MEM"] = effective_memory
        params["TIME"] = effective_walltime
        rendered = render_template(
            template=template,
            params=params,
            command=command,
            cluster=cluster,
            experiment_id=experiment_id,
            inject_epilog=True,
        )

        # Write rendered script to temp file and SCP to cluster
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(rendered)
            local_script = f.name

        try:
            remote_script = os.path.join(cluster.project_path, f"xg_{experiment_id}.sh")

            # Ensure markers directory exists on cluster
            ssh.run(f"mkdir -p {cluster.scratch_path}/.xgenius/markers")

            # Upload script
            scp_result = ssh.scp_to(local_script, remote_script)
            if not scp_result.success:
                return SubmitResult(
                    success=False,
                    cluster=cluster_name,
                    error=f"Failed to upload script: {scp_result.stderr}",
                )

            # Submit via sbatch
            submit_result = ssh.run(f"sbatch --chdir='{cluster.project_path}' {remote_script}")
            if not submit_result.success:
                return SubmitResult(
                    success=False,
                    cluster=cluster_name,
                    error=f"sbatch failed: {submit_result.stderr}",
                )

            # Parse job ID from "Submitted batch job 12345"
            job_id = ""
            match = re.search(r"Submitted batch job (\d+)", submit_result.stdout)
            if match:
                job_id = match.group(1)

            # Clean up remote script
            ssh.run(f"rm -f {remote_script}")

            # Record in job tracker
            self._record_job(
                job_id=job_id,
                cluster=cluster_name,
                experiment_id=experiment_id,
                hypothesis_id=hypothesis_id,
                command=command,
            )

            self.safety.log_action(
                "submit",
                {"cluster": cluster_name, "job_id": job_id, "experiment_id": experiment_id, "command": command},
                ValidationResult(allowed=True),
            )

            return SubmitResult(
                success=True,
                job_id=job_id,
                cluster=cluster_name,
                experiment_id=experiment_id,
                command=command,
            )

        finally:
            os.unlink(local_script)

    def status(self, cluster_name: str | None = None, reconcile: bool = True) -> list[JobStatus]:
        """Check job statuses across clusters.

        Returns rich info including elapsed time, time limit, pending reason,
        and submit time so Claude can make smart resource decisions.
        Auto-reconciles local tracker with squeue state.

        Args:
            cluster_name: If provided, check only this cluster. Otherwise check all.
            reconcile: If True, reconcile local tracker first (default True).

        Returns:
            List of JobStatus for xgenius-managed jobs.
        """
        clusters_to_check = (
            [cluster_name] if cluster_name else list(self.config.clusters.keys())
        )

        all_statuses = []
        for cname in clusters_to_check:
            ssh = self._get_ssh(cname)
            cluster = self._get_cluster(cname)

            # Rich squeue format: job_id, name, state, elapsed, time_limit, nodes,
            # submit_time, start_time, num_gpus, num_cpus, memory, reason
            result = ssh.run(
                f'squeue -u {cluster.username} -o "%.18i|%.30j|%.8T|%.12M|%.12l|%.6D|%.20V|%.20S|%.4b|%.4C|%.10m|%R" --noheader'
            )

            if not result.success:
                continue

            for line in result.stdout.splitlines():
                parts = line.split("|")
                if len(parts) >= 7:
                    all_statuses.append(JobStatus(
                        job_id=parts[0].strip(),
                        name=parts[1].strip(),
                        state=parts[2].strip(),
                        elapsed=parts[3].strip(),
                        time_limit=parts[4].strip(),
                        nodes=parts[5].strip(),
                        cluster=cname,
                        submit_time=parts[6].strip() if len(parts) > 6 else "",
                        start_time=parts[7].strip() if len(parts) > 7 else "",
                        gpus=parts[8].strip() if len(parts) > 8 else "",
                        cpus=parts[9].strip() if len(parts) > 9 else "",
                        memory=parts[10].strip() if len(parts) > 10 else "",
                        reason=parts[11].strip() if len(parts) > 11 else "",
                    ))

        return all_statuses

    def job_history(self, limit: int = 50) -> list[dict]:
        """Get history of all tracked jobs with walltime and resource usage.

        Claude uses this to learn how long jobs take and adjust resource requests.
        """
        jobs_path = os.path.join(self._xgenius_dir, "jobs.jsonl")
        if not os.path.exists(jobs_path):
            return []

        entries = []
        with open(jobs_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        return entries[-limit:]

    def cancel(self, cluster_name: str, job_ids: list[str]) -> dict:
        """Cancel specific jobs on a cluster.

        Never cancels all jobs — always requires explicit job IDs.

        Args:
            cluster_name: Cluster to cancel jobs on.
            job_ids: List of specific job IDs to cancel.

        Returns:
            Dict with results per job ID.
        """
        if not job_ids:
            return {"error": "No job IDs provided. Must specify specific jobs to cancel."}

        ssh = self._get_ssh(cluster_name)
        results = {}

        for job_id in job_ids:
            result = ssh.run(f"scancel {job_id}")
            results[job_id] = {
                "cancelled": result.success,
                "error": result.stderr if not result.success else "",
            }

            if result.success:
                self._update_job_status(job_id, "cancelled")

            self.safety.log_action(
                "cancel",
                {"cluster": cluster_name, "job_id": job_id},
                ValidationResult(allowed=True),
            )

        return results

    def _find_log_path(self, job_id: str) -> str:
        """Look up the log file path from the job tracker."""
        jobs_path = os.path.join(self._xgenius_dir, "jobs.jsonl")
        if not os.path.exists(jobs_path):
            return ""
        with open(jobs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                if job.get("job_id") == job_id and job.get("log_file"):
                    return job["log_file"]
        return ""

    def _find_log_path_by_experiment(self, experiment_id: str) -> tuple[str, str, str]:
        """Look up log file path and cluster by experiment ID. Returns (log_path, job_id, cluster)."""
        jobs_path = os.path.join(self._xgenius_dir, "jobs.jsonl")
        if not os.path.exists(jobs_path):
            return "", "", ""
        with open(jobs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                if job.get("experiment_id") == experiment_id:
                    return job.get("log_file", ""), job.get("job_id", ""), job.get("cluster", "")
        return "", "", ""

    def _find_slurm_log(self, ssh: 'SSHClient', cluster: ClusterConfig, job_id: str, ext: str = "out") -> str:
        """Find a SLURM log file by searching likely paths.

        First checks the job tracker for the known path, then falls back to searching.
        """
        # First: check job tracker
        tracked_path = self._find_log_path(job_id)
        if tracked_path:
            if ext == "err":
                tracked_path = tracked_path.replace(".out", ".err")
            result = ssh.run(f"test -f {tracked_path} && echo EXISTS", timeout=10)
            if "EXISTS" in result.stdout:
                return tracked_path

        # xgenius-managed log directory
        log_dir = os.path.join(cluster.scratch_path, ".xgenius", "logs")
        candidates = [
            f"{log_dir}/*_{job_id}.{ext}",  # experiment_id_jobid.out pattern
        ]

        # Also check common SLURM locations
        for base_dir in [cluster.scratch_path, cluster.project_path]:
            candidates.append(f"{base_dir}/slurm-{job_id}.{ext}")

        for pattern in candidates:
            if "*" in pattern:
                result = ssh.run(f"ls {pattern} 2>/dev/null | head -1", timeout=10)
                if result.success and result.stdout.strip():
                    return result.stdout.strip()
            else:
                result = ssh.run(f"test -f {pattern} && echo EXISTS", timeout=10)
                if "EXISTS" in result.stdout:
                    return pattern

        return ""

    def logs(self, cluster_name: str, job_id: str, lines: int = 200) -> str:
        """Fetch SLURM stdout log for a job.

        Searches multiple likely paths to find the log file.
        """
        cluster = self._get_cluster(cluster_name)
        ssh = self._get_ssh(cluster_name)

        log_path = self._find_slurm_log(ssh, cluster, job_id, "out")
        if not log_path:
            return f"No log file found for job {job_id}. Searched scratch and project directories."

        result = ssh.run(f"tail -n {lines} {log_path}")
        if result.success:
            return f"[{log_path}]\n{result.stdout}"
        return f"Found log at {log_path} but could not read it: {result.stderr}"

    def errors(self, cluster_name: str, job_id: str, lines: int = 200) -> str:
        """Fetch SLURM stderr / crash logs for a job.

        Searches for .err files, then looks for errors in .out files.
        """
        cluster = self._get_cluster(cluster_name)
        ssh = self._get_ssh(cluster_name)

        parts = []

        # Try .err file
        err_path = self._find_slurm_log(ssh, cluster, job_id, "err")
        if err_path:
            result = ssh.run(f"tail -n {lines} {err_path}")
            if result.success and result.stdout.strip():
                parts.append(f"[{err_path}]\n{result.stdout}")

        # Also check .out for errors (tracebacks, etc.)
        out_path = self._find_slurm_log(ssh, cluster, job_id, "out")
        if out_path:
            result = ssh.run(
                f"grep -i -A10 'error\\|traceback\\|exception\\|FAILED\\|FileNotFoundError\\|ModuleNotFoundError\\|ImportError\\|RuntimeError' {out_path} | tail -n {lines}"
            )
            if result.success and result.stdout.strip():
                parts.append(f"[errors in {out_path}]\n{result.stdout}")

        # Check completion marker
        marker_path = f"{cluster.scratch_path}/.xgenius/markers/{job_id}.done"
        result = ssh.run(f"cat {marker_path} 2>/dev/null")
        if result.success and result.stdout.strip():
            parts.append(f"[completion marker]\n{result.stdout}")

        return "\n\n".join(parts) if parts else f"No error logs found for job {job_id}."

    def check_completions(self, cluster_name: str | None = None) -> list[CompletionEvent]:
        """Check for completed jobs by looking for .done marker files.

        Args:
            cluster_name: If provided, check only this cluster.

        Returns:
            List of CompletionEvent for newly completed jobs.
        """
        clusters_to_check = (
            [cluster_name] if cluster_name else list(self.config.clusters.keys())
        )

        completions = []
        for cname in clusters_to_check:
            cluster = self._get_cluster(cname)
            ssh = self._get_ssh(cname)
            marker_dir = f"{cluster.scratch_path}/.xgenius/markers"

            # List .done files
            ls_result = ssh.run(f"ls {marker_dir}/*.done 2>/dev/null")
            if not ls_result.success or not ls_result.stdout.strip():
                continue

            for marker_path in ls_result.stdout.strip().splitlines():
                marker_path = marker_path.strip()
                if not marker_path:
                    continue

                # Read marker content
                cat_result = ssh.run(f"cat {marker_path}")
                if not cat_result.success:
                    continue

                try:
                    data = json.loads(cat_result.stdout)
                    completion = CompletionEvent(
                        job_id=str(data.get("job_id", "")),
                        experiment_id=str(data.get("experiment_id", "")),
                        exit_code=int(data.get("exit_code", -1)),
                        cluster=cname,
                        completed_at=str(data.get("completed_at", "")),
                        output_dir=str(data.get("output_dir", "")),
                        walltime_seconds=int(data.get("walltime_seconds", 0)),
                    )
                    completions.append(completion)

                    # Remove the marker file after reading
                    ssh.run(f"rm -f {marker_path}")

                    # Update local job tracker
                    self._update_job_status(
                        completion.job_id,
                        "completed" if completion.exit_code == 0 else "failed",
                        gpu_hours=self._estimate_gpu_hours(cname, completion.walltime_seconds),
                    )
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        return completions

    def list_remote_files(
        self, cluster_name: str, path: str = "", pattern: str = ""
    ) -> str:
        """List files on a remote cluster.

        Args:
            cluster_name: Cluster to list files on.
            path: Directory path on cluster (defaults to project_path).
            pattern: Optional glob pattern to filter.

        Returns:
            Directory listing as string.
        """
        cluster = self._get_cluster(cluster_name)
        ssh = self._get_ssh(cluster_name)

        target_path = path or cluster.project_path
        if pattern:
            result = ssh.run(f"find {target_path} -name '{pattern}' -maxdepth 3 2>/dev/null | head -100")
        else:
            result = ssh.run(f"ls -la {target_path}")

        return result.stdout if result.success else f"Error: {result.stderr}"

    def pull_results(
        self,
        cluster_name: str,
        job_id: str = "",
        experiment_id: str = "",
        local_dir: str = "",
        excludes: list[str] | None = None,
    ) -> dict:
        """Pull results from a cluster.

        Args:
            cluster_name: Cluster to pull from.
            job_id: Specific job ID to pull results for.
            experiment_id: Specific experiment to pull results for.
            local_dir: Local directory to save results (defaults to ./results/).
            excludes: File patterns to exclude from rsync.

        Returns:
            Dict with pull status.
        """
        cluster = self._get_cluster(cluster_name)
        ssh = self._get_ssh(cluster_name)

        remote_dir = cluster.slurm.output_dir_cluster
        if not remote_dir:
            return {"success": False, "error": "No output_dir_cluster configured for this cluster"}

        if not local_dir:
            from xgenius.config import get_project_dir
            local_dir = os.path.join(get_project_dir(self.config), "results", cluster_name)

        os.makedirs(local_dir, exist_ok=True)

        default_excludes = excludes or ["*.pt", "*.pth", "*.ckpt"]
        result = ssh.rsync_from(
            remote_path=remote_dir + "/",
            local_path=local_dir,
            excludes=default_excludes,
        )

        return {
            "success": result.success,
            "cluster": cluster_name,
            "local_dir": local_dir,
            "remote_dir": remote_dir,
            "error": result.stderr if not result.success else "",
        }

    def sync_code(
        self,
        cluster_name: str,
        excludes: list[str] | None = None,
    ) -> dict:
        """Sync project code to a cluster via rsync.

        Args:
            cluster_name: Cluster to sync to.
            excludes: Additional patterns to exclude.

        Returns:
            Dict with sync status.
        """
        cluster = self._get_cluster(cluster_name)
        ssh = self._get_ssh(cluster_name)

        from xgenius.config import get_project_dir
        local_dir = get_project_dir(self.config)

        default_excludes = [
            ".git", "__pycache__", "*.pyc", ".xgenius",
            "results", "*.sif", "node_modules", ".venv",
            "wandb",
        ]
        all_excludes = default_excludes + (excludes or [])

        result = ssh.rsync_to(
            local_path=local_dir,
            remote_path=cluster.project_path,
            excludes=all_excludes,
        )

        return {
            "success": result.success,
            "cluster": cluster_name,
            "local_dir": local_dir,
            "remote_dir": cluster.project_path,
            "error": result.stderr if not result.success else "",
        }

    def reconcile(self) -> dict:
        """Reconcile local job tracker with actual SLURM state.

        For every job marked as 'submitted' or 'running' locally, check
        if it still exists in squeue. If not, and no .done marker exists,
        mark it as 'cancelled'. If a .done marker exists, process the completion.

        This handles: external cancellation (scancel), job timeouts,
        jobs that finished but markers weren't detected.

        Returns:
            Dict with reconciliation results.
        """
        jobs_path = os.path.join(self._xgenius_dir, "jobs.jsonl")
        if not os.path.exists(jobs_path):
            return {"reconciled": 0, "still_active": 0}

        # Load locally pending jobs grouped by cluster
        pending_by_cluster: dict[str, set[str]] = {}
        with open(jobs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                if job.get("status") in ("submitted", "running"):
                    cluster = job.get("cluster", "")
                    if cluster not in pending_by_cluster:
                        pending_by_cluster[cluster] = set()
                    pending_by_cluster[cluster].add(job["job_id"])

        if not pending_by_cluster:
            return {"reconciled": 0, "still_active": 0}

        # Check actual squeue state per cluster
        active_ids: set[str] = set()
        for cluster_name in pending_by_cluster:
            try:
                statuses = self.status(cluster_name=cluster_name)
                active_ids.update(s.job_id for s in statuses)
            except Exception:
                # If we can't reach the cluster, don't mark jobs as cancelled
                active_ids.update(pending_by_cluster[cluster_name])

        # Also check for .done markers (completed but not yet detected)
        try:
            completions = self.check_completions()
            completed_ids = {c.job_id for c in completions}
        except Exception:
            completed_ids = set()

        # Reconcile: mark stale jobs
        all_pending = set()
        for ids in pending_by_cluster.values():
            all_pending.update(ids)

        stale = all_pending - active_ids - completed_ids
        for job_id in stale:
            self._update_job_status(job_id, "cancelled")

        # Update running jobs
        for job_id in all_pending & active_ids:
            self._update_job_status(job_id, "running")

        still_active = len(all_pending) - len(stale) - len(completed_ids)

        return {
            "reconciled": len(stale),
            "completed_detected": len(completed_ids),
            "still_active": max(0, still_active),
            "cancelled_ids": sorted(stale) if stale else [],
        }

    def _record_job(
        self,
        job_id: str,
        cluster: str,
        experiment_id: str,
        hypothesis_id: str,
        command: str,
    ) -> None:
        """Record a submitted job in the tracker."""
        ensure_xgenius_dir(self.config)
        jobs_path = os.path.join(self._xgenius_dir, "jobs.jsonl")

        # Compute the log file path (matches what the SBATCH template uses)
        cluster_config = self._get_cluster(cluster)
        log_dir = os.path.join(cluster_config.scratch_path, ".xgenius", "logs")
        log_file = os.path.join(log_dir, f"{experiment_id}_{job_id}.out")

        entry = {
            "job_id": job_id,
            "cluster": cluster,
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "command": command,
            "status": "submitted",
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu_hours": 0,
            "log_file": log_file,
        }

        with open(jobs_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _update_job_status(self, job_id: str, new_status: str, gpu_hours: float = 0) -> None:
        """Update job status in tracker by rewriting the file."""
        jobs_path = os.path.join(self._xgenius_dir, "jobs.jsonl")
        if not os.path.exists(jobs_path):
            return

        lines = []
        with open(jobs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                if job.get("job_id") == job_id:
                    job["status"] = new_status
                    if gpu_hours:
                        job["gpu_hours"] = gpu_hours
                lines.append(json.dumps(job))

        with open(jobs_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

    def _estimate_gpu_hours(self, cluster_name: str, walltime_seconds: int) -> float:
        """Estimate GPU-hours for a completed job."""
        cluster = self._get_cluster(cluster_name)
        return cluster.slurm.num_gpus * (walltime_seconds / 3600)
