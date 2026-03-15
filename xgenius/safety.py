"""Safety validation layer for xgenius.

Enforces human-defined safety limits on all operations. Every CLI command
passes through this validator before executing any remote operation.
This is enforced in Python code — the LLM cannot bypass it.
"""

import json
import os
import re
import time
from dataclasses import dataclass

from xgenius.config import (
    SafetyConfig,
    XGeniusConfig,
    get_xgenius_dir,
    ensure_xgenius_dir,
    parse_memory_string,
    parse_walltime,
)


@dataclass
class ValidationResult:
    """Result of a safety validation check."""
    allowed: bool
    reason: str = ""
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> dict:
        d = {"allowed": self.allowed, "reason": self.reason}
        if self.warnings:
            d["warnings"] = self.warnings
        return d


@dataclass
class BudgetReport:
    """Current compute budget status."""
    gpu_hours_used: float
    gpu_hours_limit: float
    gpu_hours_remaining: float
    active_jobs: int
    max_concurrent_jobs: int
    jobs_slots_remaining: int

    def to_dict(self) -> dict:
        return {
            "gpu_hours_used": round(self.gpu_hours_used, 2),
            "gpu_hours_limit": self.gpu_hours_limit,
            "gpu_hours_remaining": round(self.gpu_hours_remaining, 2),
            "active_jobs": self.active_jobs,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "job_slots_remaining": self.jobs_slots_remaining,
        }


# Characters/patterns that indicate shell injection attempts
SHELL_INJECTION_PATTERNS = [
    r"[;&|`$]",           # Shell operators
    r"\$\(",              # Command substitution
    r">\s*/",             # Redirect to absolute path
    r"<\(",               # Process substitution
    r"\beval\b",          # eval command
    r"\bexec\b",          # exec command
    r"\bsource\b",        # source command
    r"\.\s+/",            # dot-source
]


class SafetyValidator:
    """Validates all operations against human-defined safety limits.

    Safety checks are enforced at the Python level. The LLM interacts
    with xgenius only through CLI commands, and every command calls
    this validator before executing.
    """

    def __init__(self, config: XGeniusConfig):
        self.config = config
        self.safety = config.safety
        self._xgenius_dir = get_xgenius_dir(config)

    def validate_job_submission(
        self,
        num_gpus: int,
        num_cpus: int,
        memory: str,
        walltime: str,
        gpu_type: str = "",
        cluster_name: str = "",
    ) -> ValidationResult:
        """Validate job resource requests against safety limits."""
        warnings = []

        # Check GPU type is allowed for this cluster
        if gpu_type and cluster_name and cluster_name in self.config.clusters:
            available = self.config.clusters[cluster_name].slurm.available_gpu_types
            if available and gpu_type not in available:
                return ValidationResult(
                    allowed=False,
                    reason=f"GPU type '{gpu_type}' not available on {cluster_name}. Available: {available}",
                )

        # Check GPU limit
        if num_gpus > self.safety.max_gpus_per_job:
            return ValidationResult(
                allowed=False,
                reason=f"Requested {num_gpus} GPUs exceeds limit of {self.safety.max_gpus_per_job}",
            )

        # Check CPU limit
        if num_cpus > self.safety.max_cpus_per_job:
            return ValidationResult(
                allowed=False,
                reason=f"Requested {num_cpus} CPUs exceeds limit of {self.safety.max_cpus_per_job}",
            )

        # Check memory limit
        req_mem = parse_memory_string(memory)
        max_mem = parse_memory_string(self.safety.max_memory_per_job)
        if req_mem > max_mem:
            return ValidationResult(
                allowed=False,
                reason=f"Requested memory {memory} exceeds limit of {self.safety.max_memory_per_job}",
            )

        # Check walltime limit
        req_time = parse_walltime(walltime)
        max_time = parse_walltime(self.safety.max_walltime)
        if req_time > max_time:
            return ValidationResult(
                allowed=False,
                reason=f"Requested walltime {walltime} exceeds limit of {self.safety.max_walltime}",
            )

        # Check concurrent jobs
        active = self._count_active_jobs()
        if active >= self.safety.max_concurrent_jobs:
            return ValidationResult(
                allowed=False,
                reason=f"Already {active} active jobs, limit is {self.safety.max_concurrent_jobs}",
            )

        # Check GPU-hours budget
        budget = self.get_budget()
        estimated_hours = num_gpus * (req_time / 3600)
        if budget.gpu_hours_remaining < estimated_hours:
            return ValidationResult(
                allowed=False,
                reason=f"Estimated {estimated_hours:.1f} GPU-hours would exceed remaining budget of {budget.gpu_hours_remaining:.1f}",
            )

        # Warn if getting close to limits
        if budget.gpu_hours_remaining < estimated_hours * 3:
            warnings.append(f"GPU-hours budget running low: {budget.gpu_hours_remaining:.1f} remaining")

        return ValidationResult(allowed=True, warnings=warnings)

    def validate_command(self, command: str) -> ValidationResult:
        """Validate a command string against safety rules.

        Checks:
        - Command starts with an allowed prefix
        - No forbidden patterns
        - No shell injection attempts
        """
        command = command.strip()

        if not command:
            return ValidationResult(allowed=False, reason="Empty command")

        # Check allowed prefixes
        prefix_ok = any(
            command.startswith(prefix) for prefix in self.safety.allowed_command_prefixes
        )
        if not prefix_ok:
            return ValidationResult(
                allowed=False,
                reason=f"Command must start with one of: {self.safety.allowed_command_prefixes}. Got: '{command.split()[0]}'",
            )

        # Check forbidden patterns
        command_lower = command.lower()
        for pattern in self.safety.forbidden_patterns:
            if pattern.lower() in command_lower:
                return ValidationResult(
                    allowed=False,
                    reason=f"Command contains forbidden pattern: '{pattern}'",
                )

        # Check shell injection
        for pattern in SHELL_INJECTION_PATTERNS:
            if re.search(pattern, command):
                return ValidationResult(
                    allowed=False,
                    reason=f"Command contains potentially dangerous shell pattern: '{pattern}'",
                )

        return ValidationResult(allowed=True)

    def validate_path(self, path: str, project_dir: str) -> ValidationResult:
        """Validate that a path is within the project directory."""
        abs_path = os.path.abspath(os.path.join(project_dir, path))
        abs_project = os.path.abspath(project_dir)

        if not abs_path.startswith(abs_project + os.sep) and abs_path != abs_project:
            return ValidationResult(
                allowed=False,
                reason=f"Path '{path}' escapes project directory '{project_dir}'",
            )

        return ValidationResult(allowed=True)

    def get_budget(self) -> BudgetReport:
        """Get current compute budget status."""
        gpu_hours_used = self._calculate_gpu_hours_used()
        active_jobs = self._count_active_jobs()
        gpu_hours_remaining = max(0, self.safety.max_total_gpu_hours - gpu_hours_used)
        job_slots = max(0, self.safety.max_concurrent_jobs - active_jobs)

        return BudgetReport(
            gpu_hours_used=gpu_hours_used,
            gpu_hours_limit=self.safety.max_total_gpu_hours,
            gpu_hours_remaining=gpu_hours_remaining,
            active_jobs=active_jobs,
            max_concurrent_jobs=self.safety.max_concurrent_jobs,
            jobs_slots_remaining=job_slots,
        )

    def log_action(self, action: str, details: dict, result: ValidationResult) -> None:
        """Log a validation action to the audit trail."""
        ensure_xgenius_dir(self.config)
        audit_path = os.path.join(self._xgenius_dir, "audit.jsonl")

        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "allowed": result.allowed,
            "reason": result.reason,
            **details,
        }

        with open(audit_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_audit_log(self, limit: int = 50) -> list[dict]:
        """Read the most recent audit log entries."""
        audit_path = os.path.join(self._xgenius_dir, "audit.jsonl")
        if not os.path.exists(audit_path):
            return []

        entries = []
        with open(audit_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        return entries[-limit:]

    def _count_active_jobs(self) -> int:
        """Count jobs with status 'submitted' or 'running' from job tracker."""
        jobs_path = os.path.join(self._xgenius_dir, "jobs.jsonl")
        if not os.path.exists(jobs_path):
            return 0

        active = 0
        with open(jobs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                if job.get("status") in ("submitted", "running"):
                    active += 1
        return active

    def _calculate_gpu_hours_used(self) -> float:
        """Calculate total GPU-hours consumed from completed jobs."""
        jobs_path = os.path.join(self._xgenius_dir, "jobs.jsonl")
        if not os.path.exists(jobs_path):
            return 0.0

        total = 0.0
        with open(jobs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                if job.get("status") == "completed" and "gpu_hours" in job:
                    total += job["gpu_hours"]
        return total
