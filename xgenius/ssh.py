"""Structured SSH/SCP/rsync operations for xgenius.

Wraps subprocess SSH calls with structured return values and error handling.
All remote operations go through this module.
"""

import subprocess
import time
from dataclasses import dataclass

from xgenius.config import ClusterConfig


@dataclass
class SSHResult:
    """Structured result from an SSH operation."""
    stdout: str
    stderr: str
    returncode: int
    cluster: str
    command: str
    duration_seconds: float

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "cluster": self.cluster,
            "command": self.command,
            "duration_seconds": round(self.duration_seconds, 2),
            "success": self.success,
        }


class SSHClient:
    """Structured SSH client for a single cluster.

    Uses subprocess to invoke SSH/SCP commands. No persistent connections.
    """

    def __init__(self, cluster: ClusterConfig):
        self.cluster = cluster
        self.hostname = cluster.hostname
        self.username = cluster.username

    @property
    def _target(self) -> str:
        return f"{self.username}@{self.hostname}"

    def run(self, command: str, timeout: int = 120) -> SSHResult:
        """Execute a command on the remote cluster via SSH.

        Args:
            command: Shell command to execute remotely.
            timeout: Timeout in seconds.

        Returns:
            SSHResult with stdout, stderr, returncode.
        """
        start = time.monotonic()
        try:
            result = subprocess.run(
                ["ssh", self._target, command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.monotonic() - start
            return SSHResult(
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                returncode=result.returncode,
                cluster=self.cluster.name,
                command=command,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return SSHResult(
                stdout="",
                stderr=f"SSH command timed out after {timeout}s",
                returncode=-1,
                cluster=self.cluster.name,
                command=command,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return SSHResult(
                stdout="",
                stderr=str(e),
                returncode=-1,
                cluster=self.cluster.name,
                command=command,
                duration_seconds=duration,
            )

    def scp_to(self, local_path: str, remote_path: str, timeout: int = 300) -> SSHResult:
        """Copy a file from local to remote via SCP."""
        start = time.monotonic()
        remote_full = f"{self._target}:{remote_path}"
        try:
            result = subprocess.run(
                ["scp", local_path, remote_full],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.monotonic() - start
            return SSHResult(
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                returncode=result.returncode,
                cluster=self.cluster.name,
                command=f"scp {local_path} -> {remote_path}",
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return SSHResult(
                stdout="",
                stderr=f"SCP timed out after {timeout}s",
                returncode=-1,
                cluster=self.cluster.name,
                command=f"scp {local_path} -> {remote_path}",
                duration_seconds=duration,
            )

    def scp_from(self, remote_path: str, local_path: str, timeout: int = 300) -> SSHResult:
        """Copy a file from remote to local via SCP."""
        start = time.monotonic()
        remote_full = f"{self._target}:{remote_path}"
        try:
            result = subprocess.run(
                ["scp", remote_full, local_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.monotonic() - start
            return SSHResult(
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                returncode=result.returncode,
                cluster=self.cluster.name,
                command=f"scp {remote_path} -> {local_path}",
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return SSHResult(
                stdout="",
                stderr=f"SCP timed out after {timeout}s",
                returncode=-1,
                cluster=self.cluster.name,
                command=f"scp {remote_path} -> {local_path}",
                duration_seconds=duration,
            )

    def rsync_to(
        self,
        local_path: str,
        remote_path: str,
        excludes: list[str] | None = None,
        timeout: int = 600,
    ) -> SSHResult:
        """Rsync files from local to remote."""
        start = time.monotonic()
        cmd = ["rsync", "-avz"]
        for excl in (excludes or []):
            cmd.append(f"--exclude={excl}")
        # Ensure trailing slash for directory sync
        if not local_path.endswith("/"):
            local_path += "/"
        cmd.extend([local_path, f"{self._target}:{remote_path}"])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            duration = time.monotonic() - start
            return SSHResult(
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                returncode=result.returncode,
                cluster=self.cluster.name,
                command=f"rsync {local_path} -> {remote_path}",
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return SSHResult(
                stdout="",
                stderr=f"rsync timed out after {timeout}s",
                returncode=-1,
                cluster=self.cluster.name,
                command=f"rsync {local_path} -> {remote_path}",
                duration_seconds=duration,
            )

    def rsync_from(
        self,
        remote_path: str,
        local_path: str,
        excludes: list[str] | None = None,
        timeout: int = 600,
    ) -> SSHResult:
        """Rsync files from remote to local."""
        start = time.monotonic()
        cmd = ["rsync", "-avz"]
        for excl in (excludes or []):
            cmd.append(f"--exclude={excl}")
        cmd.extend([f"{self._target}:{remote_path}", local_path])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            duration = time.monotonic() - start
            return SSHResult(
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                returncode=result.returncode,
                cluster=self.cluster.name,
                command=f"rsync {remote_path} -> {local_path}",
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return SSHResult(
                stdout="",
                stderr=f"rsync timed out after {timeout}s",
                returncode=-1,
                cluster=self.cluster.name,
                command=f"rsync {remote_path} -> {local_path}",
                duration_seconds=duration,
            )
