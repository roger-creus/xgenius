"""Container lifecycle management for xgenius.

Provides step-by-step build/push operations with rich structured output.
Designed to be called by Claude, who handles debugging and iteration.
Each operation returns detailed results so Claude can diagnose failures
and decide next steps.
"""

import os
import subprocess
import time

from xgenius.config import XGeniusConfig, ClusterConfig, get_project_dir
from xgenius.ssh import SSHClient


def _run_step(cmd: list[str], description: str, cwd: str = None, timeout: int = 600) -> dict:
    """Run a single build step and return structured results."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
        duration = time.monotonic() - start
        return {
            "step": description,
            "success": result.returncode == 0,
            "stdout": result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout,  # last 3k chars
            "stderr": result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr,
            "returncode": result.returncode,
            "duration_seconds": round(duration, 1),
        }
    except subprocess.TimeoutExpired:
        return {
            "step": description,
            "success": False,
            "stdout": "",
            "stderr": f"Timed out after {timeout}s",
            "returncode": -1,
            "duration_seconds": timeout,
        }
    except FileNotFoundError:
        return {
            "step": description,
            "success": False,
            "stdout": "",
            "stderr": f"Command not found: {cmd[0]}",
            "returncode": -1,
            "duration_seconds": 0,
        }


class ContainerManager:
    """Step-by-step container operations with structured output for Claude."""

    def __init__(self, config: XGeniusConfig):
        self.config = config
        self.project_dir = get_project_dir(config)

    def docker_build(
        self,
        dockerfile: str = "",
        image_name: str = "",
        tag: str = "latest",
        registry: str = "",
    ) -> dict:
        """Run docker build. Returns structured result for Claude to inspect.

        If it fails, Claude should read the error, fix the Dockerfile, and retry.
        """
        dockerfile = dockerfile or self.config.project.dockerfile
        image_name = image_name or self.config.project.container_image.replace(".sif", "")
        docker_tag = f"{registry}/{image_name}:{tag}" if registry else f"{image_name}:{tag}"

        return _run_step(
            ["docker", "build", "-t", docker_tag, "-f", dockerfile, "."],
            description=f"docker build -t {docker_tag}",
            cwd=self.project_dir,
            timeout=1200,
        )

    def docker_test(
        self,
        image_name: str = "",
        tag: str = "latest",
        registry: str = "",
        test_command: str = "",
    ) -> dict:
        """Run tests inside the Docker container.

        If no test_command is provided, tries `python -m pytest tests/ -x` if
        a tests/ directory exists, otherwise runs a basic Python import check.
        """
        image_name = image_name or self.config.project.container_image.replace(".sif", "")
        docker_tag = f"{registry}/{image_name}:{tag}" if registry else f"{image_name}:{tag}"

        if not test_command:
            tests_dir = os.path.join(self.project_dir, "tests")
            if os.path.isdir(tests_dir):
                test_command = "python -m pytest tests/ -x --timeout=120 -q"
            else:
                test_command = "python -c \"print('Container runs OK')\""

        return _run_step(
            ["docker", "run", "--rm", docker_tag, "bash", "-c", test_command],
            description=f"docker test: {test_command[:80]}",
            cwd=self.project_dir,
            timeout=300,
        )

    def docker_push(
        self,
        image_name: str = "",
        tag: str = "latest",
        registry: str = "",
    ) -> dict:
        """Push Docker image to registry."""
        image_name = image_name or self.config.project.container_image.replace(".sif", "")
        docker_tag = f"{registry}/{image_name}:{tag}" if registry else f"{image_name}:{tag}"

        return _run_step(
            ["docker", "push", docker_tag],
            description=f"docker push {docker_tag}",
            cwd=self.project_dir,
            timeout=600,
        )

    def singularity_build(
        self,
        image_name: str = "",
        tag: str = "latest",
        registry: str = "",
    ) -> dict:
        """Convert Docker image to Singularity .sif file.

        Tries singularity first, falls back to apptainer.
        If registry is set, pulls from registry. Otherwise builds from local docker daemon.
        """
        image_name = image_name or self.config.project.container_image.replace(".sif", "")
        docker_tag = f"{registry}/{image_name}:{tag}" if registry else f"{image_name}:{tag}"
        sif_path = os.path.join(self.project_dir, f"{image_name}.sif")

        # Try singularity first, then apptainer
        for cmd_name in ["singularity", "apptainer"]:
            if registry:
                result = _run_step(
                    [cmd_name, "pull", sif_path, f"docker://{docker_tag}"],
                    description=f"{cmd_name} pull from registry",
                    cwd=self.project_dir,
                    timeout=1200,
                )
            else:
                result = _run_step(
                    [cmd_name, "build", sif_path, f"docker-daemon://{docker_tag}"],
                    description=f"{cmd_name} build from local docker",
                    cwd=self.project_dir,
                    timeout=1200,
                )

            if result["success"]:
                result["image_path"] = sif_path
                result["image_size"] = _get_file_size(sif_path)
                return result

            # If command not found, try the other one
            if "Command not found" in result.get("stderr", ""):
                continue
            else:
                # Command exists but failed — return the error
                return result

        return {
            "step": "singularity/apptainer build",
            "success": False,
            "stdout": "",
            "stderr": "Neither singularity nor apptainer found. Install one of them.",
            "returncode": -1,
            "duration_seconds": 0,
        }

    def build_all(
        self,
        dockerfile: str = "",
        image_name: str = "",
        tag: str = "latest",
        registry: str = "",
        skip_tests: bool = False,
    ) -> dict:
        """Run the full build pipeline: docker build → test → singularity convert.

        Returns results for each step. If any step fails, subsequent steps are skipped
        and Claude should inspect the failure, fix the issue, and retry.
        """
        steps = []

        # Step 1: Docker build
        build_result = self.docker_build(dockerfile, image_name, tag, registry)
        steps.append(build_result)
        if not build_result["success"]:
            return {"success": False, "failed_at": "docker_build", "steps": steps}

        # Step 2: Test (optional)
        if not skip_tests:
            test_result = self.docker_test(image_name, tag, registry)
            steps.append(test_result)
            # Tests failing is informational, don't block the build
            if not test_result["success"]:
                steps[-1]["warning"] = "Tests failed but continuing with build"

        # Step 3: Push to registry (if registry set)
        if registry:
            push_result = self.docker_push(image_name, tag, registry)
            steps.append(push_result)
            if not push_result["success"]:
                return {"success": False, "failed_at": "docker_push", "steps": steps}

        # Step 4: Convert to Singularity
        sing_result = self.singularity_build(image_name, tag, registry)
        steps.append(sing_result)
        if not sing_result["success"]:
            return {"success": False, "failed_at": "singularity_build", "steps": steps}

        return {
            "success": True,
            "image_path": sing_result.get("image_path", ""),
            "image_size": sing_result.get("image_size", ""),
            "steps": steps,
        }

    def push_to_cluster(
        self,
        cluster_name: str,
        image_path: str = "",
    ) -> dict:
        """Push Singularity image to a cluster and verify it works.

        Returns structured results for each step.
        """
        cluster = self.config.clusters.get(cluster_name)
        if not cluster:
            return {"success": False, "error": f"Unknown cluster: {cluster_name}"}

        image_path = image_path or self.config.project.container_image
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.project_dir, image_path)

        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}

        ssh = SSHClient(cluster)
        remote_path = os.path.join(cluster.image_path, os.path.basename(image_path))
        steps = []

        # Step 1: Create remote directory
        result = ssh.run(f"mkdir -p {cluster.image_path}")
        steps.append({
            "step": "create remote directory",
            "success": result.success,
            "detail": result.stderr if not result.success else f"mkdir -p {cluster.image_path}",
        })
        if not result.success:
            return {"success": False, "failed_at": "mkdir", "steps": steps}

        # Step 2: SCP the image
        result = ssh.scp_to(image_path, remote_path, timeout=1800)
        steps.append({
            "step": "scp image to cluster",
            "success": result.success,
            "detail": result.stderr if not result.success else f"Uploaded to {remote_path}",
            "duration_seconds": round(result.duration_seconds, 1),
        })
        if not result.success:
            return {"success": False, "failed_at": "scp", "steps": steps, "cluster": cluster_name}

        # Step 3: Verify file exists and check size
        result = ssh.run(f"ls -lh {remote_path}")
        file_exists = result.success and remote_path.split("/")[-1] in result.stdout
        steps.append({
            "step": "verify file on cluster",
            "success": file_exists,
            "detail": result.stdout.strip() if result.success else result.stderr,
        })

        # Note: we do NOT test-run the container here because automation nodes
        # (robot.*) don't allow apptainer/singularity commands. The container
        # will be tested when the first SBATCH job runs on a compute node.

        return {
            "success": file_exists,
            "cluster": cluster_name,
            "remote_path": remote_path,
            "steps": steps,
        }

    def verify_on_cluster(self, cluster_name: str, image_name: str = "") -> dict:
        """Verify a Singularity image exists on a cluster.

        Note: we only check file existence and size. We cannot test-run
        the container from automation nodes (robot.*) because apptainer/singularity
        is not in the allowed commands. The container runs on compute nodes via SBATCH.
        """
        cluster = self.config.clusters.get(cluster_name)
        if not cluster:
            return {"success": False, "error": f"Unknown cluster: {cluster_name}"}

        image_name = image_name or os.path.basename(self.config.project.container_image)
        ssh = SSHClient(cluster)

        image_path = os.path.join(cluster.image_path, image_name)
        result = ssh.run(f"ls -lh {image_path}")

        if not result.success or image_name not in result.stdout:
            return {"success": False, "cluster": cluster_name, "error": f"Image not found at {image_path}"}

        return {
            "success": True,
            "cluster": cluster_name,
            "image_path": image_path,
            "detail": result.stdout.strip(),
        }


def _get_file_size(path: str) -> str:
    """Get human-readable file size."""
    try:
        size = os.path.getsize(path)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except OSError:
        return "unknown"
