"""Background watcher daemon for xgenius.

Polls clusters for completed jobs and triggers Claude Code to continue
the autonomous research loop.
"""

import json
import os
import subprocess
import sys
import time

from xgenius.config import load_config, get_xgenius_dir, ensure_xgenius_dir
from xgenius.jobs import JobManager


def _load_pending_jobs(xgenius_dir: str) -> set[str]:
    """Load job IDs that are still pending/running."""
    jobs_path = os.path.join(xgenius_dir, "jobs.jsonl")
    if not os.path.exists(jobs_path):
        return set()

    pending = set()
    with open(jobs_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            if job.get("status") in ("submitted", "running"):
                pending.add(job["job_id"])
    return pending


def _build_trigger_prompt(completions: list, remaining_jobs: int) -> str:
    """Build the prompt to send to Claude when jobs complete."""
    parts = []

    for c in completions:
        status = "successfully" if c.exit_code == 0 else f"with exit code {c.exit_code}"
        parts.append(
            f"Experiment {c.experiment_id} (job {c.job_id}) finished {status} "
            f"on {c.cluster}. Output at: {c.output_dir}"
        )

    prompt = "\n".join(parts)

    if remaining_jobs > 0:
        prompt += f"\n\n{remaining_jobs} job(s) still running/pending."
        prompt += "\nRun 'xgenius status --json' to check their status."
    else:
        prompt += "\n\nAll submitted jobs have completed."

    prompt += "\nRun 'xgenius journal context' for full research state."
    prompt += "\nAnalyze the results and decide next steps."

    return prompt


def run_watcher(config_path: str = "xgenius.toml", verbose: bool = False) -> None:
    """Run the background watcher daemon.

    Polls clusters for .done markers. When jobs complete, triggers
    Claude Code to continue the research loop.

    Args:
        config_path: Path to xgenius.toml.
        verbose: Print status messages.
    """
    # Ensure ANTHROPIC_API_KEY doesn't interfere with subscription auth.
    # claude -p uses setup-token / OAuth when no API key is set.
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]
        if verbose:
            print("xgenius watch: Removed ANTHROPIC_API_KEY from environment (using subscription auth)")

    config = load_config(config_path)
    xgenius_dir = ensure_xgenius_dir(config)
    job_manager = JobManager(config)

    poll_interval = config.watcher.poll_interval_seconds
    trigger_cmd = config.watcher.trigger_command
    log_path = os.path.join(xgenius_dir, "watcher.log")

    def _log(msg: str) -> None:
        """Write timestamped message to watcher log and optionally to stdout."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        if verbose:
            print(f"xgenius watch: {msg}")

    def _is_claude_running() -> bool:
        """Check if a Claude Code process is already running in the project directory."""
        from xgenius.config import get_project_dir
        project_dir = get_project_dir(config)
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"claude.*{os.path.basename(project_dir)}"],
                capture_output=True, text=True, timeout=5,
            )
            # Also check for any claude process with our project dir as cwd
            result2 = subprocess.run(
                ["pgrep", "-af", "claude"],
                capture_output=True, text=True, timeout=5,
            )
            # Filter for claude processes (not xgenius watch, not grep itself)
            for line in result2.stdout.strip().splitlines():
                if "claude" in line and "xgenius" not in line and "pgrep" not in line:
                    return True
            return False
        except Exception:
            return False

    _log(f"Started. Polling every {poll_interval}s.")
    _log(f"Trigger command: {trigger_cmd}")
    _log(f"Watching clusters: {list(config.clusters.keys())}")
    _log(f"Log file: {log_path}")

    while True:
        try:
            # Don't do anything if Claude is already running
            if _is_claude_running():
                _log("Claude is already running. Skipping this cycle.")
                time.sleep(poll_interval)
                continue

            # Reconcile local tracker with actual SLURM state
            recon = job_manager.reconcile()
            reconciled_count = recon.get("reconciled", 0)
            reconciled_ids = recon.get("cancelled_ids", [])
            if reconciled_count > 0:
                _log(f"Reconciled {reconciled_count} stale job(s): {reconciled_ids}")

            # Check how many jobs are still pending
            pending = _load_pending_jobs(xgenius_dir)

            # If jobs disappeared from squeue (preempted/killed/timed out),
            # trigger Claude so it can investigate and resubmit
            if reconciled_count > 0:
                remaining = len(pending)
                prompt = (
                    f"{reconciled_count} job(s) disappeared from SLURM (preempted, timed out, or killed): "
                    f"{', '.join(reconciled_ids)}.\n"
                    f"Run 'xgenius errors --job-id ID --cluster CLUSTER --json' to check what happened.\n"
                    f"Run 'xgenius logs --job-id ID --cluster CLUSTER --json' to see output.\n"
                )
                if remaining > 0:
                    prompt += f"{remaining} job(s) still running/pending.\n"
                prompt += "Run 'xgenius journal context' for full research state. Investigate and decide next steps."

                _log(f"Triggering Claude for {reconciled_count} disappeared job(s)")

                from xgenius.config import get_project_dir
                project_dir = get_project_dir(config)
                trigger_parts = trigger_cmd.split()
                trigger_parts.extend(["-p", prompt])
                subprocess.run(trigger_parts, cwd=project_dir)
                _log("Claude finished processing disappeared jobs")
                continue

            if not pending:
                time.sleep(poll_interval)
                continue

            _log(f"{len(pending)} pending job(s). Checking for completions...")

            # Check for completions (.done markers)
            completions = job_manager.check_completions()

            if completions:
                for c in completions:
                    _log(f"Job {c.job_id} ({c.experiment_id}) completed (exit={c.exit_code})")

                # Pull results for completed jobs
                for c in completions:
                    try:
                        job_manager.pull_results(
                            cluster_name=c.cluster,
                            job_id=c.job_id,
                        )
                        _log(f"Pulled results for {c.experiment_id}")
                    except Exception as e:
                        _log(f"Failed to pull results for {c.experiment_id}: {e}")

                # Update pending count
                remaining = len(_load_pending_jobs(xgenius_dir))

                # Build prompt and trigger Claude
                prompt = _build_trigger_prompt(completions, remaining)
                _log(f"Triggering Claude with {len(completions)} completion(s), {remaining} remaining")

                from xgenius.config import get_project_dir
                project_dir = get_project_dir(config)

                trigger_parts = trigger_cmd.split()
                trigger_parts.extend(["-p", prompt])

                _log("Waiting for Claude to finish processing...")
                subprocess.run(trigger_parts, cwd=project_dir)
                _log("Claude finished. Resuming polling.")

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            _log("Stopped by user.")
            break
        except Exception as e:
            _log(f"Error: {e}")
            time.sleep(poll_interval)
