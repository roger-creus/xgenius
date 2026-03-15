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

    if verbose:
        print(f"xgenius watch: Started. Polling every {poll_interval}s.")
        print(f"xgenius watch: Trigger command: {trigger_cmd}")
        print(f"xgenius watch: Watching clusters: {list(config.clusters.keys())}")

    while True:
        try:
            # Reconcile local tracker with actual SLURM state
            recon = job_manager.reconcile()
            if recon["reconciled"] > 0 and verbose:
                print(f"xgenius watch: Reconciled {recon['reconciled']} stale job(s)")

            # Check how many jobs are still pending
            pending = _load_pending_jobs(xgenius_dir)

            if not pending:
                if verbose:
                    print("xgenius watch: No pending jobs. Waiting for new submissions...")
                time.sleep(poll_interval)
                continue

            if verbose:
                print(f"xgenius watch: {len(pending)} pending job(s). Checking for completions...")

            # Check for completions
            completions = job_manager.check_completions()

            if completions:
                if verbose:
                    for c in completions:
                        print(f"xgenius watch: Job {c.job_id} completed (exit={c.exit_code})")

                # Pull results for completed jobs
                for c in completions:
                    try:
                        job_manager.pull_results(
                            cluster_name=c.cluster,
                            job_id=c.job_id,
                        )
                        if verbose:
                            print(f"xgenius watch: Pulled results for {c.experiment_id}")
                    except Exception as e:
                        if verbose:
                            print(f"xgenius watch: Failed to pull results for {c.experiment_id}: {e}")

                # Update pending count
                remaining = len(_load_pending_jobs(xgenius_dir))

                # Build prompt and trigger Claude
                prompt = _build_trigger_prompt(completions, remaining)

                if verbose:
                    print(f"xgenius watch: Triggering Claude...")
                    print(f"xgenius watch: Prompt: {prompt[:200]}...")

                # Get the project directory for --cwd
                from xgenius.config import get_project_dir
                project_dir = get_project_dir(config)

                # Trigger Claude Code — wait for it to finish before polling again.
                # This prevents race conditions if Claude is still processing
                # when more jobs complete. New completions will be picked up
                # on the next poll cycle after Claude exits.
                trigger_parts = trigger_cmd.split()
                trigger_parts.extend(["-p", prompt])

                if verbose:
                    print("xgenius watch: Waiting for Claude to finish processing...")

                subprocess.run(
                    trigger_parts,
                    cwd=project_dir,
                )

                if verbose:
                    print("xgenius watch: Claude finished. Resuming polling.")

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            if verbose:
                print("\nxgenius watch: Stopped by user.")
            break
        except Exception as e:
            print(f"xgenius watch: Error: {e}", file=sys.stderr)
            time.sleep(poll_interval)
