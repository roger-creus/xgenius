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

    # Lock file to prevent concurrent triggers
    lock_path = os.path.join(xgenius_dir, "watcher.lock")
    from xgenius.config import get_project_dir
    project_dir = get_project_dir(config)

    def _is_trigger_locked() -> bool:
        """Check if a previous watcher trigger is still running (lock file only)."""
        if not os.path.exists(lock_path):
            return False
        try:
            with open(lock_path) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            os.remove(lock_path)
            return False

    def _acquire_lock():
        with open(lock_path, "w") as f:
            f.write(str(os.getpid()))

    def _release_lock():
        if os.path.exists(lock_path):
            os.remove(lock_path)

    _log(f"Started. Polling every {poll_interval}s.")
    _log(f"Trigger command: {trigger_cmd}")
    _log(f"Watching clusters: {list(config.clusters.keys())}")
    _log(f"Log file: {log_path}")

    while True:
        try:
            # Don't trigger if a previous watcher trigger is still running
            if _is_trigger_locked():
                _log("Previous trigger still running. Skipping this cycle.")
                time.sleep(poll_interval)
                continue

            db = job_manager.db

            # Full state sync: update DB from squeue + .done markers
            recon = job_manager.reconcile()
            synced = recon.get("synced", 0)
            disappeared_ids = recon.get("disappeared_ids", [])
            completed = recon.get("completed", 0)

            if synced > 0:
                _log(f"Synced {synced} job(s): {completed} completed, {len(disappeared_ids)} disappeared, {recon.get('still_active', 0)} still active")

            # Check how many jobs are still active (from DB)
            pending = db.get_active_job_ids()

            # If jobs disappeared, trigger Claude to investigate
            if disappeared_ids:
                prompt = db.build_wakeup_prompt(reconciled_ids=disappeared_ids)
                _log(f"Triggering Claude for {len(disappeared_ids)} disappeared job(s)")

                from xgenius.config import get_project_dir
                project_dir = get_project_dir(config)
                trigger_parts = trigger_cmd.split()
                trigger_parts.extend(["-p", prompt])
                _acquire_lock()
                try:
                    subprocess.run(trigger_parts, cwd=project_dir)
                finally:
                    _release_lock()
                _log("Claude finished processing disappeared jobs")
                continue

            if not pending:
                _log("No pending jobs. Waiting for new submissions...")
                time.sleep(poll_interval)
                continue

            _log(f"{len(pending)} pending job(s). Checking for completions...")

            # Check for completions (.done markers)
            completions = job_manager.check_completions()

            if completions:
                for c in completions:
                    _log(f"Job {c.job_id} ({c.experiment_id}) completed (exit={c.exit_code})")

                # Pull results for ALL completed jobs and mark in DB
                for c in completions:
                    try:
                        job_manager.pull_results(
                            cluster_name=c.cluster,
                            job_id=c.job_id,
                        )
                        db.mark_results_pulled(c.job_id)
                        _log(f"Pulled results for {c.experiment_id}")
                    except Exception as e:
                        _log(f"Failed to pull results for {c.experiment_id}: {e}")

                # Also pull results for any previously completed but unpulled jobs
                unpulled = db.get_completed_not_pulled()
                for job in unpulled:
                    if job["job_id"] not in {c.job_id for c in completions}:
                        try:
                            job_manager.pull_results(cluster_name=job["cluster"], job_id=job["job_id"])
                            db.mark_results_pulled(job["job_id"])
                            _log(f"Pulled missed results for {job['experiment_id']}")
                        except Exception as e:
                            _log(f"Failed to pull missed results for {job['experiment_id']}: {e}")

                # Build prompt from DB (full status overview) and trigger Claude
                prompt = db.build_wakeup_prompt(completions=completions)
                _log(f"Triggering Claude with {len(completions)} completion(s), {len(db.get_active_job_ids())} remaining")

                from xgenius.config import get_project_dir
                project_dir = get_project_dir(config)

                trigger_parts = trigger_cmd.split()
                trigger_parts.extend(["-p", prompt])

                _log("Waiting for Claude to finish processing...")
                _acquire_lock()
                try:
                    subprocess.run(trigger_parts, cwd=project_dir)
                finally:
                    _release_lock()
                _log("Claude finished. Resuming polling.")

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            _log("Stopped by user.")
            break
        except Exception as e:
            _log(f"Error: {e}")
            time.sleep(poll_interval)
