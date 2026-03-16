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

    def _is_claude_active() -> bool:
        """Check if any Claude process is active in this project directory.

        Checks both the watcher lock AND running claude processes whose
        cwd matches our project directory.
        """
        # Check watcher lock first
        if os.path.exists(lock_path):
            try:
                with open(lock_path) as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)
                return True
            except (ValueError, ProcessLookupError, PermissionError):
                os.remove(lock_path)

        # Check for claude processes in our project directory
        try:
            result = subprocess.run(
                ["bash", "-c", f"lsof +D '{project_dir}' 2>/dev/null | grep claude | head -1"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                return True
        except Exception:
            pass

        # Simpler fallback: check /proc for claude processes with our cwd
        try:
            result = subprocess.run(
                ["bash", "-c", f"ls -la /proc/*/cwd 2>/dev/null | grep '{project_dir}' | xargs -I{{}} dirname {{}} | xargs -I{{}} basename {{}} | while read pid; do cat /proc/$pid/cmdline 2>/dev/null | tr '\\0' ' ' | grep -q claude && echo $pid; done"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                return True
        except Exception:
            pass

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
            # Don't trigger if Claude is already active in this project
            if _is_claude_active():
                _log("Claude is active in this project. Skipping this cycle.")
                time.sleep(poll_interval)
                continue

            db = job_manager.db

            # Reconcile DB with actual SLURM state
            recon = job_manager.reconcile()
            reconciled_count = recon.get("reconciled", 0)
            reconciled_ids = recon.get("cancelled_ids", [])
            if reconciled_count > 0:
                _log(f"Reconciled {reconciled_count} stale job(s): {reconciled_ids}")

            # Check how many jobs are still pending (from DB)
            pending = db.get_pending_job_ids()

            # If jobs disappeared, trigger Claude to investigate
            if reconciled_count > 0:
                prompt = db.build_wakeup_prompt(reconciled_ids=reconciled_ids)
                _log(f"Triggering Claude for {reconciled_count} disappeared job(s)")

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
                _log(f"Triggering Claude with {len(completions)} completion(s), {len(db.get_pending_job_ids())} remaining")

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
