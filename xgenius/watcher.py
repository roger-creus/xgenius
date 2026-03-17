"""Background watcher daemon for xgenius.

Simple and reliable: poll for .done markers, update DB, pull results, trigger Claude.
No clever reconciliation, no process detection — just the basics done right.
"""

import os
import subprocess
import sys
import time

from xgenius.config import load_config, get_xgenius_dir, get_project_dir, ensure_xgenius_dir
from xgenius.jobs import JobManager


def run_watcher(config_path: str = "xgenius.toml", verbose: bool = False) -> None:
    """Run the background watcher daemon.

    Every cycle:
    1. Update DB from squeue (sync running/pending states)
    2. Check for .done markers (completions)
    3. Pull results for completed jobs
    4. If any new completions → trigger Claude with fresh session
    5. Sleep and repeat

    Args:
        config_path: Path to xgenius.toml.
        verbose: Print status messages.
    """
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]

    config = load_config(config_path)
    xgenius_dir = ensure_xgenius_dir(config)
    project_dir = get_project_dir(config)
    job_manager = JobManager(config)
    db = job_manager.db

    poll_interval = config.watcher.poll_interval_seconds
    trigger_cmd = config.watcher.trigger_command
    log_path = os.path.join(xgenius_dir, "watcher.log")
    lock_path = os.path.join(xgenius_dir, "watcher.lock")

    def _log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        if verbose:
            print(f"xgenius watch: {msg}")

    def _is_locked() -> bool:
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

    _log(f"Started. Polling every {poll_interval}s.")
    _log(f"Trigger command: {trigger_cmd}")
    _log(f"Watching clusters: {list(config.clusters.keys())}")

    while True:
        try:
            if _is_locked():
                _log("Previous trigger still running. Skipping.")
                time.sleep(poll_interval)
                continue

            # Step 1: Update DB from squeue (sync actual states)
            active_ids = db.get_active_job_ids()
            if active_ids:
                seen_in_squeue = set()
                reachable_clusters = set()
                for cluster_name in config.clusters:
                    try:
                        statuses = job_manager.status(cluster_name=cluster_name)
                        reachable_clusters.add(cluster_name)
                        for s in statuses:
                            if s.job_id in active_ids:
                                db.sync_job_state(s.job_id, s.state)
                                seen_in_squeue.add(s.job_id)
                    except Exception as e:
                        _log(f"Failed to query {cluster_name}: {e}")

                # Mark active jobs NOT in squeue as disappeared
                # (only for jobs on clusters we successfully queried)
                if reachable_clusters:
                    for job in db.get_pending_jobs():
                        if job["job_id"] not in seen_in_squeue and job["cluster"] in reachable_clusters:
                            db.mark_disappeared(job["job_id"])

            # Step 2: Check for .done markers
            completions = []
            try:
                completions = job_manager.check_completions()
            except Exception as e:
                _log(f"Failed to check completions: {e}")

            # Step 3: Pull results AND slurm logs for new completions
            # Note: results_pulled is NOT set here — only after Claude succeeds
            for c in completions:
                try:
                    job_manager.pull_results(cluster_name=c.cluster, job_id=c.job_id)
                    _log(f"Completed: {c.experiment_id} (job {c.job_id}, exit={c.exit_code}) — results pulled")
                except Exception as e:
                    _log(f"Completed: {c.experiment_id} (job {c.job_id}, exit={c.exit_code}) — results pull failed: {e}")

                try:
                    job_manager.pull_slurm_logs(c.cluster, c.job_id, c.experiment_id)
                    _log(f"Pulled slurm logs for {c.experiment_id}")
                except Exception as e:
                    _log(f"Failed to pull slurm logs for {c.experiment_id}: {e}")

            # Step 4: Trigger Claude if there's work to do
            # Either new completions from markers, OR completed jobs in DB
            # that haven't been analyzed yet (e.g., from a previous failed trigger)
            needs_trigger = len(completions) > 0

            if not needs_trigger:
                # Check if DB has completed jobs that Claude hasn't processed yet
                # (happens when Claude hit rate limits or failed on previous trigger)
                completed_not_pulled = db.get_completed_not_pulled()
                if completed_not_pulled:
                    needs_trigger = True
                    # Build fake completions list so the prompt shows what needs processing
                    completions = [
                        type('Completion', (), {
                            'job_id': j['job_id'],
                            'experiment_id': j['experiment_id'],
                            'exit_code': j.get('exit_code', -1),
                            'cluster': j['cluster'],
                        })()
                        for j in completed_not_pulled
                    ]
                    _log(f"Retrying: {len(completed_not_pulled)} completed jobs awaiting Claude processing")

            if needs_trigger:
                prompt = db.build_wakeup_prompt(completions=completions if completions else None)
                remaining = len(db.get_active_job_ids())
                _log(f"Triggering Claude: {len(completions)} new completion(s), {remaining} still active")

                trigger_parts = trigger_cmd.split()
                trigger_parts.extend(["-p", prompt])

                with open(lock_path, "w") as f:
                    f.write(str(os.getpid()))
                try:
                    result = subprocess.run(trigger_parts, cwd=project_dir)
                    if result.returncode != 0:
                        _log(f"Claude exited with error (code {result.returncode}). Will retry next cycle — completed jobs remain unprocessed.")
                    else:
                        # Claude succeeded — mark all completed jobs as processed
                        # This prevents retriggering for these jobs next cycle
                        for job in db.get_completed_not_pulled():
                            db.mark_results_pulled(job["job_id"])
                        _log("Claude finished successfully. Marked jobs as processed.")
                finally:
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
            else:
                active = len(db.get_active_job_ids())
                if active > 0:
                    _log(f"{active} active job(s). No completions yet.")
                else:
                    _log("No active jobs. Waiting for new submissions...")

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            _log("Stopped by user.")
            break
        except Exception as e:
            _log(f"Error: {e}")
            time.sleep(poll_interval)
