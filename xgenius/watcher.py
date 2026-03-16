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


def _build_trigger_prompt(completions: list, remaining_jobs: int, xgenius_dir: str = "") -> str:
    """Build the prompt to send to Claude when jobs complete.

    Includes a full status breakdown by hypothesis so Claude knows
    exactly which hypotheses are ready for analysis vs still running.
    """
    parts = []

    # List what just completed
    parts.append("## Newly completed experiments:")
    for c in completions:
        status = "SUCCESS" if c.exit_code == 0 else f"FAILED (exit={c.exit_code})"
        parts.append(f"- {c.experiment_id} (job {c.job_id}, {c.cluster}): {status}")

    # Build hypothesis status summary from jobs.jsonl
    if xgenius_dir:
        jobs_path = os.path.join(xgenius_dir, "jobs.jsonl")
        if os.path.exists(jobs_path):
            hyp_status = {}  # {hypothesis_id: {submitted: [], running: [], completed: [], failed: [], cancelled: []}}
            with open(jobs_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    job = json.loads(line)
                    hid = job.get("hypothesis_id", "unknown")
                    status = job.get("status", "unknown")
                    eid = job.get("experiment_id", job.get("job_id", "?"))
                    if hid not in hyp_status:
                        hyp_status[hid] = {"submitted": [], "running": [], "completed": [], "failed": [], "cancelled": []}
                    if status in hyp_status[hid]:
                        hyp_status[hid][status].append(eid)

            if hyp_status:
                parts.append("\n## Hypothesis status summary:")
                for hid in sorted(hyp_status.keys()):
                    s = hyp_status[hid]
                    total = sum(len(v) for v in s.values())
                    done = len(s["completed"])
                    failed = len(s["failed"])
                    running = len(s["running"]) + len(s["submitted"])
                    cancelled = len(s["cancelled"])

                    if running == 0 and total > 0:
                        tag = "READY FOR ANALYSIS" if done > 0 else "ALL CANCELLED"
                    else:
                        tag = f"WAITING ({running} still running)"

                    parts.append(f"- {hid}: {done} done, {failed} failed, {running} running, {cancelled} cancelled — **{tag}**")

    parts.append(f"\n{remaining_jobs} job(s) still running/pending total.")
    parts.append("")
    parts.append("## Your next steps:")
    parts.append("1. Run `xgenius journal context` for full research state")
    parts.append("2. For hypotheses marked READY FOR ANALYSIS: pull results, analyze, record in journal and results bank")
    parts.append("3. For hypotheses still WAITING: do NOT conclude yet, wait for all experiments to finish")
    parts.append("4. If all hypotheses are analyzed: formulate new hypotheses and submit more experiments")

    return "\n".join(parts)


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
                prompt = _build_trigger_prompt(completions, remaining, xgenius_dir=xgenius_dir)
                _log(f"Triggering Claude with {len(completions)} completion(s), {remaining} remaining")

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
