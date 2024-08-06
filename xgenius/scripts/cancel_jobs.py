import os
import json
import subprocess
import argparse
from rich.console import Console

def cancel_jobs(cluster):
    console = Console()
    try:
        result = subprocess.run(
            ["ssh", f"{cluster['username']}@{cluster['cluster_name']}", "scancel -u $USER"],
            capture_output=True, text=True, check=True
        )
        return f"All jobs canceled on {cluster['cluster_name']}"
    except subprocess.CalledProcessError as e:
        return f"Error canceling jobs on {cluster['cluster_name']}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Cancel all jobs for the user on specified clusters.")
    parser.add_argument('--cluster_config', type=str, default="cluster_config.json", 
                        help="Path to the cluster configuration file.")

    args = parser.parse_args()

    console = Console()

    if not os.path.exists(args.cluster_config):
        console.print(f"[bold red]Cluster configuration file not found: {args.cluster_config}[/bold red]")
        return
    
    with open(args.cluster_config, "r") as f:
        clusters = json.load(f)

    console.print("[bold yellow]Canceling all jobs on all clusters...[/bold yellow]")

    for cluster in clusters:
        status = cancel_jobs(cluster)
        console.print(status)

    console.print("[bold green]Job cancellation process completed.[/bold green]")

if __name__ == "__main__":
    main()
