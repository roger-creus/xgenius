import os
import json
import subprocess
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
    console = Console()
    config_file = os.getenv("XGENIUS_CLUSTER_CONFIG", "cluster_config.json")
    
    if not os.path.exists(config_file):
        console.print(f"[bold red]Cluster configuration file not found: {config_file}[/bold red]")
        return
    
    with open(config_file, "r") as f:
        clusters = json.load(f)

    console.print("[bold yellow]Canceling all jobs on all clusters...[/bold yellow]")

    for cluster in clusters:
        status = cancel_jobs(cluster)
        console.print(status)

    console.print("[bold green]Job cancellation process completed.[/bold green]")

if __name__ == "__main__":
    main()
