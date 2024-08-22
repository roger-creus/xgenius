import os
import json
import subprocess
import argparse
from rich.console import Console

def pull_repos(cluster):
    console = Console()
    try:
        command = f"cd {cluster['project_path']} && git pull"
        result = subprocess.run(
            ["ssh", f"{cluster['username']}@{cluster['cluster_name']}", command],
            capture_output=True, text=True, check=True
        )
        return f"Pulled latest code on {cluster['cluster_name']} at {cluster['project_path']}"
    except subprocess.CalledProcessError as e:
        return f"Error pulling latest code on {cluster['cluster_name']}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Pull latest code from git repositories on all clusters.")
    parser.add_argument('--config', default="cluster_config.json", help="Path to the cluster configuration JSON file (default: cluster_config.json)")
    args = parser.parse_args()

    console = Console()
    config_file = args.config

    if not os.path.exists(config_file):
        console.print(f"[bold red]Cluster configuration file not found: {config_file}[/bold red]")
        return
    
    with open(config_file, "r") as f:
        clusters = json.load(f)

    console.print("[bold yellow]Pulling latest code from repositories on all clusters...[/bold yellow]")

    for cluster in clusters:
        status = pull_repos(cluster)
        console.print(status)

    console.print("[bold green]Repositories updated on all clusters.[/bold green]")

if __name__ == "__main__":
    main()