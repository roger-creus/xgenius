import os
import subprocess
import json
from rich.console import Console

def pull_results(cluster, remote_dir, local_dir):
    console = Console()
    try:
        os.makedirs(local_dir, exist_ok=True)
        remote_path = f"{cluster['username']}@{cluster['cluster_name']}:{remote_dir}"
        result = subprocess.run(
            ["rsync", "-avz", remote_path, local_dir],
            capture_output=True, text=True, check=True
        )
        return f"Results pulled from {remote_path} to {local_dir}"
    except subprocess.CalledProcessError as e:
        return f"Error pulling results from {remote_dir} on {cluster['cluster_name']}: {e}"

def main():
    console = Console()
    cluster_config_file = os.getenv("XGENIUS_CLUSTER_CONFIG", "cluster_config.json")
    
    if not os.path.exists(cluster_config_file):
        console.print(f"[bold red]Cluster configuration file not found: {cluster_config_file}[/bold red]")
        return
    
    with open(cluster_config_file, "r") as f:
        clusters = json.load(f)

    # Prompt for local directory
    local_dir = console.input("[bold yellow]Enter the local directory to save the results: [/bold yellow]")

    console.print("[bold yellow]Pulling results from all clusters...[/bold yellow]")

    for cluster in clusters:
        cluster_name = cluster['cluster_name']
        remote_dir = console.input(f"[bold yellow]Enter the remote directory to pull results from for {cluster_name}: [/bold yellow]")
        if remote_dir:
            status = pull_results(cluster, remote_dir, local_dir)
            console.print(status)
        else:
            console.print(f"[bold red]No remote directory specified for {cluster_name}. Skipping...[/bold red]")

    console.print("[bold green]Results pulling process completed.[/bold green]")

if __name__ == "__main__":
    main()
