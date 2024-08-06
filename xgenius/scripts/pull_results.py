import os
import subprocess
import json
import argparse
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
        console.print(f"[bold red]Failed to pull results from {remote_dir} on {cluster['cluster_name']}: {e}[/bold red]")

def main():
    parser = argparse.ArgumentParser(description="Pull results from remote clusters.")
    parser.add_argument('--cluster_config', type=str, default="cluster_config.json", 
                        help="Path to the cluster configuration file.")
    parser.add_argument('--run_config', type=str, default="run_config.json", 
                        help="Path to the run configuration file.")
    parser.add_argument('--local_dir', type=str, default=".", 
                        help="Local directory to save the results.")

    args = parser.parse_args()

    console = Console()

    if not os.path.exists(args.cluster_config):
        console.print(f"[bold red]Cluster configuration file not found: {args.cluster_config}[/bold red]")
        return
    
    if not os.path.exists(args.run_config):
        console.print(f"[bold red]Run configuration file not found: {args.run_config}[/bold red]")
        return
    
    with open(args.cluster_config, "r") as f:
        clusters = json.load(f)

    with open(args.run_config, "r") as f:
        run_config = json.load(f)

    console.print("[bold yellow]Pulling results from all clusters...[/bold yellow]")

    for cluster in clusters:
        cluster_name = cluster['cluster_name']
        remote_dir = run_config.get(cluster_name, {}).get("OUTPUT_DIR_IN_CLUSTER")
        
        if remote_dir:
            status = pull_results(cluster, remote_dir, args.local_dir)
            console.print(status)
        else:
            console.print(f"[bold red]No remote directory found for {cluster_name}. Skipping...[/bold red]")

    console.print("[bold green]Results pulling process completed.[/bold green]")

if __name__ == "__main__":
    main()
