import os
import json
import subprocess
import argparse
from rich.console import Console

def load_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)

def get_excludes_from_gitignore(project_path):
    gitignore_path = os.path.join(project_path, ".gitignore")
    if not os.path.exists(gitignore_path):
        return []
    
    with open(gitignore_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def sync_repo(cluster, exclude_patterns, local_dir):
    console = Console()
    remote_dir = f"{cluster['username']}@{cluster['cluster_name']}:{cluster['project_path']}"
    
    # Constructing rsync command
    rsync_command = ["rsync", "-avz", "--delete"]
    
    # Add excludes for hidden files and folders
    rsync_command.extend(["--exclude", ".*"])
    
    # Add custom excludes from gitignore or provided by user
    for pattern in exclude_patterns:
        rsync_command.extend(["--exclude", pattern])
    
    # Add source and destination
    rsync_command.extend([local_dir, remote_dir])
    
    try:
        console.print(f"[bold yellow]Syncing {local_dir} to {remote_dir}...[/bold yellow]")
        subprocess.run(rsync_command, check=True)
        return f"Successfully synced {local_dir} to {remote_dir}"
    except subprocess.CalledProcessError as e:
        return f"Error syncing {local_dir} to {remote_dir}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Sync code directories to remote clusters using rsync.")
    parser.add_argument("--config", default="cluster_config.json", help="Path to the cluster configuration JSON file.")
    parser.add_argument("--exclude", nargs="*", help="Additional exclude patterns to add to rsync.")
    parser.add_argument("--local_dir", default="./", help="Local path to the project directory.")
    
    
    args = parser.parse_args()
    
    # Load cluster configuration
    config_file = os.getenv("XGENIUS_CLUSTER_CONFIG", args.config)
    
    console = Console()
    
    if not os.path.exists(config_file):
        console.print(f"[bold red]Cluster configuration file not found: {config_file}[/bold red]")
        return
    
    clusters = load_config(config_file)
    
    console.print("[bold yellow]Starting sync process for all clusters...[/bold yellow]")
    
    for cluster in clusters:
        # Gather exclude patterns
        exclude_patterns = args.exclude or []
        gitignore_excludes = get_excludes_from_gitignore(args.local_dir)
        exclude_patterns.extend(gitignore_excludes)
        
        # Sync the repository
        status = sync_repo(cluster, exclude_patterns, args.local_dir)
        console.print(status)
    
    console.print("[bold green]Sync process completed.[/bold green]")

if __name__ == "__main__":
    main()
