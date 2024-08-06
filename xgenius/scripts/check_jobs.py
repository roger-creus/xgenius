import os
import json
import subprocess
import argparse
from rich.console import Console
from rich.table import Table
from datetime import datetime

def get_job_status(cluster):
    try:
        result = subprocess.run(
            ["ssh", f"{cluster['username']}@{cluster['cluster_name']}", "squeue -u $USER"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error retrieving job status for {cluster['cluster_name']}: {e}"

def print_job_statuses(statuses):
    console = Console()
    table = Table(title="Cluster Job Statuses")

    table.add_column("Cluster", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")

    for cluster, status in statuses.items():
        table.add_row(cluster, status)

    console.print(table)

def save_job_statuses(statuses, output_file):
    with open(output_file, "w") as f:
        for cluster, status in statuses.items():
            f.write(f"Cluster: {cluster}\n{status}\n\n")

def main():
    parser = argparse.ArgumentParser(description="Retrieve and display job statuses from remote clusters.")
    parser.add_argument('--cluster_config', type=str, default="cluster_config.json", 
                        help="Path to the cluster configuration file.")

    args = parser.parse_args()

    console = Console()

    if not os.path.exists(args.cluster_config):
        console.print(f"[bold red]Cluster configuration file not found: {args.cluster_config}[/bold red]")
        return
    
    with open(args.cluster_config, "r") as f:
        clusters = json.load(f)

    statuses = {}
    for cluster in clusters:
        status = get_job_status(cluster)
        statuses[cluster['cluster_name']] = status

    print_job_statuses(statuses)

