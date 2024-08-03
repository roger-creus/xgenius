import os
import json
import subprocess
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
    console = Console()
    config_file = os.getenv("XGENIUS_CLUSTER_CONFIG", "cluster_config.json")
    
    if not os.path.exists(config_file):
        console.print(f"[bold red]Cluster configuration file not found: {config_file}[/bold red]")
        return
    
    with open(config_file, "r") as f:
        clusters = json.load(f)

    statuses = {}
    for cluster in clusters:
        status = get_job_status(cluster)
        statuses[cluster['cluster_name']] = status

    print_job_statuses(statuses)

    save_to_file = console.input("Do you want to save the statuses to a file? (y/n): ")
    if save_to_file.lower() == 'y':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = console.input(f"Enter output file name [default: job_statuses_{timestamp}.txt]: ") or f"job_statuses_{timestamp}.txt"
        save_job_statuses(statuses, output_file)
        console.print(f"[bold green]Job statuses saved to {output_file}[/bold green]")

if __name__ == "__main__":
    main()
