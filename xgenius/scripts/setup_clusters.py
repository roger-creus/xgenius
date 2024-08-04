import json
import os
import shutil
import importlib.resources as pkg_resources
from rich.console import Console
from rich.prompt import Prompt
from rich import print
from xgenius import sbatch_templates

def copy_default_templates():
    from rich.console import Console
    console = Console()

    with pkg_resources.path(sbatch_templates, '') as source_dir:
        dest_dir = os.getenv("XGENIUS_TEMPLATES_DIR", os.path.expanduser("~/.xgenius/sbatch_templates"))
        os.makedirs(dest_dir, exist_ok=True)
        
        for filename in os.listdir(source_dir):
            shutil.copy(os.path.join(source_dir, filename), dest_dir)

        console.print(f"[bold green]Default templates copied to [italic yellow]{dest_dir}[/italic yellow]. You can add or modify templates there.[/bold green]")

def setup_clusters():
    console = Console()
    clusters = []

    console.print("\n[bold blue]Welcome to Cluster Manager[/bold blue]", style="bold underline")
    console.print("Please enter details for each cluster you want to configure.\n")
    console.print("[italic yellow]IMPORTANT:[/italic yellow] The name you enter must match the names in your .ssh/config file.\n")

    done = False
    while not done:
        cluster_name = Prompt.ask("[bold green]Enter cluster name (or 'done' to finish)[/bold green]")
        
        if cluster_name.lower() == 'done':
            print("\n[bold blue]Cluster configuration completed.[/bold blue]")
            done = True
            break
        
        username = Prompt.ask(f"[bold green]Enter username for {cluster_name}[/bold green]")
        image_path = Prompt.ask(f"[bold green]Enter the path where the singularity image will be saved in {cluster_name} (e.g., recommended: absoulte path for cluster scratch)[/bold green]")
        project_path = Prompt.ask(f"[bold green]Enter your ABSOLUTE project path for {cluster_name} (e.g., /home/{cluster_name}/<your-project-path-in-the-cluster>)[/bold green]")

        sbatch_template = Prompt.ask(f"[bold green]Enter sbatch template name for {cluster_name} (default: slurm_partition_template.sbatch)[/bold green]", default="slurm_partition_template.sbatch")
        console.print(f"[italic]Check {os.getenv('XGENIUS_TEMPLATES_DIR', os.path.expanduser('~/.xgenius/sbatch_templates'))} for available templates.[/italic]")
        
        cluster_config = {
            "cluster_name": cluster_name,
            "username": username,
            "image_path": image_path,
            "project_path": project_path,
            "sbatch_template": sbatch_template
        }

        clusters.append(cluster_config)
        console.print("\n[green]Cluster configuration added.[/green]\n")

    config_path = "./cluster_config.json" 
    with open(config_path, 'w') as f:
        json.dump(clusters, f, indent=4)

    console.print(f"\n[bold blue]Cluster configuration saved to [italic]{config_path}[/italic].[/bold blue]")

def main():
    copy_default_templates()
    setup_clusters()

if __name__ == "__main__":
    main()