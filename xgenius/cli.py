import argparse
import json
import subprocess
import os

def get_local_template_dir():
    return os.getenv("XGENIUS_TEMPLATES_DIR", os.path.expanduser("~/.xgenius/sbatch_templates"))

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_selected_clusters(cluster_config, selected_clusters):
    if selected_clusters:
        cluster_names = selected_clusters.split(',')
        return [c for c in cluster_config if c['cluster_name'] in cluster_names]
    return cluster_config

def pull_repos(cluster_config, selected_clusters=None):
    clusters = get_selected_clusters(cluster_config, selected_clusters)
    for cluster in clusters:
        username = cluster['username']
        project_path = cluster['project_path']
        cluster_name = cluster['cluster_name']
        print(f"Pulling latest code on {cluster_name}...")
        subprocess.run(['ssh', f'{username}@{cluster_name}', f'cd {project_path} && git pull'], check=True)

def push_image(cluster_config, image_path, selected_clusters=None):
    clusters = get_selected_clusters(cluster_config, selected_clusters)
    for cluster in clusters:
        username = cluster['username']
        image_path_remote = os.path.join(cluster['image_path'], os.path.basename(image_path))
        cluster_name = cluster['cluster_name']
        print(f"Pushing image to {cluster_name}...")
        subprocess.run(['scp', image_path, f'{username}@{cluster_name}:{image_path_remote}'], check=True)

def submit_jobs(cluster_config, run_config, cluster_name, command, pull_repos_before_submit=False):
    if pull_repos_before_submit:
        pull_repos(cluster_config, selected_clusters=cluster_name)
    
    cluster = next((c for c in cluster_config if c['cluster_name'] == cluster_name), None)
    if not cluster:
        print(f"Cluster {cluster_name} not found in the configuration.")
        return

    run_params = run_config.get(cluster_name, {})
    sbatch_template_path = os.path.join(get_local_template_dir(), cluster['sbatch_template'])

    if not os.path.exists(sbatch_template_path):
        print(f"SBATCH template {sbatch_template_path} not found.")
        return

    with open(sbatch_template_path, 'r') as file:
        sbatch_template = file.read()

    for key, value in run_params.items():
        sbatch_template = sbatch_template.replace(f'{{{{{key}}}}}', str(value))
    
    sbatch_template = sbatch_template.replace('{{COMMAND}}', command)

    local_sbatch_script_path = os.path.join(os.getcwd(), 'submit_job.sh')
    with open(local_sbatch_script_path, 'w') as file:
        file.write(sbatch_template)

    username = cluster['username']
    project_path = cluster['project_path']
    remote_sbatch_script_path = os.path.join(project_path, 'submit_job.sh')

    print(f"Uploading sbatch script to {cluster_name}...")
    subprocess.run(['scp', local_sbatch_script_path, f'{username}@{cluster_name}:{remote_sbatch_script_path}'], check=True)

    print(f"Submitting job to {cluster_name}...")
    subprocess.run(['ssh', f'{username}@{cluster_name}', f'sbatch {remote_sbatch_script_path}'], check=True)

    os.remove(local_sbatch_script_path)

def main():
    parser = argparse.ArgumentParser(description="Cluster Manager CLI")
    parser.add_argument('--config', required=True, help="Path to the cluster configuration JSON file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_parser = subparsers.add_parser('push-image', help="Push Singularity image to clusters")
    push_parser.add_argument('--image', required=True, help="Path to the local Singularity image file")
    push_parser.add_argument('--clusters', help="Comma-separated list of clusters to push the image to")

    submit_parser = subparsers.add_parser('submit-jobs', help="Submit jobs to the clusters")
    submit_parser.add_argument('--run-config', required=True, help="Path to the run configuration JSON file")
    submit_parser.add_argument('--cluster', required=True, help="Name of the cluster to submit the job to")
    submit_parser.add_argument('--run-command', required=True, help="Command to run inside the container")
    submit_parser.add_argument('--pull-repos', action='store_true', help="Pull the latest code from git repositories before submitting jobs")

    args = parser.parse_args()

    cluster_config = load_config(args.config)

    if args.command == 'push-image':
        push_image(cluster_config, args.image, selected_clusters=args.clusters)
    elif args.command == 'submit-jobs':
        run_config = load_config(args.run_config)
        submit_jobs(cluster_config, run_config, args.cluster, args.run_command, pull_repos_before_submit=args.pull_repos)

if __name__ == "__main__":
    main()
