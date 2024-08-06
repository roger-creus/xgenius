import json
import os
import subprocess

def get_local_template_dir():
    return os.getenv("XGENIUS_TEMPLATES_DIR", os.path.expanduser("~/.xgenius/sbatch_templates"))

def load_config(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    if isinstance(data, list):
        config = {item['cluster_name']: item for item in data}
    elif isinstance(data, dict):
        config = data
    else:
        raise ValueError("Unsupported configuration format")
    return config

def get_selected_clusters(cluster_config, selected_clusters):
    if isinstance(cluster_config, dict):
        cluster_config = [{'cluster_name': k, **v} for k, v in cluster_config.items()]

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
    
    # Convert cluster_config if it's a dictionary
    if isinstance(cluster_config, dict):
        cluster_config = [{'cluster_name': k, **v} for k, v in cluster_config.items()]

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
    subprocess.run(['ssh', f'{username}@{cluster_name}', f'cd {project_path} && sbatch submit_job.sh'], check=True)

    os.remove(local_sbatch_script_path)
