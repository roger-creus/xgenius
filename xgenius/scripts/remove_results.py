import argparse
import subprocess
from xgenius.utils import load_config
from IPython import embed

def get_output_dirs(run_config, selected_clusters):
    if selected_clusters:
        cluster_names = selected_clusters.split(',')
        return {name: run_config[name]["OUTPUT_DIR_IN_CLUSTER"] for name in cluster_names if name in run_config}
    return {name: conf["OUTPUT_DIR_IN_CLUSTER"] for name, conf in run_config.items()}

def remove_results(cluster_config, run_config, selected_clusters=None):
    output_dirs = get_output_dirs(run_config, selected_clusters)

    for cluster_name, output_dir in output_dirs.items():
        try:
            cluster_info = cluster_config[cluster_name]
        except KeyError:
            print(f"Cluster {cluster_name} not found in the configuration.")
        
        username = cluster_info['username']
        print(f"Removing directory {output_dir} on {cluster_name}...")
        try:
            subprocess.run(['ssh', f'{username}@{cluster_name}', f'rm -rf {output_dir}'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove directory {output_dir} on {cluster_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Remove results directories on specified clusters.")
    parser.add_argument('--cluster_config', default="./cluster_config.json", help="Path to the cluster configuration JSON file")
    parser.add_argument('--run_config', default="./run_config.json", help="Path to the run configuration JSON file")
    parser.add_argument('--clusters', help="Comma-separated list of clusters to remove results from")

    args = parser.parse_args()

    cluster_config = load_config(args.cluster_config)
    run_config = load_config(args.run_config)

    remove_results(cluster_config, run_config, selected_clusters=args.clusters)

if __name__ == "__main__":
    main()
