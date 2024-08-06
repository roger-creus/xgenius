import os
import json
import argparse
from xgenius.cli import load_config

def main():
    parser = argparse.ArgumentParser(description="Submit a batch of experiments.")
    parser.add_argument('--cluster_config', type=str, default="cluster_config.json", help="Path to the JSON file with cluster configurations.")
    parser.add_argument('--run_config', type=str, default="run_config.json", help="Path to the JSON file with run configurations.")
    parser.add_argument('--batch_file', type=str, required=True, help="Path to the JSON file defining the batch of experiments.")
    parser.add_argument('--pull_repos', action='store_true', help="Pull repositories before submitting jobs.")
    parser.add_argument('--dry_run', action='store_true', help="Print commands without executing them.")

    args = parser.parse_args()

    with open(args.batch_file, 'r') as f:
        batch_config = json.load(f)

    clusters = load_config(args.cluster_config)
    run_configs = load_config(args.run_config)

    for experiment in batch_config.get('experiments', []):
        cluster = clusters.get(experiment['cluster'])
        if not cluster:
            print(f"Cluster {experiment['cluster']} not found in config.")
            continue

        run_config = run_configs.get(experiment['cluster'])
        if not run_config:
            print(f"Run config for cluster {experiment['cluster']} not found.")
            continue

        command = experiment['command']
        cluster_name = experiment['cluster']
        run_command = f"xgenius --cluster_config={args.cluster_config} submit_jobs --run_config={args.run_config} --cluster={cluster_name} --run_command=\"{command}\""
        if args.pull_repos:
            run_command += " --pull_repos"

        if args.dry_run:
            print(f"Dry run: {run_command}")
        else:
            print(f"Submitting job: {run_command}")
            os.system(run_command)

    print("Batch submission complete.")

if __name__ == "__main__":
    main()
