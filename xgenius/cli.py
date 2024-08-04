import argparse
from xgenius.utils import load_config, push_image, submit_jobs

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
