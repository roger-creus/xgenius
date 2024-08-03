

# Pre-requisites
Singularity installed on your local machine
Docker installed on your local machine
docker login works


# Local Set-up

# Define environment variable
First define the environment variable for the path where slurm template files will be saved

export XGENIUS_TEMPLATES_DIR=/path/to/your/templates

Recommendation: export XGENIUS_TEMPLATES_DIR=<your_project_path>/slurm_templates
Important: Add this to your bashrc or  ~/.zshrc !!

# Set-up cluster config

Run:

setup-clusters

And follow the prompts to make your cluster configuration. You can add as many clusters as you want.

# Set-up run config

Pass the cluster_config.json file path to the following command with the path where the run_config.json file will be created. The latter contains placeholder values that you need to fill!

The placeholder values are created according to the associated sbatch template with each of the clusters in cluster_config.json

generate-run-config path/to/cluster_config.json path/to/run_config.json

You are now all set up! Let's run some experiments remotely!

# Running Experiments

1) Push your singularity image to the clusters you want:

cluster-manager --config=path/to/cluster_config.json push-image --image=path/to/singularity_image.sif --clusters=cluster1,cluster2,cluster3

2) Submit your jobs with 

cluster-manager --config=path/to/cluster_config.json submit-jobs --run-config=path/to/run_config.json --cluster=cluster1 --run-command="python test.py" --pull-repos

Note that the --pull-repos flag is optional, and what it does is that if yout code in the cluster is a github repository, it will pull changes before running the jobs. You should always include it!


# Gotchas

The docker