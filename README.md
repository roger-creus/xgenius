# xgenius 🚀

`xgenius` is a command-line tool for managing remote jobs and containerized experiments across multiple clusters. It simplifies the process of building Docker images, converting them to Singularity format, and submitting jobs to clusters using SLURM.

## Pre-requisites 🛠️

- You have a working Dockerfile.
- Singularity installed on your local machine.
- Docker installed on your local machine.
- `docker login` works.

## Local Set-up 🧩

### (Optional) Build Singularity Container from Dockerfile 🐳

```bash
xgenius-build-image --dockerfile=/path/to/Dockerfile --name=<output_image_name> --tag=<tag> --registry=<your_docker_username>
```
where `--dockerfile` is the ABSOLUTE path to your Dockerfile.

This command will build a Docker image, push it to your Docker registry, and then pull it to your local machine as a Singularity image. The Singularity image will be saved in the current directory under the name `<output_image_name>.sif` (the `.sif` extension will be added automatically).

### Define Environment Variable 🌍

First, define the environment variable for the path where SLURM template files will be saved:

```bash
export XGENIUS_TEMPLATES_DIR=/path/to/your/templates
```

**Recommendation:** export XGENIUS_TEMPLATES_DIR=<your_project_path>/slurm_templates

**Important:** Add this to your `bashrc` or `~/.zshrc` to make it permanent!

### Set-Up Cluster Configuration 🏗️

Run:

```bash
xgenius-setup-clusters
```

Follow the prompts to configure your cluster settings. You can add as many clusters as you want. Finish by answering 'done' at the end or the config file won’t be saved!

This creates `cluster_config.json` in the current directory.

### Set-Up Run Configuration ⚙️

Pass the `cluster_config.json` file path to the following command to create `run_config.json`:

```bash
xgenius-setup-runs path/to/cluster_config.json
```

This creates `run_config.json` with placeholder values. The placeholder values are created according to the associated SLURM template for each cluster in `cluster_config.json`.

You are now all set up! Let’s run some experiments remotely!

## Running Experiments 🧪

1. Push your Singularity image to the clusters you want:
    ```bash
    xgenius-push-image --config=path/to/cluster_config.json --image=path/to/singularity_image.sif --clusters=cluster1,cluster2,cluster3
    ```

2. Submit your jobs with:
    ```bash
    xgenius-submit-jobs --config=path/to/cluster_config.json --run-config=path/to/run_config.json --cluster=cluster1 --run-command="python test.py" --pull-repos
    ```

    Note: The `--pull-repos` flag is optional. It pulls changes from GitHub repositories before running the jobs. Always include it if your code is in a GitHub repository!

Done! Your jobs are now running on the cluster! 🎉

## Examples 📝

### `cluster_config.json`

```bash
[
    {
        "cluster_name": "cluster1",
        "username": "<your_username>",
        "image_path": "$SCRATCH", # the path where the Singularity image will be saved in the cluster
        "project_path": "/path/to/project/code/in/cluster", # the path where your code is in the cluster. same as CODE_DIR_IN_CLUSTER in run_config.json
        "sbatch_template": "slurm_partition_template.sbatch" # the SLURM template file to use for this cluster. see the templates in the XGENIUS_TEMPLATES_DIR directory
    },
    {
        "cluster_name": "cluster2",
        "username": "<your_username>",
        "image_path": "$SCRATCH", 
        "project_path": "/path/to/project/code/in/cluster", 
        "sbatch_template": "slurm_partition_template.sbatch" 
    }
]
```

### `run_config.json`
```bash
{
    "cluster1": {
        "SINGULARITY_COMMAND": "singularity", # or 'apptainer' depending on the cluster
        "GPUS_PER_TASK": "1",
        "IMAGE_NAME": "<your_singularity_image_name>.sif",
        "PARTITION": "<partition_name>",
        "CODE_DIR_IN_CLUSTER": "/path/to/project/code/in/cluster",
        "OUTPUT_DIR_IN_CONTAINER": "/path/to/output/dir/in/container", # set this to the directory where your code writes output
        "TIME": "23:59:00", # for the time limit of the job
        "MODULES_TO_LOAD": "singularity", # or 'apptainer' depending on the cluster + any other modules
        "MEM_PER_CPU": "12G", # example RAM memory per CPU
        "OUTPUT_DIR_IN_CLUSTER": "$SCRATCH/runs", # your code outputs will be saved here. OUTPUT_DIR_IN_CLUSTER is binded to OUTPUT_DIR_IN_CONTAINER (see the slurm templates)
        "COMMAND": "python test.py", # the code you want to run
        "CPUS_PER_GPU": "12", # example CPUs
        "OUTPUT_FILE": "$SCRATCH/slurm-%j.out" # the logs file of the job
    }
}
```
## Gotchas ⚠️

1. Your Dockerfile must copy the source code of your project to the container. 

    E.g. If the code to run experiments in your project is all under `src/`, then you must have the following line in your Dockerfile:

    COPY ./src /src

    Only then you will be able to run the experiments in the cluster setting:

    xgenius-submit-jobs --config=path/to/cluster_config.json --run-config=path/to/run_config.json --cluster=cluster1 --run-command="python src/test.py" --pull-repos

2. Your project should be a GitHub repository. This is because the `--pull-repos` flag in the submit-jobs command will only work with GitHub repositories. If your project is not a GitHub repository, you will have to manually copy the code to the cluster.