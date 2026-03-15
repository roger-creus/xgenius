#!/bin/bash
#SBATCH --account={{ACCOUNT}}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{NUM_CPUS}}
#SBATCH --gres=gpu:{{GPU_TYPE}}{{NUM_GPUS}}
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}
#SBATCH --output={{LOG_FILE}}
#SBATCH --job-name=xg-{{EXPERIMENT_ID}}

mkdir -p {{OUTPUT_DIR_IN_CLUSTER}}
mkdir -p {{LOG_DIR}}

module load {{MODULES_TO_LOAD}}

# 1. Copy your container on the compute node
rsync -avz {{IMAGE_PATH}} $SLURM_TMPDIR

# 2. Executing your code with Apptainer
{{SINGULARITY_COMMAND}} exec --nv \
        -H {{CODE_DIR_IN_CLUSTER}} \
        --bind {{CODE_DIR_IN_CLUSTER}}:/src \
        --bind {{OUTPUT_DIR_IN_CLUSTER}}:{{OUTPUT_DIR_IN_CONTAINER}} \
        --workdir /src \
        $SLURM_TMPDIR/{{IMAGE_NAME}} \
        bash -c "cd /src && {{COMMAND}}"
