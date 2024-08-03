#!/bin/bash
#SBATCH --account={{ACCOUNT}}
#SBATCH --cpus-per-task={{NUM_CPUS}}
#SBATCH --gres=gpu:{{NUM_GPUS}}
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}
#SBATCH --output={{OUTPUT_FILE}}

mkdir -p {{OUTPUT_DIR_IN_CLUSTER}}

module load {{MODULES_TO_LOAD}}

# 1. Copy your container on the compute node
rsync -avz {{IMAGE_PATH}} $SLURM_TMPDIR

# 2. Executing your code with Apptainer
{{SINGULARITY_COMMAND}} exec --nv \
        -H {{CODE_DIR_IN_CLUSTER}} \
        --bind {{OUTPUT_DIR_IN_CLUSTER}}:{{OUTPUT_DIR_IN_CONTAINER}} \
        --workdir {{CODE_DIR_IN_CLUSTER}} \
        $SLURM_TMPDIR/{{IMAGE_NAME}} \
        bash -c "cd / && {{COMMAND}}"
