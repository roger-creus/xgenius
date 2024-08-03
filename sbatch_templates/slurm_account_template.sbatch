#!/bin/bash
#SBATCH --account={{ACCOUNT}}
#SBATCH --ntasks=1
#SBATCH --gpus-per-task={{GPUS_PER_TASK}}
#SBATCH --cpus-per-gpu={{CPUS_PER_GPU}}
#SBATCH --time={{TIME}}
#SBATCH --output={{OUTPUT_FILE}}
#SBATCH --mem-per-cpu={{MEM_PER_CPU}}

mkdir -p {{OUTPUT_DIR_IN_CLUSTER}}

module load {{MODULES_TO_LOAD}}

# 1. Copy your container on the compute node
rsync -avz {{IMAGE_PATH}} $SLURM_TMPDIR

# 2. Executing your code with Apptainer
apptainer exec --nv \
        -H {{CODE_DIR}} \
        --bind {{OUTPUT_DIR_IN_CLUSTER}}:{{OUTPUT_DIR_IN_CONTAINER}} \
        --workdir {{CODE_DIR}} \
        $SLURM_TMPDIR/{{IMAGE_NAME}} \
        bash -c "cd / && {{COMMAND}}"
