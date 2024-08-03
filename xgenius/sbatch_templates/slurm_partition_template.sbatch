#!/bin/bash
#SBATCH --partition={{PARTITION}}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{NUM_CPUS}}
#SBATCH --gres=gpu:{{NUM_GPUS}}
#SBATCH --time={{TIME}}
#SBATCH --mem={{MEM}}
#SBATCH --output={{OUTPUT_FILE}}

mkdir -p {{OUTPUT_DIR_IN_CLUSTER}}

module load {{MODULES_TO_LOAD}}

rsync -avz $SCRATCH/{{IMAGE_NAME}} $SLURM_TMPDIR

{{SINGULARITY_COMMAND}} exec --nv \
        -H {{CODE_DIR_IN_CLUSTER}} \
        --bind {{OUTPUT_DIR_IN_CLUSTER}}:{{OUTPUT_DIR_IN_CONTAINER}} \
        --workdir {{CODE_DIR_IN_CLUSTER}} \
        $SLURM_TMPDIR/{{IMAGE_NAME}} \
        bash -c "cd / && {{COMMAND}}"
