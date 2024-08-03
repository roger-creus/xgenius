#!/bin/bash
#SBATCH --partition={{PARTITION}}
#SBATCH --ntasks=1
#SBATCH --gpus-per-task={{GPUS_PER_TASK}}
#SBATCH --cpus-per-gpu={{CPUS_PER_GPU}}
#SBATCH --time={{TIME}}
#SBATCH --output={{OUTPUT_FILE}}
#SBATCH --mem-per-cpu={{MEM_PER_CPU}}

mkdir -p {{OUTPUT_DIR_IN_CLUSTER}}

module load {{MODULES_TO_LOAD}}

rsync -avz $SCRATCH/{{IMAGE_NAME}} $SLURM_TMPDIR

{{SINGULARITY_COMMAND}} exec --nv \
        -H {{CODE_DIR_IN_CLUSTER}} \
        --bind {{OUTPUT_DIR_IN_CLUSTER}}:{{OUTPUT_DIR_IN_CONTAINER}} \
        --workdir {{CODE_DIR_IN_CLUSTER}} \
        $SLURM_TMPDIR/{{IMAGE_NAME}} \
        bash -c "cd / && {{COMMAND}}"
