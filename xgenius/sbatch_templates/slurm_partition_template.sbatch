#!/bin/bash
#SBATCH --partition={{PARTITION}}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{NUM_CPUS}}
#SBATCH --gres=gpu:{{GPU_TYPE}}{{NUM_GPUS}}
#SBATCH --time={{TIME}}
#SBATCH --mem={{MEM}}
#SBATCH --output={{LOG_FILE}}
#SBATCH --job-name={{RUN_ID}}-{{EXPERIMENT_ID}}

mkdir -p {{OUTPUT_DIR_IN_CLUSTER}}
mkdir -p {{LOG_DIR}}

module load {{MODULES_TO_LOAD}}

rsync -avz $SCRATCH/{{IMAGE_NAME}} $SLURM_TMPDIR

{{SINGULARITY_COMMAND}} exec --nv \
        -H {{CODE_DIR_IN_CLUSTER}} \
        --bind {{CODE_DIR_IN_CLUSTER}}:/src \
        --bind {{OUTPUT_DIR_IN_CLUSTER}}:{{OUTPUT_DIR_IN_CONTAINER}} \
        --workdir /src \
        $SLURM_TMPDIR/{{IMAGE_NAME}} \
        bash -c "cd /src && {{COMMAND}}"
