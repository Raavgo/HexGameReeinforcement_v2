#!/bin/sh
#SBATCH -a 1-20
#SBATCH -o ./logs/output_eval.%a.out # STDOUT
. /opt/conda/etc/profile.d/conda.sh
conda activate hexEnv
srun python evaluate.py