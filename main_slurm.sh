#!/bin/sh
#SBATCH -a 1
#SBATCH -o ./logs/output_main.%a.out # STDOUT
. /opt/conda/etc/profile.d/conda.sh
conda activate hexEnv
srun python main.py