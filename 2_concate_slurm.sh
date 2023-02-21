#!/bin/sh
#SBATCH -a 1
#SBATCH -o ./logs/dump.out # STDOUT
. /opt/conda/etc/profile.d/conda.sh
conda activate hexEnv
srun python concate_arrays.py