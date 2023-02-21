#!/bin/sh

# SLURM SUBMIT SCRIPT
#SBATCH -a 1
#SBATCH -o ./logs/output_train.%a.out # STDOUT
. /opt/conda/etc/profile.d/conda.sh
conda activate hexEnv
srun python train.py