#!/bin/sh
#SBATCH -w lab-aicl-n[3-4]
#SBATCH --time-min=50
#SBATCH -o ./logs/dump.%a.out # STDOUT
. /opt/conda/etc/profile.d/conda.sh
conda activate hexEnv
srun python reserve.py