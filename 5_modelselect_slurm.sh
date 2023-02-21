#!/bin/sh
#SBATCH -a 1
#SBATCH -o ./logs/output_select.%a.out # STDOUT
. /opt/conda/etc/profile.d/conda.sh
conda activate hexEnv
srun python model_selection.py