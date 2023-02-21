#!/bin/sh
#SBATCH -a 1-20
#SBATCH -o ./logs/output_generator.%a.out # STDOUT
. /opt/conda/etc/profile.d/conda.sh
conda activate hexEnv
srun python generate_train_data.py

