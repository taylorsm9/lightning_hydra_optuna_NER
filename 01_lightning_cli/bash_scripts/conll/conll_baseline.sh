#!/bin/bash

## Begin SLURM Batch Commands
#SBATCH --cpus-per-task=31
#SBATCH --gpus=1
#SBATCH --partition=YOUR_PARTITION
#SBATCH --mem=59G
#SBATCH --time=0-01:00
#SBATCH --mail-user YOUR EMAIL
#SBATCH --mail-type=ALL
#SBATCH --output=conll_baseline.log

python activate venv
srun python main.py fit --config config/conll_2003/conll.yaml

## ** End Of SLURM Batch Commands **
