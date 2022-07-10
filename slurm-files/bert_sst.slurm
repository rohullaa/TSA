#!/bin/bash

#SBATCH --job-name=in5550
#SBATCH --account=nn9851k
#SBATCH --mail-type=FAIL
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=accel

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-transformers/4.14.1-gomkl-2019b-Python-3.7.4

# {1} can be /cluster/shared/nlpl/data/vectors/latest/209.zip on Saga

python3 bert_sst.py
