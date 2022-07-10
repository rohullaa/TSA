#!/bin/bash

#SBATCH --job-name=Nor
#SBATCH --account=nn9851k
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks-per-node=12


# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-nlptools/2021.01-gomkl-2019b-Python-3.7.4
module load nlpl-pytorch/1.7.1-gomkl-2019b-cuda-10.1.243-Python-3.7.4
module load nlpl-transformers/4.14.1-gomkl-2019b-Python-3.7.4


python3 run_NorBert.py