#!/bin/bash

#SBATCH --job-name=in5550
#SBATCH --account=nn9851k
#SBATCH --mail-type=FAIL
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3G
#
# by default, request eight cores 
#SBATCH --cpus-per-task=8

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /cluster/shared/nlpl/software/eb/etc/all/
module load nlpl-scipy-ecosystem/2021.01-gomkl-2019b-Python-3.7.4
module load nlpl-scikit-bundle/0.22.2.post1-gomkl-2019b-Python-3.7.4
module load nlpl-simple_elmo/0.9.0-gomkl-2019b-Python-3.7.4

# {1} can be /cluster/shared/nlpl/data/vectors/latest/209.zip on Saga

python3 elmo_wsd.py --input bank_wsd.tsv --elmo ${1}

