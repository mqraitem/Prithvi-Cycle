#!/bin/bash
# Activate your environment

#$ -P ivc-ml
#$ -l gpus=1
#$ -pe omp 6
#$ -j y
#$ -l h_rt=12:00:00
#$ -l gpu_memory=48G

export PATH=/projectnb/ivc-ml/mqraitem/miniconda3/bin:$PATH
source activate geo

# Run your commands
python main_regression.py $args