#!/bin/bash

#SBATCH --job-name=8cnn
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=48:00:00

module purge
module load pytorch/python3.6/0.2.0_3
source ../venv1/bin/activate

python 8cnn.py