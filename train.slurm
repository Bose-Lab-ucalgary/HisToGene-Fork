#!/bin/bash

####### Reserve computing resources #############
#SBATCH --job-name=histogene
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-a100

####### Run your script #########################
source ~/software/init-conda
conda activate histogene-env


python train.py --dataset HEST1K --gene_list 3CA-copy --gpus 2 --strategy ddp_find_unused_parameters_false --num_workers 1 --epochs 20 --precision 16 --force_restart