#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -t 1-00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=L_CBMs
#SBATCH --output=logs/out.out
#SBATCH --error=logs/err.err
#SBATCH -N 1

python=/home/emanuele.marconato/miniconda3/envs/lcbm/bin/python

$python main.py --dataset shapes3d \
# --model cbm_base --batch_size 64 --lr 0.001 --latent_dim 32 \
# --wandb sml_lab --beta 0.0001 --w_c 2 --w_rec 0.1 --exp_decay 0.95 --n_epochs 200