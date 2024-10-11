#!/bin/bash

# The name of the conda environment
ENV_NAME="cbm"

# The path to the Python script
SCRIPT_PATH="./main.py"

# Activate the conda environment
#source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Run the Python script multiple times
echo "Running scripts"

#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88000 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88001 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88002 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88003 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88004 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version fullsup --w_label 1

#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88010 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88011 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88012 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88013 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version nolabel --w_label 0
#ython $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88014 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version nolabel --w_label 0

#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88020 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88021 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88022 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88023 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88024 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 12sup --w_label 1

#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88030 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88031 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88032 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88033 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88034 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 30sup --w_label 1

#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88000 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88001 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88002 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88003 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88004 --sup_version fullsup --w_label 1

#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88010 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88011 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88012 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88013 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88014 --sup_version nolabel --w_label 0

#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88020 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88021 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88022 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88023 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88024 --sup_version 12sup --w_label 1

#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88030 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88031 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88032 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88033 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88034 --sup_version 30sup --w_label 1
echo "Finished shapes3d models"

#python train_lfcbm.py --dataset shapes3d --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.1 --interpretability_cutoff 0.01
#python train_lfcbm.py --dataset shapes3d --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.28 --interpretability_cutoff 0.85
#python train_lfcbm.py --dataset shapes3d --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.29 --interpretability_cutoff 0.85
#python train_lfcbm.py --dataset shapes3d --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.30 --interpretability_cutoff 0.85
#python train_lfcbm.py --dataset shapes3d --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.31 --interpretability_cutoff 0.85
#python train_lfcbm.py --dataset shapes3d --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.32 --interpretability_cutoff 0.85

#python train_lfcbm.py --dataset celeba --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.1 --interpretability_cutoff 0.01
#python train_lfcbm.py --dataset celeba --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.28 --interpretability_cutoff 0.6
#python train_lfcbm.py --dataset celeba --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.29 --interpretability_cutoff 0.6
#python train_lfcbm.py --dataset celeba --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.30 --interpretability_cutoff 0.6
#python train_lfcbm.py --dataset celeba --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.31 --interpretability_cutoff 0.6
#python train_lfcbm.py --dataset celeba --batch_size 64 --saga_batch_size 64 --clip_cutoff 0.32 --interpretability_cutoff 0.6

echo "Finished lfcmb models"

#python $SCRIPT_PATH --model betaglancenet --dataset celeba --latent_dim 39 --seed 88000 --beta 1 --n_epochs 24 --w_c 1 --z_capacity 10 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset celeba --latent_dim 39 --seed 98000 --beta 8 --n_epochs 50 --w_c 0 --z_capacity 10 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88001 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88002 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88003 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88004 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version fullsup --w_label 1

#python $SCRIPT_PATH --model betaglancenet --dataset celeba --latent_dim 39 --seed 88010 --beta 1 --n_epochs 24 --w_c 1 --z_capacity 10 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model betaglancenet --dataset celeba --latent_dim 39 --seed 98010 --beta 8 --n_epochs 50 --w_c 0 --z_capacity 10 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88011 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88012 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88013 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version nolabel --w_label 0
#ython $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88014 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version nolabel --w_label 0

#python $SCRIPT_PATH --model betaglancenet --dataset celeba --latent_dim 39 --seed 88020 --beta 1 --n_epochs 24 --w_c 1 --z_capacity 10 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88021 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88022 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88023 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88024 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 12sup --w_label 1

#python $SCRIPT_PATH --model betaglancenet --dataset celeba --latent_dim 39 --seed 88030 --beta 1 --n_epochs 24 --w_c 1 --z_capacity 10 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88031 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88032 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88033 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88034 --beta 1 --n_epochs 150 --w_c 1 --z_capacity 10 --sup_version 30sup --w_label 1

#python $SCRIPT_PATH --model cbmbase --dataset celeba --n_epochs 10 --latent_dim 39 --seed 88000 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88001 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88002 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88003 --sup_version fullsup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88004 --sup_version fullsup --w_label 1

#python $SCRIPT_PATH --model cbmbase --dataset celeba --n_epochs 10 --latent_dim 39 --seed 88010 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88011 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88012 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88013 --sup_version nolabel --w_label 0
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88014 --sup_version nolabel --w_label 0

#python $SCRIPT_PATH --model cbmbase --dataset celeba --n_epochs 10 --latent_dim 39 --seed 88020 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88021 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88022 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88023 --sup_version 12sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88024 --sup_version 12sup --w_label 1

#python $SCRIPT_PATH --model cbmbase --dataset celeba --n_epochs 10 --latent_dim 39 --seed 88030 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88031 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88032 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88033 --sup_version 30sup --w_label 1
#python $SCRIPT_PATH --model cbmbase --dataset shapes3d --n_epochs 25 --latent_dim 42 --seed 88034 --sup_version 30sup --w_label 1

echo "Finished CelebA scripts"



python main.py --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88888 --beta 0.1 --n_epochs 60 --batch_size 256 --w_c 1 --z_capacity 0 --sup_version fullsup --w_label 1 --lr 0.0001 --wandb debryu --w_rec 80
python main.py --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88888 --beta 0.1 --n_epochs 60 --batch_size 256 --w_c 1 --z_capacity 0 --sup_version nolabel --w_label 0 --lr 0.0001 --wandb debryu --w_rec 80
python main.py --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88888 --beta 2 --n_epochs 60 --batch_size 256 --w_c 1 --z_capacity 0 --sup_version 12sup --w_label 1 --lr 0.0001 --wandb debryu --w_rec 80
python main.py --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88888 --beta 2 --n_epochs 60 --batch_size 256 --w_c 1 --z_capacity 0 --sup_version 30sup --w_label 1 --lr 0.0001 --wandb debryu --w_rec 80

python main.py --model betaglancenet --dataset celeba --latent_dim 39 --seed 88888 --beta 0.1 --n_epochs 60 --batch_size 64 --w_c 1 --z_capacity 0 --sup_version fullsup --w_label 1 --lr 0.0001 --wandb debryu --w_rec 80
python main.py --model betaglancenet --dataset celeba --latent_dim 39 --seed 88888 --beta 0.1 --n_epochs 60 --batch_size 64 --w_c 1 --z_capacity 0 --sup_version nolabel --w_label 0 --lr 0.0001 --wandb debryu --w_rec 80
python main.py --model betaglancenet --dataset celeba --latent_dim 39 --seed 88888 --beta 2 --n_epochs 60 --batch_size 64 --w_c 1 --z_capacity 0 --sup_version 12sup --w_label 1 --lr 0.0001 --wandb debryu --w_rec 80
python main.py --model betaglancenet --dataset celeba --latent_dim 39 --seed 88888 --beta 2 --n_epochs 60 --batch_size 64 --w_c 1 --z_capacity 0 --sup_version 30sup --w_label 1 --lr 0.0001 --wandb debryu --w_rec 80

#python main.py --model cbmbase --dataset celeba --n_epochs 20 --latent_dim 39 --seed 88888 --sup_version fullsup --w_label 1 --wandb debryu
# Deactivate the conda environment (optional)
conda deactivate