python main.py --dataset shapes3d \
--model cbmbase --lr 0.001 --w_c 1 --c_sup 0 --w_rec 0.01 --beta 0.000001 --batch_size 64 \
--exp_decay 0.9 --n_epochs 100 \
#--wandb reasoning-shortcuts
#
# python main.py --dataset mnist --model betaglancenet --beta 0.15 --posthoc --checkin C:\Users\debryu\Desktop\VS_CODE\HOME\ML\Tirocinio\interpreter\data\ckpt\betaGlanceNet_epoch49.pt


# python main.py --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 2 --beta -2
#python main.py --dataset shapes3d --model betaglancenet --latent_dim 42 --seed 8 --beta -1 --z_capacity 42
#python main.py --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 12 --beta 1 --n_epochs 200

# SEED 14 python main.py --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 14 --beta 1 --n_epochs 200 --w_c 0
# SEED 18 with no label supervision python main.py --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 14 --beta 0 --n_epochs 200 --w_c 1
# SEED 19 label supervision python main.py --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 14 --beta 0 --n_epochs 200 --w_c 1

# SEED 88 label supervision python main.py --model betaglancenet --dataset shapes3d --latent_dim 42 --seed 88 --beta 1 --n_epochs 200 --w_c 1
# SEED 89 nolabel supervision 
# SEED 90 partial supervision
# SEED 91 partial supervision (complementary to seed 90)
# SEED 92 partial supervision (removed red pill)

# SEED 89 beta 0.5 fullsup      # CelebA?
# SEED 90 beta 0.5 nolabel      # CelebA?

# SEED 109 nolabel Z_CAPACITY 42
# SEED 119 nolabel Z_CAPACITY 42 beta 0.5
# SEED 129 nolabel Z_CAPACITY 42 beta 0.0
# SEED 139 nolabel Z_CAPACITY 2 beta 1
# SEED 149 nolabel Z_CAPACITY 2 beta 0.5
# SEED 159 nolabel Z_CAPACITY 2 beta 0.0

#python main.py --model cbmbase --dataset shapes3d --n_epochs 100

#python main.py --model betaglancenet --dataset celeba --latent_dim 39 --seed 13 --beta 1 --n_epochs 10 --w_c 1 --z_capacity 5 --sup_version fullsup --w_label 1 --lr 0.0001