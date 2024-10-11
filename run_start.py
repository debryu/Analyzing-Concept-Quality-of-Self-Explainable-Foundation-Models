import os
import subprocess
import sys

result = subprocess.run([sys.executable, 'main.py --dataset shapes3d --model betaglancenet --latent_dim 42 --seed 8 --beta 0 --z_capacity 42'], check=True)
print(result.stdout)
#os.system('python main.py --dataset shapes3d --model betaglancenet --latent_dim 42 --seed 8 --beta 0 --z_capacity 42')
#os.system('python main.py --dataset shapes3d --model betaglancenet --latent_dim 42 --seed 8 --beta -0.5 --z_capacity 42')
#os.system('python main.py --dataset shapes3d --model betaglancenet --latent_dim 42 --seed 8 --beta -1 --z_capacity 42')
#os.system('python main.py --dataset shapes3d --model betaglancenet --latent_dim 42 --seed 8 --beta -2 --z_capacity 42')
#os.system('python main.py --dataset shapes3d --model betaglancenet --latent_dim 42 --seed 8 --beta -5 --z_capacity 42')
#os.system('python main.py --dataset shapes3d --model betaglancenet --latent_dim 42 --seed 8 --beta -10 --z_capacity 42')

