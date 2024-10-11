from datasets import get_dataset
import argparse
from utils.inspect_dataset import plot_concept_examples
import torch
import os
parser = argparse.ArgumentParser(description='Debug', allow_abbrev=False)
parser.add_argument('--dataset', type=str, default='shapes3d', help='',
                    choices=['shapes3d', 'celeba'])

parser.add_argument('--batch_size', type=int, default= 256, help='',
                    choices=['shapes3d', 'celeba'])


args = parser.parse_known_args()[0]
print(args)

ds = get_dataset(args)
train_loader, val_loader, (test_loader, ood_loader) = ds.get_data_loaders()



chkpt = 'betaGlanceNet_epoch19_seed13_beta2.0_2024-09-06_18-16-25'
model_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ML/Tirocinio/interpreter/data/ckpt/' + chkpt

files = os.listdir(model_folder)
model_weights = [f for f in files if f.endswith('.pt')][0]
print(model_weights)
#model = torch.load(model_folder + '/model.pth')
#plot_concept_examples(train_loader, [20,41], args, activated=True)