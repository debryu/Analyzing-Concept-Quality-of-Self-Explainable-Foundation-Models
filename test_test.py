from datasets.utils.celeba_base import check_dataset
from datasets.celeba import CelebA
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import argparse
import pickle
import os
from models.lfcbm import LFcbm, load_cbm
from utils.data_utils import get_data
from tqdm import tqdm
from collect_metrics import create_CS_LKG_dataset, compute_metrics, run_DCI
import argparse
import numpy as np
from sklearn.metrics import classification_report

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

base = "LF_CBM/saved_activations/"

activation_encoder = torch.load(os.path.join(base, "shapes3d_val_clip_RN50.pt"))

activation_image = torch.load(os.path.join(base, "shapes3d_val_clip_ViT-B16.pt"))
activation_image /= torch.norm(activation_image, dim=1, keepdim=True)
activation_image = activation_image.to(device)

activation_text = torch.load(os.path.join(base, "shapes3d_handmade_concepts_ViT-B16.pt"))
activation_text /= torch.norm(activation_text, dim=1, keepdim=True)
activation_text = activation_text.to(device)

P = activation_image @ activation_text.T
index_to_remove = [48,67]
print(P.shape[1])
mask = torch.ones(P.shape[1], dtype=bool)
mask[index_to_remove] = False
P = P[:, mask]
n_concepts = P.shape[1]
n_samples = P.shape[0]
print('Number of concepts:', n_concepts)

dataset_train = get_data("shapes3d_train")
dataset_val = get_data("shapes3d_val")
print(len(dataset_val))

dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)
dl_test = torch.utils.data.DataLoader(dataset_val, batch_size=128, shuffle=False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = load_cbm("LF_CBM/saved_models/shapes3d_cbm_2024_08_01_21_01", device = device)

model_folder = "LF_CBM/saved_models/shapes3d_cbm_2024_08_01_21_01"

loss = torch.nn.CrossEntropyLoss()

all_concepts_predictions = []
all_concepts = []
all_labels = []
all_labels_predictions = []
for i,batch in enumerate(tqdm(dl_test)):
    images, labels, concepts = batch
    batch_size = labels.shape[0]
    images = images.to(device)
    preds, pred_concepts = model(images)
    all_concepts_predictions.append(pred_concepts.detach().to('cpu'))
    all_labels.append(labels.detach().to('cpu')) 
    all_labels_predictions.append(preds.detach().to('cpu'))
    all_concepts.append(concepts.detach().to('cpu'))
    if i == 10:
        break

all_concepts = torch.cat(all_concepts, dim=0)
all_concepts_predictions = torch.cat(all_concepts_predictions, dim=0)
all_labels = torch.cat(all_labels, dim=0)
all_labels_predictions = torch.cat(all_labels_predictions, dim=0)
#print(all_concepts_predictions.shape)
#print(all_labels.shape)
#print(activation_encoder.shape)
concept_loss = np.zeros(all_concepts_predictions.shape[0])

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='shapes3d_lfcbm')
args = parser.parse_args()
# Run the a full accuracy-precision-recall-f1 report
labels_pred_list = all_labels_predictions.argmax(axis = 1).tolist()
labels_gt_list = all_labels.tolist()
concepts_pred_list = all_concepts_predictions.squeeze()
concepts_gt_list = all_concepts.squeeze()
print(concepts_pred_list.shape)
print(concepts_gt_list.shape)
labels_report = classification_report(labels_gt_list, labels_pred_list, target_names = ['label_0', 'label_1'], output_dict=True)
print('Labels f1-score: ', labels_report['macro avg']['f1-score'])

# Run DCI
DCI = run_DCI({'dci_ingredients': (concepts_pred_list,concepts_gt_list)}, mode='fast')
print(DCI)
im = torch.tensor(DCI['importance_matrix'])
im = torch.nn.functional.softmax(im, dim=0)
print(im)
print(torch.min(im,dim = 0))
print(torch.max(im,dim = 0))
DCI_disentanglement = DCI['disentanglement']
asd
metrics_dict = {'DCI':DCI, 
                'DCI_disentanglement': DCI_disentanglement,
                'CE_concepts': None,
                'CE_labels': np.mean(stats_dict['label_loss']),
                'concept_report': None,
                'label_report': labels_report,
                }


create_CS_LKG_dataset({'all_concepts_predictions':all_concepts_predictions, 
                       'all_labels':all_labels, 
                       'all_encoder':activation_encoder,
                       'concept_loss':concept_loss,
                       }, model_folder, args)
