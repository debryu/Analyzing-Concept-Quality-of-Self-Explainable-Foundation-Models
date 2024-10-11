import os
import math
import torch
import numpy as np  
from tqdm import tqdm
from torch.utils.data import DataLoader


class CS_estimator_Dataset(torch.utils.data.Dataset):
    def __init__(self, concepts, labels, encoders, args=None):
        self.args = args
        self.concepts = concepts
        self.labels = labels
        self.args = args
        self.encoders = encoders

    def __getitem__(self, idx):
        concepts = self.concepts[idx]
        labels = self.labels[idx]
        encoders = self.encoders[idx]
        return labels, concepts, encoders

    def __len__(self):
        return len(self.labels)

class Leakage_Dataset(torch.utils.data.Dataset):
    def __init__(self, concepts, labels, args=None):
        self.args = args
        self.concepts = concepts
        self.labels = labels

    def __getitem__(self, idx):
        concepts = self.concepts[idx]
        labels = self.labels[idx]
        return labels, concepts

    def __len__(self):
        return len(self.labels)

class Linear_predictor(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super(Linear_predictor, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.device = device
        self.fc.to(device)

    def forward(self, concept1, concept2):
        #print(concept1.device, concept2.device)
        x = torch.stack((concept1, concept2), dim=1)
        # Convert to LongTensor
        x = x.to(torch.float32).to(self.device)
        #print(x.shape)
        return self.fc(x)

class Linear_predictor_multiconcepts(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super(Linear_predictor_multiconcepts, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.device = device
        self.fc.to(device)

    def forward(self, x):
        # Convert to LongTensor
        x = x.to(torch.float32).to(self.device)
        return self.fc(x)

PM_SUFFIX = {"max":"_max", "avg":""}


def get_dataset_C_Y(args):
    if args.dataset == 'kandinsky':
        return 6, 2
    elif args.dataset == 'shapes3d':
        return 42, 2
    elif args.dataset == 'mnist':
        return 4, 2
    elif args.dataset == 'celeba':
        return 39, 2                # One concept is used as label
    else:
        return NotImplementedError('wrong choice')
    

def one_hot_concepts(concepts):
    I = np.unique(concepts)
    one_hots = []
    diag_matrix = np.eye(len(I))
    for sample in range(len(concepts)):
        for i in range(len(I)):
            if concepts[sample] == I[i]:
                one_hots.append(diag_matrix[i])
    one_hots = np.stack(one_hots)
    return one_hots

def ramping_beta(epoch, args, target = 1, total_steps=25):
    #linear
    if epoch < total_steps:
        args.beta = target*epoch/total_steps
    else:
        args.beta = target
    
    return args


def leakage_function(Y_loss, Y_irrelevant_C_loss, random_Y_loss, correcting_factor=0.1):
    min_loss, lkg_loss, max_loss = Y_loss, Y_irrelevant_C_loss, random_Y_loss*(1-correcting_factor)
    if lkg_loss < min_loss:
        min_loss = lkg_loss
    norm = max_loss - min_loss
    dist = max_loss - lkg_loss
    if dist < 0:
        dist = 0
    print('NORM:',norm)
    print('dist:',norm)

    return dist/norm
       