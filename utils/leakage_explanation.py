# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:06:36 2024

@author: debryu
"""

import os 
import pickle
import torch

base_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ML/Tirocinio/interpreter/data/ckpt/betaGlanceNet_epoch59_seed18211_beta0.1_2024-10-02_15-48-26/'
ar = 'args.pkl'
args = pickle.load(open(base_folder + ar,'rb'))
print(args)

linear_lkg_ckpt = 'leakage_linear_predictor.statedict'


id_translator = pickle.load(open(base_folder+'explainable_indexes_lkg.dict','rb'))['lkg_to_id']
lkg_ckpt = 'leakage_predictor.statedict'
args = pickle.load(open(base_folder + ar,'rb'))
linear_lkg = torch.load(base_folder+linear_lkg_ckpt)
lkg = torch.load(base_folder+lkg_ckpt)

print(linear_lkg)
print(lkg)

pred_0_weights,pred_1_weights = linear_lkg['hidden_fc.0.weight']

difference = pred_0_weights - pred_1_weights

print(difference)

index = difference.argmax().item()
print(index)
print(id_translator[index])
