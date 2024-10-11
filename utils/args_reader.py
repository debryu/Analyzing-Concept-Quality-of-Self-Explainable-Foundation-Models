# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:06:36 2024

@author: debryu
"""

import os 
import pickle
import torch

base_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ML/Tirocinio/interpreter/data/ckpt/betaGlanceNet_epoch19_seed28211_beta2.11_2024-10-02_20-54-10/'
ar = 'args.pkl'
args = pickle.load(open(base_folder + ar,'rb'))
print(args)
asd
linear_lkg_ckpt = 'leakage_linear_predictor.statedict'
id_translator = pickle.load(open(base_folder+'explainable_indexes_lkg.dict','rb'))['lkg_to_id']
lkg_ckpt = 'leakage_predictor.statedict'
args = pickle.load(open(base_folder + ar,'rb'))
linear_lkg = torch.load(base_folder+linear_lkg_ckpt)
lkg = torch.load(base_folder+lkg_ckpt)

c_male,c_female = linear_lkg['hidden_fc.0.weight']

difference = c_male - c_female

print(difference)
print(difference[26])
index = difference.argmax()
print(index)
print(id_translator[0])
print(id_translator[25])
print(id_translator[26])

#print(args)