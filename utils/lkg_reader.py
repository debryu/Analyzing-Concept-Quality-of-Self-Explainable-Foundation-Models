# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:06:36 2024

@author: debryu
"""

import os 
import pickle
import torch

base_folder = '//192.168.8.88/sambashare/Uni/Tesi/old-lfcbm/shapes3d_cbm_2024_09_10_18_03/'
ar = 'args.pkl'
linear_lkg_ckpt = 'leakage_linear_predictor.statedict'
id_translator = pickle.load(open(base_folder+'explainable_indexes_lkg.dict','rb'))['lkg_to_id']
lkg_ckpt = 'leakage_predictor.statedict'
args = pickle.load(open(base_folder + ar,'rb'))
linear_lkg = torch.load(base_folder+linear_lkg_ckpt)
lkg = torch.load(base_folder+lkg_ckpt)

c_male,c_female = linear_lkg['hidden_fc.0.weight']

difference = c_male - c_female

print(difference)
print(difference.max())
#index = difference.argmax().item()
index = torch.nonzero(difference < -1.5)
print(index)

indexes = [id_translator[i.item()] for i in index]
print(indexes)
#print(id_translator[index])


#print(args)