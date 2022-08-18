# -*- coding: utf-8 -*-
"""
Created on Sat Aug 1 2022

@author: alachyankar
"""

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import scipy.io as sio
import pickle
from global_setting_Pegasus import NFS_path
import pdb
import gzip
import json
import pandas as pd
from datetime import datetime
from torchvision.ops import roi_align
from tqdm import tqdm
import random
#%%
from core.dataset.Epic_pad_Dataset import Epic_pad_Dataset
from core.model.CrossAttention import CrossAttention
from core.helper.helper_func import get_bbox_features,evaluate_mAP,Logger,evaluate_k, compute_F1
from core.helper.localize_eval import LocEvaluator_HICODet
#%%
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--idx_GPU', type=int, default=0, help='')
parser.add_argument('--save_folder', type=str, default='', help='')
parser.add_argument('--load_model', type=str, default='', help='')
parser.add_argument('--comment', type=str, default='', help='')
parser.add_argument('--trainable_w2v', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--normalize_V', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--lamb', type=float, default=-1)
parser.add_argument('--is_w2v_map', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--loc_k', type=int, default=3, help='')
parser.add_argument('--mll_k_3', type=int, default=3)
parser.add_argument('--mll_k_5', type=int, default=5)
parser.add_argument('--partition', type=str, default='', help='')
opt = parser.parse_args() 
'''
python ./experiments/visual_genome_pad/1A/VG_pad_DAZLE_1A.py --idx_GPU 5 --save_folder 'trainable_w2v_no_normalize' --trainable_w2v True --normalize_V False
'''
#%%
batch_size = 30
epochs = 10
idx_GPU = opt.idx_GPU
save_folder =  opt.save_folder
label_type = "interaction"

comment=opt.comment

is_save = True

print('-'*30)
print('label_type {}'.format(label_type))
print('-'*30)
#%%
with open('./w2v/epic_kitchens_act_obj.pkl','rb') as f:
    content = pickle.load(f)
#%%
partition = opt.partition
train_Dataset = Epic_pad_Dataset("train",content)
val_Dataset = Epic_pad_Dataset('validation',content)

print("Creating Subset")
subset_val = torch.utils.data.Subset(val_Dataset, random.sample(list(range(len(val_Dataset))), len(val_Dataset) // 10))

train_dataloader = torch.utils.data.DataLoader(train_Dataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4)

test_dataloader = torch.utils.data.DataLoader(subset_val,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)
#%%
model = CrossAttention(dim_f=2048,dim_v=300,
                 init_w2v_a=content['actions_w2v'],init_w2v_o=content['objects_w2v'],
                 Z_a=content['Z_a'],Z_o=content['Z_o'],
                 trainable_w2v_a = opt.trainable_w2v,trainable_w2v_o = opt.trainable_w2v, 
                 normalize_V_a = opt.normalize_V, normalize_V_o = opt.normalize_V, normalize_F = True,
                 label_type = label_type, grid_size=train_Dataset.grid_size,
                 lamb = opt.lamb, is_w2v_map = opt.is_w2v_map)

device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
model.to(device)

if opt.load_model != '':
   model.load_state_dict(torch.load(opt.load_model,map_location=device))

#%%
params_to_update = []
params_names = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        params_names.append(name)
        print("\t",name)
#%%
lr = 0.0001
weight_decay = 0.#0.0001
momentum = 0.#0.#
#%%
optimizer  = optim.RMSprop( params_to_update ,lr=lr,weight_decay=weight_decay, momentum=momentum)
#%%
experiment_dir = 'results/{}/EpicKitchens_pad_CrossAttention_time_{}/'.format(save_folder, "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now()))
if is_save:
    os.makedirs(experiment_dir)
    with open(experiment_dir+'config.txt','w') as f:
        f.writelines(str(opt))
mAP_history = []
loss_history = []
for epoch in range(epochs):
    print("Epoch: {}".format(epoch))
    running_loss = 0
    for i_batch, (arr_file_name,arr_feature_map,arr_label) in enumerate(tqdm(train_dataloader)):
#        print("Batch: {}".format(i_batch))
        arr_feature_map = arr_feature_map.to(device)
        
        features=arr_feature_map #[B,K,C] == [brf]
        
#        print('Feature Shape: {}'.format(features.shape))
#        pdb.set_trace()
        
        features = features.to(device)
        labels = arr_label.to(device)
        
        
        model.train()
        optimizer.zero_grad()
        
        out_package=model(features)
        in_package = out_package
        in_package['labels'] = labels
#        print("in_package s:{}, labels: {}".format(in_package['s'].shape, in_package['labels'].shape))
        out_package = model.compute_loss(in_package)
        loss = out_package['loss']
        
        loss.backward()
        optimizer.step()
        
        print("Batch Loss: {}".format(loss.item()))
        running_loss += loss.item()


        if i_batch % 2500 == 0 and i_batch > 0:
            print('Epoch: {}, Batch: {}, Running_Train_Loss: {}'.format(epoch, i_batch, running_loss / i_batch))
            loss_history.append(running_loss / i_batch)
            AP,all_preds,all_labels=evaluate_mAP(test_dataloader,model,device)
            print("mAP: {}".format(np.mean(AP)))
            mAP_history.append(np.mean(AP))
            np.save(experiment_dir+'model_loss.npy', np.array(loss_history))
            np.save(experiment_dir+'model_MAP.npy', np.array(mAP_history))
            torch.save(model.state_dict(), experiment_dir+'model_{}_{}.pt'.format(epoch, i_batch))
#             
#%%
print("training finished")
if is_save:
    torch.save(model.state_dict(), experiment_dir+'model_final.pt')
    np.save(experiment_dir+'model_loss.npy', np.array(loss_history))
    np.save(experiment_dir+'model_MAP.npy', np.array(mAP_history))

