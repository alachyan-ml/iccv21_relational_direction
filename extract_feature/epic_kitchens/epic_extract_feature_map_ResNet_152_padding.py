# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 2022

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
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models.resnet as models
from PIL import Image
import numpy as np
import scipy.io as sio
import pickle
from global_setting_Pegasus import NFS_path
import pdb
import gzip
import json
from core.helper.preprocessing_func import get_img_tensor_pad
import pandas as pd
from tqdm import tqdm
import argparse
#%%
#import pdb
#%%
idx_GPU = 1
is_save = True
#%%
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
#%%
#pdb.set_trace()

parser = argparse.ArgumentParser()
parser.add_argument('type', type=str, choices=["train", "validation"], default="train")
parser.add_argument('--section', type=int, default=0)
opt = parser.parse_args() 
if (opt.type == "train"):
    assert 0 <= opt.section <= 134
else:
    assert 0 <= opt.section <= 19


# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Batch size for training (change depending on how much memory you have)
batch_size = 32

device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
#%%

model_ref = models.resnet152(pretrained=True)
model_ref.eval()

model_f = nn.Sequential(*list(model_ref.children())[:-2])
model_f.to(device)
model_f.eval()

for param in model_f.parameters():
    param.requires_grad = False

#%%
class CustomedDataset_pad(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, partition, section=0):
        self.partition = partition
        self.anno_file = "data/epic_kitchens/epic-kitchens-100-annotations/EPIC_100_{}.csv".format(partition)
        all_anno_file = "data/epic_kitchens/epic-kitchens-100-annotations/EPIC_100_train.csv"
        all_anno = pd.read_csv(all_anno_file)
        self.anno_values = self.init_values(all_anno)
        self.anno = pd.read_csv(self.anno_file, nrows=500, skiprows=list(range(1,section*500)), header=0)
        self.img_dir = "data/epic_kitchens/epic_images"
        self.anno["total_length"] = self.calculate_total_length()
        self.anno["max_index"] = self.anno["total_length"].cumsum()
        self.anno_values.to_csv("data/epic_kitchens/epic_kitchens_hoi.csv".format(partition), index=False)

    def init_values(self, anno):
        values_df =  pd.DataFrame({"verb_class": anno["verb_class"], "noun_class": anno["noun_class"]})
        dedup_values_df = values_df.drop_duplicates()
        return dedup_values_df.sort_values(by=["verb_class", "noun_class"]).reset_index(drop=True)
 
    def calculate_total_length(self):
        print(self.anno.columns)
        start_frame = self.anno["start_frame"]
        end_frame = self.anno["stop_frame"]
        return end_frame.sub(start_frame)

    def __len__(self):
        return self.anno["max_index"].max()

    def __getitem__(self, idx):
        last_indices = self.anno["max_index"].values
        last_indices_under_idx = np.searchsorted(last_indices, idx, side='left')
        last_index_row = self.anno.iloc[last_indices_under_idx]

        image_path, image_file, video_id = self.get_img_by_idx(last_index_row, idx)
        image = get_img_tensor_pad(image_path)
        label = self.get_value_by_idx(last_index_row)
        return image, label, image_file, video_id

    def get_value_by_idx(self, last_index_row):
        verb = last_index_row["verb_class"]
        noun = last_index_row["noun_class"]
        verb_noun_match = self.anno_values.index[(self.anno_values["verb_class"] == verb) & (self.anno_values["noun_class"] == noun)].tolist()
        verb_noun_index = verb_noun_match[0]
        verb_noun_zero_hot = np.zeros(self.anno_values.shape[0])
        verb_noun_zero_hot[verb_noun_index] = 1
        return verb_noun_zero_hot

    def get_img_by_idx(self, last_idx_row, idx):
        max_index_minus = last_idx_row["max_index"] - idx
        frame = last_idx_row["stop_frame"] - max_index_minus
        frame_file = "frame_{}.jpg".format("0" * (10-len(str(frame))) + str(frame))
        file_path = "data/epic_kitchens/epic_images/{}/{}".format(last_idx_row["video_id"],frame_file)
        return file_path, frame_file, last_idx_row["video_id"]

        
#%%
def sanity_check(dict_a,dict_b):
    assert dict_a.keys() == dict_b.keys()
    for k in dict_a.keys():
        if type(dict_a[k]) == np.ndarray:
            assert (dict_a[k] == dict_b[k]).all()
        else:
            assert dict_a[k] == dict_b[k]
#%%   

print("Extracting features for {} dataset section {}".format(opt.type, opt.section))

dataset = CustomedDataset_pad(opt.type, opt.section)

dataset_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)

with torch.no_grad():
    for i_batch, (imgs,labels,image_files, video_ids) in enumerate(tqdm(dataset_loader)):
        #print(i_batch)
        imgs=imgs.to(device)
        features = model_f(imgs)
        print("Features retrieved for batch {}".format(i_batch))
        for i_f in range(features.size(0)):
            feature = features[i_f].cpu().numpy()
            label = labels[i_f].cpu().numpy()
            
            image_file = image_files[i_f]
            video_id = video_ids[i_f]
            save_file = "{}_{}.pkl.gz".format(video_id, image_file.split(".")[0])
            save_file_path = "data/epic_kitchens/features_pad/{}".format(save_file)

            
            if is_save:
                content = {'feature':feature,'label':label,'image_file':image_file} 
                with gzip.open(save_file_path, 'wb') as f:
                    pickle.dump(content,f)
                
                ### need sanity check ### << load it back and compare the result
                with gzip.open(save_file_path, 'rb') as f:
                    content_load = pickle.load(f)
                sanity_check(content, content_load)
#%%
#val_dataset = CustomedDataset_pad('validation')
#dataset_loader = torch.utils.data.DataLoader(val_dataset,
#                                             batch_size=batch_size, shuffle=False,
#                                             num_workers=4)
#
#with torch.no_grad():
#    for i_batch, (imgs,labels,image_files, video_ids) in enumerate(tqdm(dataset_loader)):
#        #print(i_batch)
#        imgs=imgs.to(device)
#        features = model_f(imgs)
#        print("Features retrieved for batch {}".format(i_batch))
#        for i_f in range(features.size(0)):
#            feature = features[i_f].cpu().numpy()
#            label = labels[i_f].cpu().numpy()
#            
#            image_file = image_files[i_f]
#            video_id = video_ids[i_f]
#            save_file = "{}_{}.pkl.gz".format(video_id, image_file.split(".")[0])
#            save_file_path = "data/epic_kitchens/features_pad/{}".format(save_file)
#
#
#            
#            if is_save:
#                content = {'feature':feature,'label':label,'image_file':image_file} 
#                with gzip.open(save_file_path, 'wb') as f:
#                    pickle.dump(content,f)
#                
#                ### need sanity check ### << load it back and compare the result
#                with gzip.open(save_file_path, 'rb') as f:
#                    content_load = pickle.load(f)
#                sanity_check(content, content_load)
#
