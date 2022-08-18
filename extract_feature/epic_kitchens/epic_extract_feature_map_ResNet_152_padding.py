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
""" python extract_feature/epic_kitchens/epic_extract_feature_map_ResNet_152_padding_batched.py train --size 30 --gpu 1 --split 6 --part 0 """ 

parser = argparse.ArgumentParser()
parser.add_argument('type', type=str, choices=["train", "validation"], default="train")
parser.add_argument('--size', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--part', type=int, default=0)
opt = parser.parse_args() 


# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Batch size for training (change depending on how much memory you have)
batch_size = 32

device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
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

    def __init__(self, partition, size, split, part):
        self.partition = partition
        self.img_dir = "data/epic_kitchens/epic_images"

        all_anno_file = "data/epic_kitchens/epic-kitchens-100-annotations/EPIC_100_train.csv"
        all_anno = pd.read_csv(all_anno_file)
        self.anno_values = self.init_values(all_anno)
        self.anno_values.to_csv("data/epic_kitchens/epic_kitchens_hoi.csv".format(partition), index=False)
 
        self.anno_file = "data/epic_kitchens/epic-kitchens-100-annotations/EPIC_100_{}.csv".format(partition)
        self.anno = pd.read_csv(self.anno_file)
        self.anno["verb_noun_value"] = self.get_verb_noun_values()
        self.anno = self.anno[self.anno["verb_noun_value"] >= 0]
        if split > 0:
            split_value = len(self.anno) // split
            self.anno = self.anno.iloc[part*split_value: (part+1)*split_value] if part + 1 > split else self.anno.iloc[part*split_value:]
        self.anno["selected_frames"] = self.select_frames(size)
        self.anno["train_length"] = self.anno["selected_frames"].map(len)
        self.anno["max_index"] = self.anno["train_length"].cumsum()
        self.anno["last_index"] = self.anno["max_index"] - 1
        self.anno["start_index"] = np.insert(self.anno["max_index"].values, 0, 0)[:len(self.anno["max_index"])]
       
    def init_values(self, anno):
        values_df =  pd.DataFrame({"verb_class": anno["verb_class"], "noun_class": anno["noun_class"]})
        dedup_values_df = values_df.drop_duplicates()
        return dedup_values_df.sort_values(by=["verb_class", "noun_class"]).reset_index(drop=True)
 
    def get_verb_noun_values(self):
        values = []
        for index, row in self.anno.iterrows():
            indices = self.anno_values.index[(self.anno_values["verb_class"] == row["verb_class"]) & (self.anno_values["noun_class"] == row["noun_class"])].to_list()
            if (len(indices) > 0):
               values.append(indices[0])
            else:
               values.append(-1)
        return np.array(values)

    def select_frames(self, split):
        return [np.array(list(range(start, end+1, split))) for start, end in zip(self.anno["start_frame"], self.anno["stop_frame"])]

    def __len__(self):
        return self.anno["train_length"].sum()

    def __getitem__(self, idx):
        last_indices = self.anno["last_index"].values
        last_indices_under_idx = np.searchsorted(last_indices, idx, side='left')
        last_index_row = self.anno.iloc[last_indices_under_idx]

        image_path, image_file, video_id = self.get_img_by_idx(last_index_row, idx)
        image = get_img_tensor_pad(image_path)
        label = last_index_row["verb_noun_value"]
        return image, label, image_file, video_id
    
    def get_img_by_idx(self, last_idx_row, idx):
#w        print(last_idx_row["selected_frames"])
        adjusted_idx = idx - last_idx_row["start_index"]
#        print(adjusted_idx)
        try:
            frame = last_idx_row["selected_frames"][adjusted_idx]
        except:
            raise Exception("Failed to get frame for {} idx, in row {}".format(idx, last_idx_row))
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

print("Extracting features for {} dataset".format(opt.type))

dataset = CustomedDataset_pad(opt.type, opt.size, opt.split, opt.part)

dataset_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)

print("TOTAL SIZE: {}".format(len(dataset)))

with torch.no_grad():
    for i_batch, (imgs,labels,image_files, video_ids) in enumerate(tqdm(dataset_loader)):
        #print(i_batch)
        imgs=imgs.to(device)
        features = model_f(imgs)
        print("Features retrieved for batch {}".format(i_batch))
        for i_f in tqdm(range(features.size(0))):
            feature = features[i_f].cpu().numpy()
            label = labels[i_f].cpu().numpy()
            
            image_file = image_files[i_f]
            video_id = video_ids[i_f]
            save_file = "{}_{}.pkl.gz".format(video_id, image_file.split(".")[0])
            save_file_path = "data/epic_kitchens/features_pad/{}/{}".format(opt.type, save_file)

            
            if is_save and not os.path.exists(save_file_path):
                content = {'feature':feature,'label':label,'image_file':image_file} 
                with gzip.open(save_file_path, 'wb') as f:
                    pickle.dump(content,f)
                
                ### need sanity check ### << load it back and compare the result
                with gzip.open(save_file_path, 'rb') as f:
                    content_load = pickle.load(f)
                sanity_check(content, content_load)

