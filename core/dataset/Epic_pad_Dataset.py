# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 2022

@author: alachyankar

This file defines the Custom Dataset Epic_pad_Dataset. 

This dataset is used by the training loop to gather the word2vector and image extracted features. 

Dataset defines a length and __getitem__ method as defined by the interface. 

This is done here to provide the file name of the pickle file, the image features, and the one-hot encoded interaction label. 
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
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
import pandas as pd
import os
import glob


class Epic_pad_Dataset(Dataset):

    def __init__(self, partition, content, label_type = 'interaction'):
        super(Epic_pad_Dataset, self).__init__() 
        
        self.grid_size = 17
        
        self.partition = partition
        self.hoi = pd.read_csv('data/epic_kitchens/epic_kitchens_hoi.csv')
        self.mask = np.zeros(len(self.hoi)) 
        self.content = content
        
        self.i2a = np.argmax(self.content['Z_a'],axis=-1).astype(np.int32)
        self.len_a = self.content['Z_a'].shape[-1]
        
        self.i2o = np.argmax(self.content['Z_o'],axis=-1).astype(np.int32)
        self.len_o = self.content['Z_o'].shape[-1]
        


        self.feature_dir = 'data/epic_kitchens/features_pad/{}'.format(partition)
        self.features = sorted(glob.glob(self.feature_dir + "/*"))
        print("len features {}: {}".format(partition, len(self.features)))
        self.image_dir = 'data/epic_kitchens/epic_images'
    
    def calculate_total_length(self):
        start_frame = self.samples["start_frame"]
        end_frame = self.samples["stop_frame"]
        return end_frame.sub(start_frame)

    def get_img_by_idx(self, last_idx_row, idx):
        max_index_minus = last_idx_row["max_index"] - idx
        frame = last_idx_row["stop_frame"] - max_index_minus
        frame_file = "frame_{}".format("0" * (10-len(str(frame))) + str(frame))
        return frame_file, last_idx_row["video_id"]
 
    def __len__(self):
        return len(self.features)


    def __getitem__(self, idx):
        feature_path = self.features[idx]
        
        with gzip.open(feature_path, 'rb') as f:
            content_load = pickle.load(f)
        
        feature,label,image_file = content_load["feature"],content_load["label"],content_load["image_file"]
        
        
        label_one_hot = np.zeros(self.hoi.shape[0])
        label_one_hot[label] = 1

        shape = feature.shape
        assert shape[-1] == shape[-2] == self.grid_size
                
        return feature_path,feature,label_one_hot
