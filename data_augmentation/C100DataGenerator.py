import torch
import torch.nn as nn
import torch.nn.functional as tF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import shutil, time, os, requests, random, copy


class C100DataGen(Dataset):
    def __init__(self,phase,imgarr,s = 0.5):
        self.phase = phase
        self.imgarr = imgarr
        self.s = s
        self.mean = np.mean(self.imgarr/255.0,axis=(0,2,3),keepdims=True) # mean of the dataset
        self.std = np.std(self.imgarr/255.0,axis=(0,2,3),keepdims=True) # std of the dataset
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5), # random horizontal flip with 0.5 probability
                                              transforms.RandomResizedCrop(32,(0.8,1.0)), # random crop with 0.8-1.0 scaling factor
                                              transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.8*self.s, # random color jitter with 0.8 scaling factor
                                                                                                                 0.8*self.s, 
                                                                                                                 0.8*self.s, 
                                                                                                                 0.2*self.s)], p = 0.8),
                                                                  transforms.RandomGrayscale(p=0.2) # random grayscale with 0.2 probability
                                                                 ])]) # random apply color jitter and grayscale with 0.8 and 0.2 probability respectively
                                                                

    def __len__(self):
        return self.imgarr.shape[0]

    # get the data for a given index
    def __getitem__(self,idx):
        
        x = self.imgarr[idx] 
        #print(x.shape)
        x = x.astype(np.float32)/255.0

        x1 = self.augment(torch.from_numpy(x))
        x2 = self.augment(torch.from_numpy(x))
        
        x1 = self.preprocess(x1)
        x2 = self.preprocess(x2)
        
        return x1, x2

    # shuffle the data at the end of each epoch
    def on_epoch_end(self):
        self.imgarr = self.imgarr[random.sample(population = list(range(self.__len__())),k = self.__len__())]

    def preprocess(self,frame):
        frame = (frame-self.mean)/self.std
        return frame
    
    # data augmentation
    def augment(self, frame, transformations = None):
        
        if self.phase == 'train':
            frame = self.transforms(frame)
        else:
            return frame
        
        return frame