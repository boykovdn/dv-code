import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import os,sys
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image, ImageFilter
import numpy as np
import dataloading
import time
import datetime
import pickle

class CNNAutoencoder(nn.Module):
    """
    Convolutional autoencoder
    """
    
    def __init__(self, size_input):
        """
        :size_input: dimensions of input images
        """
        super().__init__()
        # initilize input size of first linear layer with [batchsize*hight*width*featuresize]
        self.lin_size = int(size_input/2*16)
        self.conv1 = nn.Conv1d(3,6, 3,padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv1d(3,16, 5,padding =2)
        self.h1 = nn.Linear(self.lin_size,400)
        self.h2 = nn.Linear(400,100)
        self.h3 = nn.Linear(100,10)
        self.encoded = nn.Linear(10,10)
        self.h4 = nn.Linear(10,100)
        self.h5 = nn.Linear(100,400)
        self.output = nn.Linear(400,size_input)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        print('0: ',x.shape)
        out = self.relu(self.conv1(x))
        print('1: ',out.shape)
        out = self.relu(self.pool1(out))
        print('2: ',out.shape)
        out = self.relu(self.conv2(out))
        print('3: ',out.shape)
        # resize output of last layer in order to fit to next linear input layer
        out = out.view(-1,self.num_flat_features(out))
        print('4: ',out.shape)
        out = self.relu(self.h1(out))
        print('5: ',out.shape)
        out = self.relu(self.h2(out))
        print('6: ',out.shape)
        out = self.relu(self.h3(out))
        print('7: ',out.shape)
        self.encoder_vals = out
        out = self.relu(self.encoded(out))
        print('8: ',out.shape)
        out = self.relu(self.h4(out))
        print('9: ',out.shape)
        out = self.relu(self.h5(out))
        print('10: ',out.shape)
        out = self.output(out)
        print('11',out.shape)
        
        return out
