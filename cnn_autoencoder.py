import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import os,sys
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
    
    def __init__(self, io_channels=3):
        """
        :io_channels: number of channels of input and output, should be 3, unless grayscale
        """
        super().__init__()
        # Encoder learnable layers
        self.conv1 = nn.Conv2d(io_channels, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)

        # Decoder learnable layers
        self.upconv1 = nn.ConvTranspose2d(24, 12, 3, stride=2, padding=1,output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(12, 6, 3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(6, io_channels, 3, stride=2, padding=1, output_padding=1)

        # Activation and pooling layers
        self.relu = nn.ReLU()
        self.pool2d = nn.MaxPool2d(2)
        
    def forward(self, x):
        out = self.pool2d(self.relu(self.conv1(x)))
        out = self.pool2d(self.relu(self.conv2(out)))
        out = self.pool2d(self.relu(self.conv3(out)))

        out = self.relu(self.upconv1(out))
        out = self.relu(self.upconv2(out))
        out = self.upconv3(out)
        
        return out

def train(ae, dataloader, criterion, optimizer, use_gpu=True, epochs=5):
    t_begin = time.time()

    if use_gpu:
        ae.cuda()
        criterion.cuda()
        
    losses = []
    for epoch in tqdm(range(epochs), desc='Epoch'):
        for step, example in enumerate(tqdm(dataloader, desc='Batch')):
            if use_gpu:
                example = example.cuda()
                
            optimizer.zero_grad()
            prediction = ae(example)
            loss = criterion(example, prediction)
            loss.backward()
            optimizer.step()
            
            losses.append(float(loss))
            if (step % 300) == 0:
                tqdm.write('Loss: {}\n'.format(loss))
                
    t_end = time.time()
    timestamp = datetime.datetime.fromtimestamp(t_end).strftime('%Y-%m-%d-%H-%M-%S')
    torch.save(ae.state_dict(), 'cnn_autoencoder{}.ckpt'.format(timestamp))
    time_training = t_end - t_begin
    return losses, timestamp, time_training

if __name__ == "__main__":
    image_size = (128,128)
    data_path = "/export/home/dv/dv016/datasets/cell_images/Uninfected"
    batch_size = 32
    
    ae = CNNAutoencoder()
    transforms = torchvision.transforms.Compose([ 
        torchvision.transforms.Resize(image_size), 
        torchvision.transforms.ToTensor(), 
        ])
    data = dataloading.CellImages(data_path, transforms=transforms)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters())
    print(ae)
    
    losses, timestamp, time_training = train(ae, dataloader, criterion, optimizer, use_gpu=True)
    with open("./losses{}.pickle".format(timestamp), "wb") as outfile:
        pickle.dump(losses, outfile)
    with open("./autoencoder_info{}.log".format(timestamp), "w+") as log_file:
        print("Model\n",ae, file=log_file)
        print("Criterion\n", criterion, file=log_file)
        print("Optimizer\n", optimizer, file=log_file)
        print("Transforms\n", transforms, file=log_file)
        print("Time Training\n", time_training)
