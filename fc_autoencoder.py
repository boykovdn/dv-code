import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import dataloading
import time
import datetime

class FCAutoencoder(nn.Module):
    """
    Fully connected autoencoder
    """
    
    def __init__(self, size_input):
        """
        :size_input: dimensions of input images
        """
        super().__init__()
        
        self.input_layer = nn.Linear(size_input,400)
        self.h1 = nn.Linear(400,100)
        self.h2 = nn.Linear(100,10)
        self.encoded = nn.Linear(10,100)
        self.h3 = nn.Linear(100,400)
        self.output = nn.Linear(400,size_input)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.h1(out)
        out = self.relu(out)
        out = self.h2(out)
        out = self.relu(out)
        self.encoder_vals = out
        out = self.encoded(out)
        out = self.relu(out)
        out = self.h3(out)
        out = self.relu(out)
        out = self.output(out)
        
        return out
        
        
#class MnistVectors(torch.utils.data.Dataset):
#    
#    def __init__(self, digit=0):
#        super().__init__()
#        
#        (Xtr, Ytr), (Xte, Yte) = keras.datasets.mnist.load_data()
#        X = np.concatenate((Xtr, Xte))
#        Y = np.concatenate((Ytr, Yte))
#        
#        # Use only the training dataset, I don't need a test one for an AE since it maps data onto itself.
#        self.mnist_vectors, self.labels = convert_mnist_to_vectors((X, Y))
#        self.mnist_vectors = prepare_data(self.mnist_vectors)
#        
#        if digit is not None:
#            self.mnist_vectors = self.mnist_vectors[self.labels == digit]
#            self.labels = np.zeros(self.mnist_vectors.shape[0]) + digit
#
#    def __getitem__(self, idx):
#        mvec = torch.autograd.Variable(torch.tensor(self.mnist_vectors[idx]).float(), requires_grad=True)
#        label = torch.tensor(self.labels[idx]).float()
#        
#        return mvec, label
#    
#    def __len__(self):
#        return len(self.labels)
#    
    
def train(ae, dataloader, criterion, optimizer, use_gpu=True, epochs=5):

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
                
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
    torch.save(ae.state_dict(), 'autoencoder{}.ckpt'.format(timestamp))
    return losses

image_size = (100,100)
data_path = "/home/boyko/Documents/courses/deepvision/cell_images/Uninfected"
batch_size = 16

ae = FCAutoencoder(image_size[0] * image_size[1])
transforms = torchvision.transforms.Compose([ 
    torchvision.transforms.Resize(image_size), 
    torchvision.transforms.ToTensor(), 
    dataloading.Flatten()     
    ])
data = dataloading.CellImages(data_path, transforms=transforms)
print(len(data))
dataloader = DataLoader(data, batch_size=16, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters())

#TODO Add Tensorboard / logging
train(ae, dataloader, criterion, optimizer, use_gpu=False)
