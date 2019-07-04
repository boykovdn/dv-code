import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import dataloading
import time
import datetime
import pickle

class FCAutoencoder(nn.Module):
    """
    Fully connected autoencoder
    """
    
    def __init__(self, size_input, image_size=(100,100)):
        """
        :size_input: dimensions of input images
        """
        super().__init__()
    
        self.image_size = image_size
        
        self.input_layer = nn.Linear(size_input,400)
        self.h1 = nn.Linear(400,100)
        self.h2 = nn.Linear(100,100)
        self.encoded = nn.Linear(100,100)
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
    torch.save(ae.state_dict(), 'autoencoder{}.ckpt'.format(timestamp))
    time_training = t_end - t_begin
    return losses, timestamp, time_training

if __name__ == "__main__":
    image_size = (100,100)
    data_path = "/export/home/dv/dv016/datasets/cell_images/Uninfected"
    
    ae = FCAutoencoder(image_size[0] * image_size[1])
    transforms = torchvision.transforms.Compose([ 
        torchvision.transforms.Resize(image_size), 
        torchvision.transforms.ToTensor(), 
        dataloading.Flatten()     
        ])
    data = dataloading.CellImages(data_path, transforms=transforms)
    dataloader = DataLoader(data, batch_size=256, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters())
    
    losses, timestamp, time_training = train(ae, dataloader, criterion, optimizer, use_gpu=True)
    with open("./losses{}.pickle".format(timestamp), "wb") as outfile:
        pickle.dump(losses, outfile)
    with open("./autoencoder_info{}.log".format(timestamp), "w+") as log_file:
        print("Model\n",ae, file=log_file)
        print("Criterion\n", criterion, file=log_file)
        print("Optimizer\n", optimizer, file=log_file)
        print("Transforms\n", transforms, file=log_file)
        print("Time Training\n", time_training)
