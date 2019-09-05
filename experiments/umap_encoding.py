"""
This script generates images of the UMAP 
decomposition of the autoencoder latent
space.
"""

import torch.nn as nn
import torch
import torchvision
from dv_code.cnn_autoencoder import CNNAutoencoder
from dv_code import dataloading
import argparse
from tqdm.auto import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import umap

def get_latent_representations():
    OUTPUT_DIR = './'

    parser = argparse.ArgumentParser(description='Generate Umap embedding of cnn_autoencoder latent code.')
    parser.add_argument('--model', type=str, help='Specify path to model weights of choice.', required=True)
    parser.add_argument('--output', type=str, help='Specify output directory where images will be saved.', required=False)
    parser.add_argument('--input', type=str, help='Path to input data.', required=True)
    parser.add_argument('--device', type=int, help='Select CUDA device to run network on.', required=False)
    
    args = parser.parse_args()

    torch.cuda.device(args.device) if args.device is not None else 0

    ae = CNNAutoencoder()
    ae.load_state_dict(torch.load(args.model))
    ae.eval()
    ae.cuda() if args.device is not None else 0

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128,128)),
        torchvision.transforms.ToTensor(),
        ])

    cellimages = dataloading.CellImages(args.input, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(cellimages, batch_size=1)

    latent_representations = []
    for image in tqdm(dataloader):
        if args.device is not None:
            image = image.cuda()
        prediction = ae.forward(image)
        latent_representations.append(ae.latent_representation.cpu().detach().numpy())

    return latent_representations

def get_umap_embedding(lr):
    """
    :param lr: ndarray; Latent representations

    This function assumes the first dimension of the data is the index.
    """
    lr = np.array(lr)

    if len(lr.shape) > 2:
        lr = np.reshape(lr, (lr.shape[0], -1))

    embedding = umap.UMAP(n_neighbors=5, min_dist=0.5, metric='euclidean').fit_transform(lr)

    fig = plt.figure(figsize=(8,8))
    plt.scatter(embedding[:,0], embedding[:,1], alpha=0.3)
    plt.show()


def main():

#    with open('latent_representations.pickle', 'wb') as out_pickle:
#        pickle.dump(lr, out_pickle)

    with open('latent_representations.pickle', 'rb') as in_pickle:
        lr = pickle.load(in_pickle)

    get_umap_embedding(lr)

if __name__ == '__main__':
    main()
