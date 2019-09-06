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
from dv_code.reconstructions import illustrate_performance

import argparse
from tqdm.auto import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import umap

def get_latent_representations(args, ae, transforms):
    OUTPUT_DIR = './'
    torch.cuda.device(args.device) if args.device is not None else 0

    PATH_HEALTHY = args.input + "/Uninfected"
    PATH_INFECTED = args.input + "/Parasitized"

    cellimages_healthy = dataloading.CellImages(PATH_HEALTHY, transforms=transforms)
    dataloader_healthy = torch.utils.data.DataLoader(cellimages_healthy, batch_size=1)

    cellimages_infected = dataloading.CellImages(PATH_INFECTED, transforms=transforms)
    dataloader_infected = torch.utils.data.DataLoader(cellimages_infected, batch_size=1)

    latent_representations_healthy = []
    for image in tqdm(dataloader_healthy):
        if args.device is not None:
            image = image.cuda()
        prediction = ae.forward(image)
        latent_representations_healthy.append(ae.latent_representation.cpu().detach().numpy())

    latent_representations_infected = []
    for image in tqdm(dataloader_infected):
        if args.device is not None:
            image = image.cuda()
        prediction = ae.forward(image)
        latent_representations_infected.append(ae.latent_representation.cpu().detach().numpy())
    
    return latent_representations_healthy, latent_representations_infected

def plot_umap_embedding(lr, label="Data Name"):
    """
    :param lr: ndarray; Latent representations.

    This function assumes the first dimension of the data is the index.
    """
    lr = np.array(lr)

    if len(lr.shape) > 2:
        lr = np.reshape(lr, (lr.shape[0], -1))

    embedding = umap.UMAP(n_neighbors=5, min_dist=0.5, metric='euclidean').fit_transform(lr)

    plt.scatter(embedding[:,0], embedding[:,1], alpha=0.3, label=label)

def main():

    parser = argparse.ArgumentParser(description='Generate Umap embedding of cnn_autoencoder latent code.')
    parser.add_argument('--model', type=str, help='Specify path to model weights of choice.', required=True)
    parser.add_argument('--output', type=str, help='Specify output directory where images will be saved.', required=False)
    parser.add_argument('--input', type=str, help='Path to input data.', required=True)
    parser.add_argument('--device', type=int, help='Select CUDA device to run network on.', required=False)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128,128)),
        torchvision.transforms.ToTensor(),
        ])
    
    args = parser.parse_args()

    ae = CNNAutoencoder()
    ae.load_state_dict(torch.load(args.model))
    ae.eval()
    ae.cuda() if args.device is not None else 0

    lr_h, lr_i = get_latent_representations(args, ae, transforms)

    with open('latent_representations_both.pickle', 'wb') as out_pickle:
        pickle.dump((lr_h, lr_i), out_pickle)

    with open('latent_representations_both.pickle', 'rb') as in_pickle:
        lr_h, lr_i = pickle.load(in_pickle)

    fig = plt.figure(figsize=(8,8))
    plot_umap_embedding(lr_h, label='healthy')
    plot_umap_embedding(lr_i, label='infected')
    output_dir = '/'.join(args.model.split('/')[:-1])
    plt.savefig(output_dir)

    ae.cpu()
    images_path = args.input + "/Parasitized"
    illustrate_performance(ae, output_dir, transforms, images_path=images_path, model_type='cnn')
    images_path = args.input + "/Uninfected"
    illustrate_performance(ae, output_dir, transforms, images_path=images_path, model_type='cnn')

if __name__ == '__main__':
    main()
