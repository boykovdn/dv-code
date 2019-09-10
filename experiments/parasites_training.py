from PIL import Image
from dv_code.dataloading import CellImages
from dv_code.cnn_autoencoder import CNNAutoencoder
from tqdm.auto import tqdm
import numpy as np

import torch
import torchvision

import matplotlib.pyplot as plt

DATASET_INFECTED = '/home/biv20/datasets/cell_images/Parasitized'
DATASET_HEALTHY = '/home/biv20/datasets/cell_images/Uninfected'
MODEL_WEIGHTS_PATH = '/home/biv20/sources/dv-code/meta/cnn_autoencoder_2019-09-09-12-28-56/cnn_autoencoder2019-09-09-12-28-56.ckpt'

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def main():
    
    ae = CNNAutoencoder()
    ae.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    ae.cuda()
    ae.eval()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128,128)),
        torchvision.transforms.ToTensor(),
        ])

    ds_i = CellImages(DATASET_INFECTED, transforms=transforms, recursive_load=False)
    dataloader = torch.utils.data.DataLoader(ds_i, batch_size=1)

    for image in tqdm(dataloader):
        image = image.cuda()
        prediction = ae.forward(image)
        # Remove the reconstructed background
        parasite = image - prediction
        display_img = parasite.cpu().detach()
        
        imshow(display_img[0,:])
        plt.show()

if __name__ == "__main__":
    main()
