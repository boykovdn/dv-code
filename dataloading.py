import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import os

class CellImages(torch.utils.data.Dataset):

    def __init__(self, path, transforms=None):

        """
        :transforms: torchvision.transforms.Compose object
        :path: where the Uninfected and Parasitised folders are located
        """
        self.transforms = transforms
        self.image_names_raw = self._load_data_names(path)
        assert "Uninfected" in path or "Parasitized" in path, "Expected path of format ../Uninfected or ../Parasitized, but got {}".format(path)
        assert len(self.image_names_raw) != 0, "Found no .png in {}".format(path)

    def _load_data_names(self, path):
        """
        Load .png image names - to be loaded lazily later
        :path: path to folder that contains some .png images
        """
        image_names_raw = ["{}/{}".format(path, image_name) for image_name in os.listdir(path) if ".png" in image_name]
    
        return image_names_raw

    def __getitem__(self, idx):

        image_handle = Image.open(self.image_names_raw[idx])

        if self.transforms is not None:
            return self.transforms(image_handle)
        else:
            return image_handle
  
    def __len__(self):
        return len(self.image_names_raw)

class Flatten(object):
    """
    Transformation to flatten a tensor of size (CxHxW) to (Cx(H*W))
    """
    def __call__(self, image_tensor):
        image_shape = image_tensor.size()
        
        assert len(image_shape) == 3, "Expected 3D tensor but got {}".format(image_shape)
        return image_tensor.view((image_shape[0], -1))


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((100,100)),
    torchvision.transforms.ToTensor(),
    Flatten()    
])
 

def main():
    """
    Test example
    """
    civs = CellImages("../project/cell_images/Uninfected", transforms=transforms)
    dataloader = torch.utils.data.DataLoader(civs, batch_size=16)
    
if __name__ == "__main__":
    main()
