import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import fc_autoencoder
import dataloading
import torchvision
from PIL import Image
import os
import sys

images_path = "../cell_images/Parasitized"
transforms = dataloading.transforms
to_PIL = torchvision.transforms.ToPILImage()
image_size = fc_autoencoder.image_size

def illustrate_performance(model,model_path, loss=None):
    """
    Generates images of original/reconstruction pairs, and a loss graph
    :loss: if not None, graph the loss history and save in the same dir
    """

    performance_dir = model_path[:-5]
    if not os.path.exists(performance_dir):
        os.mkdir(performance_dir)
    
    cell_images = dataloading.CellImages(images_path, transforms=transforms)

    for img_index in range(0,10):
        img_orig = cell_images[img_index].view(3,image_size[0],image_size[1])
        img_orig = to_PIL(img_orig)
        img_reconst = model.forward(cell_images[img_index]).view(3, image_size[0], image_size[1])
        img_reconst = to_PIL(img_reconst)
        
        img_new = Image.new('RGB', (image_size[0]*2, image_size[1]))
        img_new.paste(img_orig,(0,0))
        img_new.paste(img_reconst,(image_size[0],0))

        img_new.save("{}/{}.png".format(performance_dir, img_index))

def main():
    assert len(sys.argv) == 2, "python3 <script> <path/model_name>"
    model_path = sys.argv[1]
    model = fc_autoencoder.FCAutoencoder(image_size[0] * image_size[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    illustrate_performance(model, model_path)
    
if __name__ == "__main__":
    main()

    

