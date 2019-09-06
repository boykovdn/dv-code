import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dv_code import fc_autoencoder
from dv_code import cnn_autoencoder
from dv_code import dataloading
import torchvision
from PIL import Image
import os
import sys

to_PIL = torchvision.transforms.ToPILImage()

def illustrate_performance(model,
                           output_path,
                           transforms,
                           images_path="../cell_images/Parasitized",
                           model_type='fc',
                           loss=None):
    """
    Generates images of original/reconstruction pairs, and a loss graph
    :loss: if not None, graph the loss history and save in the same dir
    """
    image_size = model.image_size
    images_tag = images_path.split('/')[-1]
    # Make sure model is set to evaluation mode
    model.eval()

    performance_dir = output_path + "/{}".format(images_tag)
    if not os.path.exists(performance_dir):
        os.mkdir(performance_dir)

    cell_images = dataloading.CellImages(images_path, transforms=transforms)

    for img_index in range(0,10):
        img_orig = cell_images[img_index].view(3,image_size[0],image_size[1])
        img_orig = to_PIL(img_orig)

        if model_type == 'cnn':
            img_reconst = model.forward(cell_images[img_index].view(1,3,image_size[0], image_size[1]))
            img_reconst = to_PIL(img_reconst.squeeze(0))
        elif model_type == 'fc':
            img_reconst = model.forward(cell_images[img_index].view(3,image_size[0], image_size[1]))
            img_reconst = to_PIL(img_reconst)                                                      
        img_new = Image.new('RGB', (image_size[0]*2, image_size[1]))
        img_new.paste(img_orig,(0,0))
        img_new.paste(img_reconst,(image_size[0],0))

        img_new.save("{}/{}.png".format(performance_dir, img_index))

def main():
    assert len(sys.argv) == 3, "python3 <script> <path/model_name> <cnn or fc>"
    model_path = sys.argv[1]
    model_type = sys.argv[2]

    if model_type == 'fc':
        model = fc_autoencoder.FCAutoencoder(image_size[0] * image_size[1])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
            dataloading.Flatten()
            ])
    
    elif model_type == 'cnn':
        model = cnn_autoencoder.CNNAutoencoder()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))      
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
            ])
    
    illustrate_performance(model, model_path, transforms, model_type=model_type)
    
if __name__ == "__main__":
    main()
