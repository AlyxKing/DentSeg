import os
import sys 
import torch  
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn

#Local imports
sys.path.append(os.path.abspath("/mnt/dsml/projects/dentseg"))
from lossfunctions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.szrhow()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def apply_mask(image, mask, threshold=0.85):
    from scipy.stats import logistic
    xray = image.copy()
    seg_mask = logistic.cdf(mask.copy()) 
    xray[seg_mask>threshold] = 1
    return xray
    
def display_images(data, batch_no:int, *args):
    #Displays images from the Dentseg Dataset tensor tuple
    #figsize = (dataset.target_size, dataset.target_size)|
    data = data[batch_no]
    #argv = args[0]
    fig,axs = plt.subplots(3,8)
    xrays = data[0]
    for idx, layer in enumerate(data):
        for i, image in enumerate(layer):
            if idx == 0:
                axs[0,i].imshow(image[0])
            elif idx == 1:
                axs[idx,i].imshow(apply_mask(xrays[i][0], image[0], threshold=0.5))
            elif idx == 2:
                axs[idx,i].imshow(apply_mask(xrays[i][0], image[0], threshold=0.85))
                
def setup_logging(args):
    os.makedirs(os.path.join(args.dataset_path,"runs"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path,"models"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path,"results"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path,"models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path,"results", args.run_name), exist_ok=True)        
    os.makedirs(os.path.join(args.dataset_path,"results", args.run_name + "test_set"), exist_ok=True)  
    plt.show()

