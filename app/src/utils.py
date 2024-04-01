import os
import sys 
import torch  
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
from scipy.stats import logistic
from torchvision.utils import draw_segmentation_masks

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

# def apply_mask(image, mask, threshold=0.85):
#     from scipy.stats import logistic
#     xray = image.copy()
#     seg_mask = logistic.cdf(mask.copy()) 
#     xray[seg_mask>threshold] = 1
#     return xray

def apply_colored_mask(image, masks, threshold=0.5):
    """
    Applies multiple masks to an image, each in a specified color.
    
    Parameters:
    - image: The original grayscale image.
    - masks: A list of masks to apply.
    - colors: A list of colors corresponding to each mask.
    - threshold: The threshold for applying the masks.
    
    Returns:
    - The image with colored masks applied.
    """
    # Ensure the image is in RGBA to overlay colors
    
    colored_image = cv2.cvtColor(image*255,cv2.COLOR_GRAY2RGB)
    colored_image = torch.tensor(colored_image).detach().cpu().type(torch.uint8).permute(2,0,1)
    seg_mask = torch.tensor(logistic.cdf(masks)>threshold).detach().cpu()
    colored_image = draw_segmentation_masks(colored_image,seg_mask)
    
    return colored_image.permute(1,2,0).numpy()

def display_results(data, batch_no:int):
    """
    Displays an X-ray image with ground truth and predicted masks color-coded and overlaid.
    
    Parameters:
    - data: A tuple containing the datasets (X-rays, ground truth masks, predicted masks).
    - batch_no: The index of the image to display.
    """
    
    
    x_ray_images, ground_truth_masks, predicted_masks = data[batch_no]
    num_sets = len(data[0][0])
    fig, axs = plt.subplots(num_sets, 3, figsize=(15, 5*num_sets))
    
    # Remove the gap between images
    plt.subplots_adjust(wspace=0, hspace=0)
    
    for i in range(num_sets):
        # Plot X-Ray
        axs[i, 0].imshow(x_ray_images[i][0], cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 0].set_title('X-Ray' if i == 0 else "")

        # Plot Ground Truth Mask
        axs[i, 1].imshow(apply_colored_mask(x_ray_images[i][0],ground_truth_masks[i]))
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Ground Truth Mask' if i == 0 else "")

        # Plot Predicted Mask
        axs[i, 2].imshow(apply_colored_mask(x_ray_images[i][0],predicted_masks[i]))
        axs[i, 2].axis('off')
        axs[i, 2].set_title('Predicted Mask' if i == 0 else "")
    
    # Show the plot
    plt.show()

    
    
    # data = data[batch_no]
    # #argv = args[0]
    # fig,axs = plt.subplots(3,len(data[0]))
    # xrays = data[0]
    
    # for idx, layer in enumerate(data):
    #     for i, image in enumerate(layer):
    #         if idx == 0:
    #             axs[0,i].imshow(image[0])
    #         elif idx == 1:
    #             axs[idx,i].imshow(apply_mask(xrays[i][0], image[0], threshold=0.5))
    #         elif idx == 2:
                
    #             axs[idx,i].imshow(apply_mask(xrays[i][0], image[0], threshold=0.5))
                
def setup_logging(args):
    os.makedirs(os.path.join(args.dataset_path,"runs"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path,"models"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path,"results"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path,"models", args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path,"results", args.run_name), exist_ok=True)        
    os.makedirs(os.path.join(args.dataset_path,"results", args.run_name + "test_set"), exist_ok=True)  
    plt.show()

