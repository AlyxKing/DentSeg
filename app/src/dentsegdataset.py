#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 23:20:43 2023

@author: alex

Dataset definition/Transformations for the image dataset
"""
#Basic dependencies
import sys 
import os
import copy
import glob
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import logging

#from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import logistic

#Pytorch and albumentations
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.utils.checkpoint as checkpoint
import albumentations as A

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from albumentations.pytorch import ToTensorV2

#Local imports
#sys.path.append(os.path.abspath("/mnt/dsml/projects/dentseg"))
sys.path.append(os.path.abspath("/app/src"))
sys.path.append(os.path.abspath("/app/dataset"))
from lossfunctions import *
from utils import *
from HalfUNet import HUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class DentsegDataset(Dataset):
    """
    Handles image import, transformations, augmentation
    preparation for input into network
    """
    def __init__(self, conf, transform=True):
        """
        root_dir: Directory with all the images and annotation file.
        desired_size: size of 1:1 letterboxed image
        """
        self.do_transform = transform
        self.args = conf
        self.target_size = self.args.image_size
        self.root_dir = self.args.dataset_path
        self.img_path = os.path.join(self.root_dir, "img")
        #self.mask_path = os.path.join(self.root_dir,"masks_machine")
        self.ann_path = os.path.join(self.root_dir, "ann")
        if transform:
            self.transform = self.transform_pipeline(self.target_size)
        #list of image paths
        self.images = [file for file in glob.glob("*.jpg",root_dir=self.img_path)]
        self.masks = []
        self.ann = []
        for image in self.images:
            mask = image.rstrip('jpg') + 'png'
            anno = image + '.json'
            self.masks.append(mask)
            self.ann.append(anno)
        
        #Define Tooth classes
        with open(os.path.join(self.root_dir,"meta.json"),'r') as file:
            self.classes = json.load(file)  
        
        self.classes = sorted(self.classes['classes'],key=lambda x: int(x['title']))
        self.class_colors = [tooth['color'] for tooth in self.classes]
        self.class_colors = [self.hex_to_rgb(color) for color in self.class_colors]

        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_path, self.images[idx])
        #mask_path = os.path.join(self.mask_path, self.masks[idx])
        ann_path = os.path.join(self.ann_path, self.ann[idx])
        self.check_path = image_path
        #create class, polygon dict
        with open(ann_path,'r') as file:
            annotations = json.load(file)
            
        polys = [annotations['objects'][i]['points']['exterior'] for i in range(len(annotations['objects']))]
        classes = [annotations['objects'][i]['classTitle'] for i in range(len(annotations['objects']))]
        poly_data = dict(zip(classes,polys))
            
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        if self.args.out_c == 1:
            mask_type = 'numpy'
        else:
            mask_type = 'list'
        mask = self.create_mask_from_polygons(
            image.shape, 
            polys = poly_data, 
            output = mask_type
            )
        # Feed image and mask into transformation function
        if self.do_transform:
            augmented = self.apply_transform(
                image = image,
                mask = mask,
                transform=self.transform,
                labels=[str(i+1) for i in range(mask.shape[0])],
                )
            image = augmented['image']
            mask = augmented['mask']
            if mask_type == 'list':
                mask = torch.permute(mask,(2,0,1))
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
        return image, mask
    
    @staticmethod
    def create_mask_from_polygons(image_shape, polys:dict, output='list'):
        #Generate an empty mask for each tooth
        mask = np.zeros((32, image_shape[0], image_shape[1]),dtype=np.uint8)
        #Iterate over polys in dict
        #Catch no mask exception
        try:
            for class_id, class_polygon in polys.items():
                # Convert polygon vertices into an array of shape (1, -1, 2)
                np_polygon = np.array([class_polygon], dtype=np.int32)
                # Draw the polygon on the mask
                cv2.fillPoly(mask[int(class_id)-1], np_polygon, 1)#int(class_id) + 1)
        except:
            pass
        if output == 'list':
            return np.transpose(mask,(1,2,0))
        elif output == 'numpy':
            return np.sum(mask,axis=0)
            
    @staticmethod
    def hex_to_rgb(color):
        color = color.strip('#')
        rgb = []
        for i in (0, 2, 4):
            decimal = int(color[i:i+2], 16)
            rgb.append(decimal)
        return tuple(rgb) 

    @staticmethod
    def transform_pipeline(desired_size): 
        #Image randomisation
        transform = A.Compose([
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.CLAHE(p=0.01),
        A.ColorJitter(
            contrast=0.0,
            saturation=0.7,
            hue=0.015,
            brightness=0.4,
            p=0.2,
            ),
        A.RandomBrightnessContrast(p=0.1),
        A.Resize(
            desired_size,
            desired_size,
            always_apply=True,
        ),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
        ],additional_targets={'mask': 'mask'}
        )
        return transform

    @staticmethod
    def apply_transform(image, mask, transform, labels):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image,mask=mask,labels=labels)
        return transformed    

class EMA:
    #Work this into the training loop
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def train(args, train_set, use_ema=False, model=None, model_path=None, mode='semantic'):
    setup_logging(args)
    device = args.device
    dataset = train_set#DentsegDataset(args.dataset_path,transform=True,desired_size=args.image_size)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True,num_workers=1,) 
    if not model:
        model = HUNet(c_in=args.in_c,c_out=args.out_c,ghost_mode=True,flat=args.flat,size=args.image_size,device='cuda:0').to(device)
    model.train()
    if use_ema:
        ema = EMA(beta=0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    #due to the ratio of background to mask, pos_weight set as 13 
    loss_func = loss_function(args.lossfunc)
    logger = SummaryWriter(os.path.join(args.dataset_path,"runs", args.run_name))
    l = len(data_loader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(data_loader)
        for i, (images, maps) in enumerate(pbar):
            
            images = images.to(device)
            #For single instance segmentation, convert to boolean bit mask
            maps = maps.bool().type(torch.uint8)
            maps = maps.reshape(len(maps),
                                args.out_c, 
                                args.image_size,
                                args.image_size).float().to(device)
            predicted_maps = model(images)
            loss = loss_func(predicted_maps, maps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_ema:
                ema.step_ema(ema_model,model,step_start_ema=2000)
            if args.lossfunc == "DISCLOSS":
                l_comps = loss_func.comps
                pbar.set_postfix(LOSS=loss.item(), COMPS=l_comps, epoch="{}/{}".format(epoch+1,args.epochs))
            else:
                pbar.set_postfix(LOSS=loss.item(), epoch="{}/{}".format(epoch+1,args.epochs))
            logger.add_scalar(args.lossfunc, loss.item(), global_step=epoch * l + i)
            
        
        if epoch % 10 == 0:
            torch.save(model, os.path.join(args.dataset_path,"models", args.run_name, "ckpt.pth"))
            if use_ema:
                torch.save(ema_model, os.path.join(args.dataset_path,"models", args.run_name, "ema_ckpt.pth"))
    return model


def test(args, data, model) -> tuple:
    with torch.no_grad():
        model.eval()
        data_loader = DataLoader(data, batch_size=args.batch_size,shuffle=True,num_workers=1,) 
        loss_func = loss_function(args.evalfunc)
        logger = SummaryWriter(os.path.join(args.dataset_path,"runs", args.run_name + "test_set"))
        pbar = tqdm(data_loader)
        batch_loss = {}
        test_samples = []
        for i, (images, maps) in enumerate(pbar):
            images = images.to(device)
            maps = maps.bool().type(torch.uint8)
            maps = maps.reshape(len(maps),
                                args.out_c,
                                args.image_size,
                                args.image_size
                                ).float().to(device)
            predicted_maps = model(images)
            loss = loss_func(predicted_maps, maps)
            
            batch_loss[i] = loss.detach().cpu().numpy()
            output = (images.detach().cpu().numpy(), maps.detach().cpu().numpy(), predicted_maps.detach().cpu().numpy())
            test_samples.append(output)
            pbar.set_postfix(IOU=loss.item())
            logger.add_scalar(args.evalfunc, loss.item())
            print(f"test loss: {loss.item()}")
        batch_loss = pd.Series(batch_loss)
        print(f"mean loss: {batch_loss.mean()}")
        return test_samples, model
    
def load_dec(func, model_path, full_model=False):
    #decorator to load model
    def wrapper(*argv,**kwargs):
        if full_model:
            model = torch.load(model_path)
            return func(*argv,**kwargs,model=model)
        args = argv[0]
        model = HUNet(c_in=args.in_c,c_out=args.out_c,
                      ghost_mode=True,
                      flat=args.flat,
                      size=args.image_size,
                      device='cuda:0'
                      ).to(device)
        model.load_state_dict(torch.load(model_path))
        return func(*argv,**kwargs,model=model)
    return wrapper

def create_argparse():
    import argparse
    description = """
    Configure and run the DentSeg model. Available loss functions for training and evaluation include:
    - BCE: Binary Cross-Entropy Loss
    - IOU: Intersection Over Union Loss
    - DICE: Dice Loss
    - DICEBCE: Combination of Dice and BCE Loss
    - FOCAL: Focal Loss
    - TVERSKY: Tversky Loss
    - FOCALTVERSKY: Focal Tversky Loss
    - DISCLOSS: Discriminative Loss (for multi-instance segmentation only)
    """
    
    # Initialize the parser    
    parser = argparse.ArgumentParser(description=description)
    
    # Define arguments
    parser.add_argument("--run_name", default="DentSeg0", type=str, help="Name of the run")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for training")
    parser.add_argument("--batch_size", default=25, type=int, help="Batch size for training")
    parser.add_argument("--image_size", default=256, type=int, help="Input image size")
    parser.add_argument("--dataset_path", default="/app/dataset", type=str, 
                        help="Path to the dataset. Should include images and annotations.")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device for training (e.g., 'cuda:0')")
    parser.add_argument("--lossfunc", default="DICEBCE", type=str, 
                        help="Loss function for training (e.g., 'DICEBCE')")
    parser.add_argument("--evalfunc", default="IOU", type=str, 
                        help="Evaluation function for model performance (e.g., 'IOU')")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--in_c", default=1, type=int, help="No. of input channels")
    parser.add_argument("--out_c", default=1, type=int, help="No. of output channels")
    parser.add_argument("--flat", action='store_true', help="ON/OFF flag for half U-Net unified channel width")
    parser.add_argument("--load_model", action='store_true', help="ON/OFF flag for loading model")
    parser.add_argument("--model_name", default=None, type=str, help="Specify model to load (if different from run_name)")
    parser.add_argument("--full_model", action='store_true', help='ON/OFF switch for loading full_model as opposed to state space dict')
    return parser
    
def launch(only_test=False, load_model=False, 
           model_name=None, full_model=False) -> tuple:
    os.chdir('/')
    parser = create_argparse()
    args = parser.parse_args()
    dataset = DentsegDataset(conf=args, transform=True)
    generator = torch.Generator().manual_seed(42)
    train_data, test_data = random_split(dataset, [500,95],generator=generator)
    
    if not model_name:
        model_path = os.path.join(args.dataset_path,"models", args.run_name, "ckpt.pth")
    else:
        model_path = os.path.join(args.dataset_path,"models", args.run_name, model_name + ".pth")
        
    if load_model and not only_test:
        loader = load_dec(train, model_path, full_model=full_model)
        model = loader(args, train_data)
        output, model = test(args, test_data, model)
        return model, output
    elif load_model and only_test:
        loader = load_dec(test, model_path, full_model=full_model)
        output, model = loader(args, test_data)
        return model, output
    elif only_test and not load_model:
        try:
            output, model = test(args, test_data, model)
            return model, output
        except:
            print('Provide HalfUNet model')     
            
    model = train(args, train_data, use_ema=True)
    output = test(args, test_data, model)[0]
    return model, output

if __name__ == '__main__':
    model, output = launch()


