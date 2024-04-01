#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of loss functions for image segmentation

Almost verbatim copy paste from:
Source:https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
Original Author: https://www.kaggle.com/bigironsphere

Changes: Changed F.binary_cross_entropy in DiceBCELoss to F.binary_cross_enropy_with_logits
to enable use of the pos_weight parameter for unbalanced segmentation.

Added DISCLOSS for multi-instance segmentation
"""

import sys
import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath("/mnt/dsml/projects/dentseg"))
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

def loss_function(function='BCE', ratio=10.0):
    pos_weight = torch.tensor([ratio]).to(device)
    #weight = torch.tensor([1.0,ratio]).to(device)
    loss_functions = {}
    loss_functions['BCE'] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_functions['IOU'] = IoULoss()
    loss_functions['DICE'] = DiceLoss()
    loss_functions['DICEBCE'] = DiceBCELoss(weight=pos_weight)
    loss_functions['FOCAL'] = FocalLoss()       
    loss_functions['TVERSKY'] = TverskyLoss()
    loss_functions['FOCALTVERSKY'] = FocalTverskyLoss()
    #For use in multi-instance mode only
    loss_functions['DISCLOSS'] = DiscLoss(func=loss_functions["DICEBCE"], weight=pos_weight*10)
    return loss_functions[function]

class DiscLoss(nn.Module):
    #A dice loss function that adds an instance dicrimination term
    def __init__(self, func, weight=None, size_average=True):
        super(DiscLoss, self).__init__()
        self.loss_1 = func
        
    def forward(self, inputs, targets, alpha=0.4, beta=0.4, gamma=0.2, delta=0.5, smooth=1):
        
        #dice / diceBCE loss
        loss_1 = self.loss_1(inputs, targets)
        
        #instance discrimination loss
        pred_diff = torch.abs(inputs[:, 1:, :, :] - inputs[:, :-1, :, :])
        disc_loss1 = torch.mean(F.relu(delta - pred_diff))
        
        #instance discrimination loss 2
        inputs = (F.sigmoid(inputs)>=0.5).float()
        mask_sum = torch.sum(inputs,dim=1)
        mask_sum = mask_sum - torch.ones_like(mask_sum).float()
        mask_sum = torch.clamp(mask_sum,0)/(inputs.shape[1]/2)
        disc_loss2 = torch.mean(mask_sum)
        
        #conditional identities
        i1 = int(loss_1 < 0.7)
        i2 = int(disc_loss1 > 0.05)
        i3 = int(disc_loss2 > 0.05)
        
        #loss components
        l1 = 1*loss_1
        l2 = i1*i2*disc_loss1
        l3 = i1*i3*disc_loss2
        
        self.comps = (loss_1.item(),disc_loss1.item(),disc_loss2.item())
        
        return alpha*l1 + beta*l2 + gamma*l3
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
            
    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.weight, reduction='mean')
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        #BCE = F.binary_cross_entropy(inputs, targets, weight=self.weight, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

#ALPHA = 0.8
#GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
#ALPHA = 0.5
#BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    
# ALPHA = 0.5
# BETA = 0.5
# GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.8, beta=0.2, gamma=3):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
    
# class LovaszHingeLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(LovaszHingeLoss, self).__init__()

#     def forward(self, inputs, targets):
#         inputs = F.sigmoid(inputs)    
#         Lovasz = lovasz_hinge(inputs, targets, per_image=False)                       
#         return Lovasz

