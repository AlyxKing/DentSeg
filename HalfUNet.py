#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 00:08:34 2023

@author: alex
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation
from functools import partial

torch.set_default_tensor_type('torch.FloatTensor')
torch.backends.cuda.matmul.allow_tf32 = True
# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')

#Half UNet: Unified channel numbers, reduced complexity
#Use GhostNet module to increase feature map count
# double_conv

class SelfAttention(nn.Module):
    #layer norm of x is the query, key, and value of multiheadattention
    #serves to feed information about the local context of a pixel into the feature layers
    def __init__(self, channels, size, device="cuda:0"):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels,4,batch_first=True)
        self.lnorm = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels,channels),
            nn.GELU(),
            nn.Linear(channels, channels)
            )
        self.to(device)
        
    def forward(self, x):
        #convert x to a channels * size^2 2d matrix, swap axes
        x = x.view(-1, self.channels, int(self.size*self.size)).swapaxes(1,2)
        #apply layer norm
        x_lnorm = self.lnorm(x)
        #take the self attention value of  x_lnorm
        attention_value, _ = self.mha(x_lnorm, x_lnorm, x_lnorm)
        #add feed forward weights to the attention weights
        attention_value = self.ff_self(attention_value) + attention_value
        #twist back into original dimensions and send back to the model
        return attention_value.swapaxes(2,1).view(-1, self.channels, self.size, self.size)
        
        

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, mid_c=None, ghost=False, ratio=2, relu=True, residual=False,device="cuda:0"):
        super().__init__()
        self.mode = ghost
        self.device = device
        #Res link boolean specifier
        self.residual = residual
        if not mid_c:
            mid_c = out_c
        if self.mode:
            #ghost module efficiency mode
            self.gate = nn.Sigmoid()
            self.out_c = out_c
            init_channels = math.ceil(out_c/ratio)
            new_channels = init_channels*(ratio-1)
            
            self.primary_conv = nn.Sequential( 
                nn.Conv2d(in_c, init_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(init_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential(),
            )
            #cheap operation replicates feature maps with basic operations
            #Convolve by channel
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, kernel_size=3, padding=1,groups=init_channels,bias=False),
                nn.BatchNorm2d(new_channels),
                nn.ReLU(inplace=True) if relu else nn.Sequential()
            )
            #Attention branch of the ghost block:
            #Creates an efficient type of Self Attention
            #Convolutes horizontally and vertically respectively
            self.short_conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, groups=in_c, bias=False),
                nn.BatchNorm2d(out_c),
                nn.Conv2d(out_c, out_c, kernel_size=(1,5), stride=1, padding=(0,2), groups=in_c, bias=False),
                nn.BatchNorm2d(out_c), 
                nn.Conv2d(out_c, out_c, kernel_size=(5,1), stride=1, padding=(2,0), groups=in_c, bias=False),
                nn.BatchNorm2d(out_c)
            )
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c,mid_c,kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c,out_c,kernel_size=3,padding=1,bias=False),
            nn.GroupNorm(1,out_c),
            nn.GELU()
            )
        self.to(device)
        
    def forward(self, x):
        #Ghost mode
        #print('double_conv start')
        #print(x.device)
        if (self.residual and self.mode) or (self.mode):
            res = self.short_conv(F.avg_pool2d(x,kernel_size=2,stride=2))
            #Real conv (1/2)
            x1  = self.primary_conv(x)
            #Cheap conv (1/2)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1,x2], dim=1)
            return out[:,:self.out_c,:,:]*F.interpolate(self.gate(res),size=(out.shape[-2],out.shape[-1]),mode='nearest').to(self.device)
        elif self.residual and not self.mode:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class DownAcross(nn.Module):
    def __init__(self, in_c, out_c, ghost_mode=False,device="cuda:0"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, in_c, residual=True, ghost=ghost_mode),
            DoubleConv(in_c, out_c, ghost=ghost_mode)
            )
        
        # self.time_emb = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(emb_dim,out_c)
        #     )
        self.to(device)
        
    def forward(self,x):
        x = self.maxpool_conv(x)
        return x 

class UpScale(nn.Module):
    def __init__(self, inc_c, out_c, scale_factor,device="cuda:0"):
        super().__init__()
        self.reduce_channels = partial(
            Conv2dNormActivation,
                kernel_size=1,
                activation_layer=partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)
                )
        self.upscale = nn.Upsample(scale_factor=scale_factor, mode='bilinear',align_corners=True)
        self.smooth_conv = partial(
            Conv2dNormActivation,
                kernel_size=3,
                activation_layer=partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)
                )
        
        self.reduce_c = self.reduce_channels(in_channels=inc_c,out_channels=out_c)
        self.smooth = self.smooth_conv(in_channels=out_c,out_channels=out_c)
        self.to(device)
        
    def forward(self, x):
        x = self.reduce_c(x)
        x = self.upscale(x)
        x = self.smooth(x)
        return x

class UpStep(nn.Module):
    def __init__(self, inc_c, out_c, scale_factor,device="cuda:0"):
        super().__init__()
        self.reduce_channels = partial(
            Conv2dNormActivation,
                kernel_size=1,
                activation_layer=partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)
                )
        self.upscale = nn.Upsample(scale_factor=scale_factor, mode='bilinear',align_corners=True)
        self.smooth_conv = partial(
            Conv2dNormActivation,
                kernel_size=3,
                activation_layer=partial(nn.LeakyReLU, negative_slope=0.1, inplace=True)
                )
        
        self.reduce_c = self.reduce_channels(in_channels=inc_c,out_channels=out_c)
        self.smooth = self.smooth_conv(in_channels=out_c*2,out_channels=out_c)
        self.to(device)
        
    def forward(self, x, y):
        
        x = self.reduce_c(x)
        x = self.upscale(x)
        x = torch.cat((y,x),1)
        x = self.smooth(x)
        return x      
        
        
class OutStep(nn.Module):
    def __init__(self, in_c, end_c, ghost_mode=False,device="cuda:0"):
        super().__init__()
        self.seq = nn.Sequential(
            nn.GroupNorm(1, in_c),
            DoubleConv(in_c, in_c,residual=True, ghost=ghost_mode, device=device),
            DoubleConv(in_c, in_c,ghost=ghost_mode, device=device),
            )
        self.finalconv = nn.Sequential(
            nn.Conv2d(in_c,end_c,kernel_size=1, padding=0, bias=False),
           #nn.Sigmoid(),
            )
        self.to(device)
        
    def forward(self,x):
        x = self.seq(x)
        x = self.finalconv(x)
        return x
        
class HUNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, n_layers=4, conv_c=96, flat=True, size=256, ghost_mode=False, device="cuda:0", **kwargs):
        super().__init__()
        if flat == True:
            self.f_exp = 0
        else: self.f_exp = 1
        self.init_chan = conv_c
        self.ghost = ghost_mode
        self.device = device
        
        self.n_layers = n_layers
        
        #Creates channel schedule based on conv_c (eg. 64, 128, 512, ... , conv_c^n_layers)
        form = lambda x: conv_c*(2**x)**self.f_exp
        #incoming layer
        self.inc = DoubleConv(c_in,conv_c,ghost=ghost_mode,device=self.device)
        #Pool and double conv layers
        self.down_across = [DownAcross(x,x*(2**self.f_exp),ghost_mode=ghost_mode,device=self.device) for x in (form(i) for i in range(self.n_layers))]
        #self attention blocks for SA config (resource intensive)
        self.sa_blocks = [SelfAttention(x*(2**self.f_exp),size*(0.5**y),device=self.device) for (x,y) in zip((form(i) for i in range(self.n_layers)),(range(self.n_layers)))]
        #Upscalers for Half-UNet config
        if flat:
            self.upscalers = [UpScale(form(x+1),max(conv_c,form(x+1)/2),2**(x+1),device=self.device) for x in range(self.n_layers)]
        #Upsteps for standard UNet config
        else:
            self.upsteps = [UpStep(form(x+1),int(max(conv_c,form(x+1)/2)),2,device=self.device) for x in range(self.n_layers)]
        self.outstep = OutStep(self.init_chan,c_out,ghost_mode=ghost_mode,device=self.device)
        #Move all layers to the correct device
        self.to(self.device)
        
        #self.merge_op = Merge()
        
    
    # def pos_encoding(self, t, channels):
    #     #Converts t to a sinusoidal encoding, ready to be embedded into the half UNET
    #     inv_freq = 1.0 / (
    #         10000
    #         ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
    #     )
    #     pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
    #     pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
    #     pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    #     return pos_enc
        
    def forward(self, x):
        levels = []
        x = self.inc(x)
        self.pass_up = x
        levels.append(x)
        
        #Down
        for layer in range(self.n_layers):
            x = self.down_across[layer](x)
            if not self.ghost:
                x = self.sa_blocks[layer](x)
            #For Half-UNet conf.
            if self.f_exp == 0:
                self.pass_up = self.pass_up + self.upscalers[layer](x)
            else:
                levels.append(x)
                
        #Up (For standard UNet config)
        if self.f_exp != 0:
            for layer in reversed(range(0,self.n_layers)):
                x = self.upsteps[layer](x,levels[layer])
            self.pass_up = x
            
        output = self.outstep(self.pass_up)
        return output
    


        
        