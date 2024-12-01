

import torch
import numpy as np



def ade(pred,target):
    ade_mean, ade_max, ade_min = [], [], []
    num_ag = pred.shape[2]
    diff = torch.sqrt((pred[:,:,:,0]-target[:,:,:,0])**2 + (pred[:,:,:,1]-target[:,:,:,1])**2)
    for i in range(num_ag):
        ade = diff[:,:,i].view(pred.shape[0],-1).mean(1)
        ade_mean.append(ade.mean().item())
        ade_max.append(ade.max().item())
        ade_min.append(ade.min().item())
        
    return ade_mean, ade_max, ade_min



def fde(pred,target):
    fde_mean, fde_max, fde_min = [], [], []
    num_ag = pred.shape[2]
    diff = torch.sqrt((pred[:,:,:,0]-target[:,:,:,0])**2 + (pred[:,:,:,1]-target[:,:,:,1])**2)
    for i in range(num_ag):
        fde = diff[:,-1,i].view(pred.shape[0],-1).mean(1)
        fde_mean.append(fde.mean().item())
        fde_max.append(fde.max().item())
        fde_min.append(fde.min().item())

    return fde_mean, fde_max, fde_min


