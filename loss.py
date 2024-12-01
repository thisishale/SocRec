

import torch
from metrics import * 
from dist import *
import torch.nn.functional as F

def comp_loss(V_pred,V_pred_infer,V_target,q_dist, p_dist, p_masked, recon, recon_infer, V_obs,args, flag_recon,phase="train"):
    loss_2 = 0
    num_ag = V_target.shape[2]
    loss_social = torch.tensor(0,device=V_pred.device,dtype=torch.float)
    loss_social_recon = torch.tensor(0,device=V_pred.device,dtype=torch.float)
    if phase == "train":
        loss_vae = p_masked.kl().sum()
        loss_vae /= num_ag
        loss_vae = loss_vae.clamp_min_(args.min_clip_vae)

        diff = recon - V_obs[...,:args.output_size_recon]
        loss_recon = diff.pow(2).sum() 
        loss_recon /= num_ag

        diff = recon_infer - V_obs[...,:args.output_size_recon]
        dist = diff.pow(2).sum(dim=-1).sum(dim=1) 
        loss_recon_sample = dist.min(dim=0)[0] 
        loss_recon_sample = loss_recon_sample.mean()



        loss_2 = q_dist.kl(p_dist).sum()
        loss_2 /= num_ag
        loss_2 = loss_2.clamp_min_(args.min_clip_cvae)

        diff = V_pred - V_target
        loss_1 = diff.pow(2).sum() 
        loss_1 /= num_ag 

        diff_infer = V_pred_infer - V_target
        dist = diff_infer.pow(2).sum(dim=-1).sum(dim=1)
        loss_3 = dist.min(dim=0)[0]
        loss_3 = loss_3.mean()

        mask = torch.triu(torch.ones((num_ag,num_ag), device = V_pred_infer.device), diagonal=1) 
        if num_ag > 1:
            for i in range(V_pred_infer.size(1)):
                temp = [((V_pred_infer[j,i,:,None] - V_pred_infer[j,i,:])**2).sum(-1) for j in range(V_pred_infer.size(0))]
                temp = [temp_each.triu(diagonal=1) for temp_each in temp]
                temp = [torch.masked_select(temp_each, mask.bool()) for temp_each in temp]
                loss_social_ = [(F.relu((args.epsilon)**2 - temp_each)).sum() for temp_each in temp]
                loss_social += sum(loss_social_)
            div = num_ag*(num_ag-1)/2
            loss_social /= div
            for i in range(recon_infer.size(1)):
                temp = [((recon_infer[j,i,:,None] - recon_infer[j,i,:])**2).sum(-1) for j in range(recon_infer.size(0))]
                temp = [temp_each.triu(diagonal=1) for temp_each in temp]
                temp = [torch.masked_select(temp_each, mask.bool()) for temp_each in temp]
                loss_social_ = [(F.relu((args.epsilon)**2 - temp_each)).sum() for temp_each in temp]
                loss_social_recon += sum(loss_social_)
            div = num_ag*(num_ag-1)/2
            loss_social_recon /= div
            loss = loss_1 + (loss_2*args.w_pred_kl) + loss_3 + (loss_social*args.w_social) + (loss_social_recon*args.w_social) + loss_recon+ loss_recon_sample + loss_vae

        else:
            loss = loss_1 + (loss_2*args.w_pred_kl) + loss_3 + loss_recon + loss_recon_sample + loss_vae
            
        return loss
    else:
        diff = V_pred_infer - V_target
        dist = diff.pow(2).sum(dim=-1).sum(dim=1)
        loss_1 = dist.min(dim=0)[0]
        loss_1 = loss_1.mean() 
        loss = loss_1

        return loss

def while_train(V_pred_infer,V_target,recon_infer, V_obs, args):
    diff = recon_infer - V_obs[...,:args.output_size_recon]
    temp = diff.pow(2).sum(dim=-1).sum(dim=1)
    if args.min_mean==0:
        dist_recon = temp.mean(dim=1).min(dim=0)[0]
    else:
        dist_recon = temp.mean(dim=1).mean(dim=0)
    dist = temp.mean(dim=1)
    _, loss_recon_index_min = torch.topk(dist,dim=0,largest=False,k=args.sample_num)#gives us the min sorted values and their indices.
    diff_infer = V_pred_infer - V_target
    temp = diff_infer.pow(2).sum(dim=-1).sum(dim=1)
    if args.min_mean==0:
        dist_infer = temp.mean(dim=1).min(dim=0)[0]
    else:
        dist_infer = temp.mean(dim=1).mean(dim=0)
    dist = temp.mean(dim=1)
    _, loss_infer_index_min = torch.topk(dist,dim=0,largest=False,k=args.sample_num)
    return dist_infer, dist_recon, loss_infer_index_min, loss_recon_index_min #mean is for the agents