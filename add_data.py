import os

import torch
import numpy as np
from metrics import * 
from dist import *
import pickle
import copy
from rotate import *

def add_data(train_set, new_traj_all_orig, args, epoch, model):
    
    Shape = train_set.obs_full.shape
    num_targets = [new_traj_all_orig[i].shape[1] for i in range(len(new_traj_all_orig))]
    new_num_targets = sum(num_targets)+ Shape[1]
    obs_full_all = torch.zeros(train_set.obs_full.shape[0],new_num_targets,train_set.obs_full.shape[2])
    obs_full_all[:,:Shape[1],:] = train_set.obs_full.clone() #scene_normed_obs_traj with (rel (from non scene_normed traj)) with mask that is 1, mask is gonna be taken care of also, in get_item
    pred_full_all = torch.zeros(train_set.pred_full.shape[0],new_num_targets,train_set.pred_full.shape[2])
    pred_full_all[:,:Shape[1],:] = train_set.pred_full.clone()#scene_normed_pred_traj with (rel (from non scene_normed traj))
    obs_traj_full_all = torch.zeros(train_set.obs_traj_full.shape[0],new_num_targets,train_set.obs_traj_full.shape[2])
    obs_traj_full_all[:,:Shape[1],:] = train_set.obs_traj_full.clone()#ground truth for reconstruction (with rel) for ADE, FDE (pos and rel both no scene_normed)
    pred_traj_all = torch.zeros(train_set.pred_traj.shape[0],new_num_targets,train_set.pred_traj.shape[2])
    pred_traj_all[:,:Shape[1],:] = train_set.pred_traj.clone()#ground truth for prediction for ADE, FDE, no scene_normed
    masked_obs_full_all = torch.zeros(train_set.masked_obs_full.shape[0],new_num_targets,train_set.masked_obs_full.shape[2])
    masked_obs_full_all[:,:Shape[1],:] = train_set.masked_obs_full.clone()
    new_traj_all = copy.deepcopy(new_traj_all_orig)
    traj_save = []
    for i in range(len(new_traj_all_orig)):
        num_new_ag = num_targets[i]
        new_scene_norm = new_traj_all_orig[i][:args.obs_seq_len,:,:2].contiguous().view(-1,2).mean(0)
        theta = torch.rand(1).to(new_traj_all_orig[i].device) * np.pi * 2
        
        new_traj_all_orig[i], _ = rotation_2d_torch(new_traj_all_orig[i].permute(1,0,2), theta, new_scene_norm)
        new_traj_all_orig[i] = new_traj_all_orig[i].permute(1,0,2)
        new_scene_norm = new_traj_all_orig[i][:args.obs_seq_len,:,:2].contiguous().view(-1,2).mean(0)
        new_traj_all[i] = new_traj_all_orig[i].repeat(1,1,2)

        new_traj_scene_norm = new_traj_all[i][...,:2] - new_scene_norm
        new_traj_all[i][1:,:,2:] = new_traj_all[i][1:,:,:2] - new_traj_all[i][:-1,:,:2]
        new_traj_all[i][0,:,2:] = (new_traj_all[i][1,:,2:]).clone()
        temp_obs_full_all = torch.cat((new_traj_scene_norm[:args.obs_seq_len],\
                                        new_traj_all[i][:args.obs_seq_len,:,2:],\
                                        torch.zeros((*new_traj_all[i][:args.obs_seq_len].shape[:2],1), device=new_traj_scene_norm.device)), axis=-1)
        obs_full_all[:,Shape[1]+sum(num_targets[:i]):Shape[1]+sum(num_targets[:i+1])] = temp_obs_full_all
        temp_traj_full_all = new_traj_all[i][:args.obs_seq_len]
        obs_traj_full_all[:,Shape[1]+sum(num_targets[:i]):Shape[1]+sum(num_targets[:i+1])] = temp_traj_full_all
        temp_masked_obs_full_all, temp = train_set.create_missing(obs_full_all[:,Shape[1]+sum(num_targets[:i]):Shape[1]+sum(num_targets[:i+1])].clone())
        masked_obs_full_all[:,Shape[1]+sum(num_targets[:i]):Shape[1]+sum(num_targets[:i+1])] = temp_masked_obs_full_all
        with torch.no_grad():
            pred_traj, *_ = model(temp_obs_full_all.unsqueeze(0), None, temp_masked_obs_full_all.cuda().unsqueeze(0), None, None,\
                                                                                new_scene_norm, epoch, 0, args, 1, 0, 'val')
        traj = torch.cat((new_traj_all_orig[i],pred_traj[0]))
        traj_scene_norm = traj - new_scene_norm
        traj_rel = traj[1:] - traj[:-1]
        traj_rel = torch.cat((traj_rel[0].unsqueeze(0), traj_rel),axis=0)
        pred_full_all[:,Shape[1]+sum(num_targets[:i]):Shape[1]+sum(num_targets[:i+1])] = torch.cat((traj_scene_norm[args.obs_seq_len:],traj_rel[args.obs_seq_len:,:]), axis=-1)
        pred_traj_all[:,Shape[1]+sum(num_targets[:i]):Shape[1]+sum(num_targets[:i+1])] = traj[args.obs_seq_len:,:,:2]
        
        train_set.masked_obs_full = masked_obs_full_all 
        train_set.obs_full = obs_full_all
        train_set.pred_full = pred_full_all
        train_set.pred_traj = pred_traj_all
        train_set.obs_traj_full = obs_traj_full_all
        train_set.scene_norm.append(new_scene_norm)
        train_set.flag_recon.extend([1])
        train_set.num_seq += 1 
        train_set.n_m_all.append(temp)
        train_set.seq_start_end.extend([[train_set.seq_start_end[-1][1],train_set.seq_start_end[-1][1]+num_new_ag]])
        traj_save.append(traj)

