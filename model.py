import os
import torch
import torch.nn as nn
import numpy as np
from utils_model import *
import torch
from torch.nn.functional import *
from AgentAwareAttention import *
from mlp import *
from dist import *
from Encoder import Encoder
from Decoder import Decoder
from Decoder_recon import Decoder_recon
from F_Encoder import F_Encoder
from loss import while_train
import random


class agent_former(nn.Module):
    def __init__(self, args):
        super(agent_former,self).__init__()
        self.main_dir = os.path.dirname(__file__)
        self.f_encoder = F_Encoder(args)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.last_epoch = args.epochs_thresh
        self.decoder_masked = Decoder_recon(args)
        self.num_params = args.num_params
        self.num_params_dist = args.num_params *2
        self.pz = nn.Linear(args.model_dim, self.num_params_dist)
        self.pz_masked = nn.Linear(args.model_dim, self.num_params_dist)
        
        self.sample_num = args.sample_num
        self.compute_sample =  args.compute_sample
        self.out_mlp_dim = [512,256]
        self.indices = []
        self.out_mlp = MLP(args.model_dim, self.out_mlp_dim, 'relu')
        self.qz = nn.Linear(self.out_mlp.out_dim, self.num_params_dist)
        initialize_weights(self.pz.modules())
        initialize_weights(self.qz.modules())
        self.flag_to_reset = 0
        self.du = {}
        self.dl = {}
        self.l = {}
        self.D = {}
        self.w = {}
        self.l_count = {}
        
    def forward(self,v, v_fut, v_masked,obs_traj_gt_pos_rel, pred_traj_gt_pos,scene_norm, epoch,sample_full_loss, args,flag_recon, index, phase='val'):
        
        if epoch==0 and phase == "train":
            self.l[index.item()] = []
            self.du[index.item()] = []
            self.dl[index.item()] = []
            self.l_count[index.item()] = 0

        recon_infer = 0 # for valid and test time when we dont have this
        recons_temp, recons_temp_save = [], []
        if phase == "train":
            sample_num = 1
        else:
            sample_num = self.sample_num
        v_enc, v_enc_pooled = self.encoder(v) # v_enc [obs*n,1,dim] v_enc_pooled [n,dim]
        v_enc_masked, _ = self.encoder(v_masked)
        p_z_masked_params = self.pz_masked(v_enc_masked.repeat_interleave(sample_num, dim=1).view(-1,v_enc_masked.shape[-1]))  
        prior_masked = Normal(params=p_z_masked_params)  
        z_p_masked = prior_masked.sample()
        recon = self.decoder_masked(v,z_p_masked,v_enc_masked,scene_norm,sample_num)
        
        if phase == "train":    
            v_fut_enc = self.f_encoder(v_fut, v_enc,scene_norm,sample_num)
            p_z_params = self.pz(v_enc_pooled.repeat_interleave(sample_num, dim=0))  
            prior = Normal(params=p_z_params)   
            q_z_params = self.qz(v_fut_enc.repeat_interleave(sample_num, dim=0))
            q_dist = Normal(params=q_z_params)
            z_q = q_dist.sample()        
        else:
            p_z_params = self.pz(v_enc_pooled.repeat_interleave(sample_num, dim=0))  
            prior = Normal(params=p_z_params)  
            q_dist = prior
            z_p = prior.sample()
        if phase == "train":
            v_dec = self.decoder(v,z_q,v_enc,scene_norm,sample_num)
            v_dec = v_dec.unsqueeze(2).permute(2,1,0,3)
        else:
            v_dec_infer = self.decoder(v,z_p,v_enc,scene_norm,sample_num)
            v_dec_infer = v_dec_infer.view(-1,sample_num,*v_dec_infer.shape[1:])
            v_dec_infer = v_dec_infer.permute(1,2,0,3)
            v_dec = v_dec_infer
            self.flag_to_reset = 0 
            if epoch - self.last_epoch == args.every_few -1:
                self.last_epoch = epoch + 1
                
        if self.compute_sample and phase=='train':
            sample_num = self.sample_num
            p_z_params_infer = self.pz(v_enc_pooled.repeat_interleave(sample_num, dim=0))
            prior_infer = Normal(params=p_z_params_infer)
            z_infer = prior_infer.sample()
            v_dec_infer = self.decoder(v,z_infer,v_enc,scene_norm,sample_num)
            v_dec_infer = v_dec_infer.view(-1, sample_num, *v_dec_infer.shape[1:]).permute(1,2,0,3)  

            p_z_masked_params_infer = self.pz_masked(v_enc_masked.repeat_interleave(sample_num, dim=1).view(-1,v_enc_masked.shape[-1]))  
            prior_masked_infer = Normal(params=p_z_masked_params_infer)  
            z_p_masked_infer = prior_masked_infer.sample()
            recon_infer = self.decoder_masked(v,z_p_masked_infer,v_enc_masked,scene_norm,sample_num)

            loss_infer, _, _, loss_recon_index = while_train(v_dec_infer, \
                        pred_traj_gt_pos, recon_infer, obs_traj_gt_pos_rel, args)
            if epoch==self.last_epoch and flag_recon==0 and phase=='train': 
                self.l_count[index.item()] = (np.array(self.dl[index.item()])>np.array(self.du[index.item()])).sum()
                self.D[index.item()] = self.l_count[index.item()] < args.D*len(self.du[index.item()])
            if epoch == self.last_epoch and flag_recon==0 and phase=='train':
                if self.D[index.item()]:
                    self.flag_to_reset = 1
                    self.indices.append(index.item())
                    if args.choice_recon == 'top':
                        for k in range(args.sample_num):
                            recons_temp_save.append(torch.stack([recon_infer[loss_recon_index[k],:,i,:] for i in range(recon_infer.shape[2])]).permute(1,0,2).detach())
                    elif args.choice_recon == 'random':
                        l = list(range(args.sample_num))
                        random.shuffle(l)
                        for k in range(args.sample_num):
                            recons_temp_save.append(torch.stack([recon_infer[l[k],:,i,:] for i in range(recon_infer.shape[2])]).permute(1,0,2).detach())
            if flag_recon==0 and phase=='train':
                if epoch==self.last_epoch: 
                    self.l[index.item()] = []
                    self.du[index.item()] = []
                    self.dl[index.item()] = []
                self.l[index.item()].append((loss_infer).cpu().detach().numpy())
                if len(self.l[index.item()]) > 1:
                    self.dl[index.item()].append((min((self.l[index.item()][-1]-self.l[index.item()][-2]),0)*np.log(self.l[index.item()][-1]/self.l[index.item()][-2])))
                    self.du[index.item()].append((max(self.l[index.item()][-1]-self.l[index.item()][-2],0)*np.log(self.l[index.item()][-1]/self.l[index.item()][-2])))
        return v_dec, v_dec_infer, q_dist, prior, recon, recon_infer, prior_masked, recons_temp_save[:args.k_recon], self.flag_to_reset

