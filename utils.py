import os
import math

import torch
import numpy as np


from torch.utils.data import Dataset
import pickle
from metrics import * 
from rotate import *
import random



def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, args, data_dir, obs_len=8, pred_len=8, skip=1,
        min_ped=0, delim='\t', phase="train"):
        super(TrajectoryDataset, self).__init__()
        self.sparsity = args.sparsity
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        self.n_m_all = []
        seq_list_rel = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                        self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                        self.seq_len))
                num_peds_considered = 0
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
            

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            [start, end]
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        self.seq_traj_ = torch.from_numpy(
            seq_list).type(torch.float)
        
        self.obs_full, self.pred_full = [], []
        self.scene_norm, self.flag_recon = [], []
        obs_traj_rel, pred_traj_rel = [], []
        self.obs_traj, self.pred_traj = [], []
        for ss in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[ss]
            theta = torch.rand(1).to(self.seq_traj_.device) * np.pi * 2
            self.scene_norm.append(self.seq_traj_[start:end,:, :obs_len].permute(0,2,1).contiguous().view(-1,2).mean(0))
            self.flag_recon.append(0)
            if phase == 'train':
                self.seq_traj, _ = rotation_2d_torch(self.seq_traj_[start:end].permute(0,2,1), theta, self.scene_norm[-1])
                self.scene_norm[-1] = self.seq_traj[:,:obs_len].contiguous().view(-1,2).mean(0)
                seq_traj_scene_norm = self.seq_traj - self.scene_norm[-1]
            else: 
                self.seq_traj = self.seq_traj_[start:end].permute(0,2,1)
                seq_traj_scene_norm = self.seq_traj - self.scene_norm[-1]
            self.obs_traj.append(self.seq_traj[:,:obs_len,:])
            self.pred_traj.append(self.seq_traj[:,obs_len:,:])
            obs_traj_rel.append(self.obs_traj[-1][:, 1:, :] - self.obs_traj[-1][:, :-1, :])
            obs_traj_rel[-1] = torch.cat((obs_traj_rel[-1][:, 0, :].unsqueeze(1),obs_traj_rel[-1]),axis=1) #repeats the first one
            pred_traj_rel.append(self.pred_traj[-1] - torch.cat((self.obs_traj[-1][:, -1, :].unsqueeze(1), self.pred_traj[-1][:, :-1, :]),axis=1))
            self.obs_full.append(torch.cat((seq_traj_scene_norm[:,:obs_len,:],obs_traj_rel[-1],\
                                            torch.zeros(*obs_traj_rel[-1].shape[:2],1)), axis=-1))
            self.pred_full.append(torch.cat((seq_traj_scene_norm[:,obs_len:,:],pred_traj_rel[-1]), axis=-1))

        self.obs_traj = torch.vstack(self.obs_traj).permute(1,0,2)
        obs_traj_rel = torch.vstack(obs_traj_rel).permute(1,0,2) 

        self.pred_traj = torch.vstack(self.pred_traj).permute(1,0,2)
        pred_traj_rel = torch.vstack(pred_traj_rel).permute(1,0,2)

        self.obs_traj_full = torch.cat((self.obs_traj, obs_traj_rel), dim = -1)
        self.obs_full = torch.vstack(self.obs_full).permute(1,0,2)
        self.pred_full = torch.vstack(self.pred_full).permute(1,0,2)
        self.masked_obs_full = self.obs_full.clone().detach()
        for ss in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[ss]
            self.masked_obs_full[:obs_len,start:end,:], self.n_m = self.create_missing(self.obs_full[:obs_len,start:end,:])
            self.n_m_all.append(self.n_m)



    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            torch.cat((self.masked_obs_full[:,start:end, :4],\
                      self.masked_obs_full[:,start:end,4].view(*self.masked_obs_full[:,start:end,4].shape[:2],1)),axis=-1),
            torch.cat((self.obs_full[:,start:end, :4],torch.zeros(*self.obs_full[:,start:end].shape[:2],1)), axis=-1), # n_nodes, 2, 8
            self.pred_full[:,start:end, :],
            self.pred_traj[:,start:end, :], # n_nodes, 2, 12
            self.obs_traj_full[:,start:end, :],
            self.scene_norm[index],
            self.flag_recon[index],
            index
        ]
        return out
    
    def create_missing(self,v):
        
        v_new = torch.zeros((v.shape[0], v.shape[1], v.shape[-1]))#define new v
        v_new = v.clone().detach()
        n_m = {}
        n_m["id"] = []
        n_m["timestep"] = []
        sequence = list(range(v.shape[0]*v.shape[1]))
        # the sequence is treated as:
        # [[0,1,2],
        #  [3,4,5],
        #  [6,7,8],
        #  [9,10,11],
        #  ...,
        #  [21,22,23]] for a case where we have 3 pedestrians.
        banned_end = [(self.obs_len-1)*v.shape[1]+n for n in range(v.shape[1])] 
        banned_start = list(range(v.shape[1])) 
        for ban in banned_end: sequence.remove(ban) 
        for ban in banned_start: sequence.remove(ban) 
        num_elim = math.floor(self.sparsity * len(sequence)) # was 0.6 before
        rands = random.sample(sequence, num_elim)
        id_timestep = [rand//v.shape[1] for rand in rands]
        id_ag = [rand%v.shape[1] for rand in rands]
        assert self.obs_len-1 not in id_timestep
        assert 0 not in id_timestep
        n_m["timestep"]=id_timestep
        n_m["id"]=id_ag
        for i,id_t in enumerate(id_timestep):
            v_new[id_t,id_ag[i],:4] = 0
            v_new[id_t,id_ag[i],4] = 1
        return v_new.type(torch.float), n_m


