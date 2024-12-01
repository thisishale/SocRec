
from metrics import * 
from dist import *

def reset_data(orig_loader_len, orig_loader_targets, train_set, new_traj_all, args):
    loader_len = len(train_set)
    # loader_targets = train_set.obs_full.shape[1]
    
    if loader_len>orig_loader_len:
        train_set.masked_obs_full = train_set.masked_obs_full[:,:orig_loader_targets]
        train_set.obs_full = train_set.obs_full[:,:orig_loader_targets]
        train_set.pred_full = train_set.pred_full[:,:orig_loader_targets]
        train_set.pred_traj = train_set.pred_traj[:,:orig_loader_targets]
        train_set.obs_traj_full = train_set.obs_traj_full[:,:orig_loader_targets]
        train_set.pred_traj = train_set.pred_traj[:,:orig_loader_targets]
        train_set.scene_norm = train_set.scene_norm[:orig_loader_len]
        train_set.flag_recon = train_set.flag_recon[:orig_loader_len]
        train_set.num_seq = orig_loader_len
        train_set.seq_start_end = train_set.seq_start_end[:orig_loader_len]
        train_set.n_m_all = train_set.n_m_all[:orig_loader_len]
