
import torch
import numpy as np

from utils import * 
from metrics import * 

def test(KSTEPS, model, test_set, args, epoch):
    model.eval()


    masked_obs_full_test = torch.cat((test_set.masked_obs_full[:,:, :4],\
                        test_set.masked_obs_full[:,:,4].view(*test_set.masked_obs_full[:,:,4].shape[:2],1)),axis=-1).cuda()
    obs_full_test = (torch.cat((test_set.obs_full[:,:,:4],torch.zeros(*test_set.obs_full.shape[:2],1)), axis=-1)).cuda()
    pred_full_test = (test_set.pred_full).cuda()
    pred_traj_test_ = (test_set.pred_traj).cuda()
    obs_traj_full_test = (test_set.obs_traj_full).cuda()
    temp = test_set.scene_norm
    scene_norm_test_ = [tensor.cuda() for tensor in temp]
    flag_recon_test_ = torch.FloatTensor(test_set.flag_recon).cuda()

    ademin, ademax, ademean = [], [], []
    fdemin, fdemax, fdemean = [], [], []
    ademin_recon, ademax_recon, ademean_recon = [], [], []
    fdemin_recon, fdemax_recon, fdemean_recon = [], [], []
    step =0 
    recon_full_loss, sample_full_loss = 0, 0 # doesnt matter
    len_test = len(scene_norm_test_)
        # make a list of indices from 0 to len_train
    indices_loop = np.linspace(0, len_test-1, len_test, dtype=int)


    random.shuffle(indices_loop)
    cnt = 0

    for ind in indices_loop: 

        start, end = test_set.seq_start_end[ind]

        #Get data
        masked_obs_traj = masked_obs_full_test[:,start:end].unsqueeze(0).contiguous()
        obs_traj = obs_full_test[:,start:end].unsqueeze(0).contiguous()
        fut_traj = pred_full_test[:,start:end, :].unsqueeze(0).contiguous()
        pred_traj_gt_pos = pred_traj_test_[:,start:end, :].unsqueeze(0).contiguous()
        obs_traj_gt_pos_rel = obs_traj_full_test[:,start:end, :].unsqueeze(0).contiguous()
        scene_norm = scene_norm_test_[ind].unsqueeze(0).contiguous()
        flag_recon = flag_recon_test_[ind].unsqueeze(0).contiguous()

        _, pred_traj_infer, _, _, recon, _, _, _, _, flag_to_reset = model(obs_traj, fut_traj, masked_obs_traj, obs_traj_gt_pos_rel, pred_traj_gt_pos, scene_norm, epoch, sample_full_loss, args, flag_recon, ind,'val')

        ade_mean, ade_max, ade_min = ade(pred_traj_infer,pred_traj_gt_pos)
        ademin.extend(ade_min)
        ademax.extend(ade_max)
        ademean.extend(ade_mean)
        fde_mean, fde_max, fde_min = fde(pred_traj_infer,pred_traj_gt_pos)
        fdemin.extend(fde_min)
        fdemax.extend(fde_max)
        fdemean.extend(fde_mean)
        ade_mean_recon, ade_max_recon, ade_min_recon = ade(recon,obs_traj_gt_pos_rel)
        ademin_recon.extend(ade_min_recon)
        ademax_recon.extend(ade_max_recon)
        ademean_recon.extend(ade_mean_recon)
        fde_mean_recon, fde_max_recon, fde_min_recon = fde(recon,obs_traj_gt_pos_rel)
        fdemin_recon.extend(fde_min_recon)
        fdemax_recon.extend(fde_max_recon)
        fdemean_recon.extend(fde_mean_recon)
        
    ade_ = sum(ademin)/len(ademin)
    ade_max_ = sum(ademax)/len(ademax)
    ade_mean_ = sum(ademean)/len(ademean)
    fde_ = sum(fdemin)/len(fdemin)
    fde_max_ = sum(fdemax)/len(fdemax)
    fde_mean_ = sum(fdemean)/len(fdemean)
    ade_recon_ = sum(ademin_recon)/len(ademin_recon)
    ade_recon_max_ = sum(ademax_recon)/len(ademax_recon)
    ade_recon_mean_ = sum(ademean_recon)/len(ademean_recon)
    fde_recon_ = sum(fdemin_recon)/len(fdemin_recon)
    fde_recon_max_ = sum(fdemax_recon)/len(fdemax_recon)
    fde_recon_mean_ = sum(fdemean_recon)/len(fdemean_recon)
    return ade_, fde_, ade_max_, fde_max_, ade_mean_, fde_mean_, ade_recon_, fde_recon_, ade_recon_max_,\
          fde_recon_max_, ade_recon_mean_, fde_recon_mean_

