from loss import comp_loss
import torch
import os
import torch
import numpy as np
import torch.optim as optim
from utils import * 
from metrics import * 
from model import *
from loss import comp_loss
from custom_schd import *
from add_data import add_data
from reset_data import reset_data
import pickle

def train_val(train_set, val_set, args, log):

    orig_loader_len = len(train_set)
    orig_loader_targets = train_set.obs_full.shape[1]
    main_dir = os.path.dirname(__file__)
    model = agent_former(args).cuda()

    #Training settings 
    if args.no_scheduler == False:
        if args.scheduler_type == 'steplr':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            optim_sched = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        if args.scheduler_type == 'multisteplr':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            optim_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,80], gamma=args.gamma)
        elif args.scheduler_type == 'warmup':
            optimizer_ = optim.Adam(model.parameters(), lr=args.initial_lr_)
            optim_sched = custom_schd(optimizer_)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)



    checkpoint_dir = os.path.join(main_dir, "checkpoint",args.dataset,args.name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    sample_full_loss = 0


    masked_obs_full_train = torch.cat((train_set.masked_obs_full[:,:, :4],\
                        train_set.masked_obs_full[:,:,4].view(*train_set.masked_obs_full[:,:,4].shape[:2],1)),axis=-1).cuda()
    obs_full_train = (torch.cat((train_set.obs_full[:,:,:4],torch.zeros(*train_set.obs_full.shape[:2],1)), axis=-1)).cuda()
    pred_full_train = (train_set.pred_full).cuda()
    pred_traj_train_ = (train_set.pred_traj).cuda()
    obs_traj_full_train = (train_set.obs_traj_full).cuda()
    temp = train_set.scene_norm
    scene_norm_train_ = [tensor.cuda() for tensor in temp]
    flag_recon_train_ = torch.FloatTensor(train_set.flag_recon).cuda()

    masked_obs_full_val = torch.cat((val_set.masked_obs_full[:,:, :4],\
                        val_set.masked_obs_full[:,:,4].view(*val_set.masked_obs_full[:,:,4].shape[:2],1)),axis=-1).cuda()
    obs_full_val = (torch.cat((val_set.obs_full[:,:,:4],torch.zeros(*val_set.obs_full.shape[:2],1)), axis=-1)).cuda()    
    pred_full_val = (val_set.pred_full).cuda()
    pred_traj_val_ = (val_set.pred_traj).cuda()
    obs_traj_full_val = (val_set.obs_traj_full).cuda()
    temp = val_set.scene_norm
    scene_norm_val_ = [tensor.cuda() for tensor in temp]
    flag_recon_val_ = torch.FloatTensor(val_set.flag_recon).cuda()
    
    for epoch in range(args.num_epochs):
        loss_batch = 0 
        batch_count = 0

        seq_gt, new_traj_all = [], []
            
        model.train()
        print("*"*100)
        print("started training")

        len_train = len(scene_norm_train_)
        indices_loop = np.linspace(0, len_train-1, len_train, dtype=int)

        random.shuffle(indices_loop)
        cnt = 0

        for ind in indices_loop: 

            start, end = train_set.seq_start_end[ind]
            batch_count+=1

            masked_obs_traj = masked_obs_full_train[:,start:end].unsqueeze(0).contiguous()
            obs_traj = obs_full_train[:,start:end].unsqueeze(0).contiguous()
            fut_traj = pred_full_train[:,start:end, :].unsqueeze(0).contiguous()
            pred_traj_gt_pos = pred_traj_train_[:,start:end, :].unsqueeze(0).contiguous()
            obs_traj_gt_pos_rel = obs_traj_full_train[:,start:end, :].unsqueeze(0).contiguous()
            scene_norm = scene_norm_train_[ind].unsqueeze(0).contiguous()
            flag_recon = flag_recon_train_[ind].unsqueeze(0).contiguous()

            if args.scheduler_type == 'steplr':
                optimizer.zero_grad()
            elif args.scheduler_type == 'multisteplr':
                optimizer.zero_grad()
            elif args.scheduler_type == 'warmup':
                optim_sched.optimizer.zero_grad()

            pred_traj, pred_traj_infer, q_dist, p_dist, recon,\
                  recon_infer, p_masked, new_traj, flag_to_reset = model(obs_traj, fut_traj, masked_obs_traj, obs_traj_gt_pos_rel,\
                                                                                         pred_traj_gt_pos, scene_norm, epoch, sample_full_loss, args,\
                                                                                              flag_recon, ind, 'train')
        
                

            loss = comp_loss(pred_traj, pred_traj_infer, pred_traj_gt_pos, q_dist,\
                            p_dist, p_masked, recon, recon_infer, obs_traj_gt_pos_rel, args, flag_recon, "train") #P_dist is basically q here in the literature. 
            loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

            if args.scheduler_type == 'steplr' or args.no_scheduler == True:
                optimizer.step()
            elif args.scheduler_type == 'multisteplr':
                optimizer.step()
            elif args.scheduler_type == 'warmup':
                optim_sched.step(epoch, args)
            
            loss_batch += loss.item()

            if len(new_traj)!=0:
                new_traj_all.extend(new_traj)

            cnt += 1
        log.add_scalar('train/loss', loss_batch/batch_count, epoch)

        model.eval()
        if epoch > args.epochs_thresh and flag_to_reset==1:
            reset_data(orig_loader_len, orig_loader_targets, train_set, new_traj_all, args)
        if epoch>args.epochs_thresh-1 and len(new_traj_all)!=0:
            add_data(train_set, new_traj_all, args, epoch, model)

        
        loss_batch = 0 
        batch_count = 0
        
        len_val = len(scene_norm_val_)
        indices_loop = np.linspace(0, len_val-1, len_val, dtype=int)


        random.shuffle(indices_loop)
        cnt = 0

        for ind in indices_loop: 

            start, end = val_set.seq_start_end[ind]
            batch_count+=1

            #Get data
            masked_obs_traj = masked_obs_full_val[:,start:end].unsqueeze(0).contiguous()
            obs_traj = obs_full_val[:,start:end].unsqueeze(0).contiguous()
            fut_traj = pred_full_val[:,start:end, :].unsqueeze(0).contiguous()
            pred_traj_gt_pos = pred_traj_val_[:,start:end, :].unsqueeze(0).contiguous()
            obs_traj_gt_pos_rel = obs_traj_full_val[:,start:end, :].unsqueeze(0).contiguous()
            scene_norm = scene_norm_val_[ind].unsqueeze(0).contiguous()
            flag_recon = flag_recon_val_[ind].unsqueeze(0).contiguous()

            pred_traj, pred_traj_infer, q_dist, p_dist, recon, recon_infer, p_masked, _,flag_to_reset = model(obs_traj, fut_traj, masked_obs_traj, obs_traj_gt_pos_rel, pred_traj_gt_pos,\
                                                                                 scene_norm, epoch, sample_full_loss,args,flag_recon, ind, 'val')


            loss = comp_loss(pred_traj, pred_traj_infer,pred_traj_gt_pos, q_dist, p_dist, p_masked, recon, recon_infer, obs_traj_gt_pos_rel, args, flag_recon, "val")
            loss_batch += loss.item()
            if cnt%1000 == 0:
                print('VALD:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count,'\t Sample:',cnt)

        log.add_scalar('val/loss', loss_batch/batch_count, epoch)
        torch.save(model.state_dict(),checkpoint_dir+'/val_best_%04d.pth' % epoch)  # OK
        
            
        if args.no_scheduler == False:
            if args.scheduler_type == 'steplr':
                optim_sched.step()
                for param_group in optimizer.param_groups:
                    print("lr is ",param_group['lr'])
                    log.add_scalar('train/lr', param_group['lr'], epoch)
            if args.scheduler_type == 'multisteplr':
                optim_sched.step()
                for param_group in optimizer.param_groups:
                    print("lr is ",param_group['lr'])
                    log.add_scalar('train/lr', param_group['lr'], epoch)
            elif args.scheduler_type == 'warmup':
                for param_group in optim_sched.optimizer.param_groups:
                    print("lr is ",param_group['lr'])
                    log.add_scalar('train/lr', param_group['lr'], epoch)

        else:
            for param_group in optimizer.param_groups:
                print("lr is ",param_group['lr'])
                log.add_scalar('train/lr', param_group['lr'], epoch)
        print(param_group['lr'])



       