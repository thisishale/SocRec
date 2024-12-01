import os
from utils import * 
from metrics import * 
import argparse
from model import *
from train_val import *
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
main_dir = os.path.dirname(__file__)
parser.add_argument('--input_size', type=int, default=5)
parser.add_argument('--input_size_f', type=int, default=4)
parser.add_argument('--input_size_decoder', type=int, default=2)
parser.add_argument('--output_size', type=int, default=2)
parser.add_argument('--output_size_recon', type=int, default=2)
parser.add_argument('--model_dim', type=int, default=256)
parser.add_argument('--min_clip_cvae', type=float, default=0)
parser.add_argument('--min_clip_vae', type=float, default=0)
parser.add_argument('--w_pred_kl', type=float, default=10)
parser.add_argument('--w_social', type=float, default=10)
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--sample_num', type=int, default=20)
parser.add_argument('--num_params', type=int, default=16)
parser.add_argument('--en_layers', type=int, default=1)
parser.add_argument('--dec_layers', type=int, default=1)
parser.add_argument('--dec_layers_recon', type=int, default=1)
parser.add_argument('--dim_feedforward', type=int, default=512)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--D', type=float, default=0.4)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--min_mean', type=int, default=0)
parser.add_argument('--k_recon', type=int, default=1)
parser.add_argument('--choice_recon', default='top', help='top or random')
parser.add_argument('--every_few', type=int, default=3)
parser.add_argument('--epochs_thresh', type=int, default=2,
                    help='number of epochs before adding')  
parser.add_argument('--compute_sample', default=False, action='store_true')
parser.add_argument('--no_scheduler', default=False, action='store_true')

parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')  
parser.add_argument('--dataset_folder',type=str,default='E:/Datasets/eth_debug_longer/')
   

parser.add_argument('--batch_size', type=int, default=1,
                    help='minibatch size')          
parser.add_argument('--num_epochs', type=int, default=150,
                    help='number of epochs')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--max_lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--initial_lr_', type=float, default=1e-5)
parser.add_argument('--warmup', type=float, default=50)
parser.add_argument('--step_size', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--sparsity', type=float, default=0.3)
parser.add_argument('--name', default='1',
                    help='model name')
parser.add_argument('--scheduler_type', default='steplr',
                    help='scheduler type')

                    
args = parser.parse_args()


print('*'*30)
print("Training initiating....")
print(args)


#Data prep     
data_set = os.path.join(args.dataset_folder, args.dataset)
dset_train = TrajectoryDataset(
        args,
        os.path.join(data_set, "train"),
        obs_len=args.obs_seq_len,
        pred_len=args.pred_seq_len,
        skip=1, phase="train")

dset_val = TrajectoryDataset(
        args,
        os.path.join(data_set, "val"),
        obs_len=args.obs_seq_len,
        pred_len=args.pred_seq_len,
        skip=1, phase="val")



#Defining the model 
log=SummaryWriter(os.path.join(main_dir,'logs',args.dataset,args.name))



print('Training started ...')
train_val(dset_train, dset_val, args, log)
    




