import os
import math
import sys

import torch
import torch.nn as nn

from utils_model import *

import torch
from torch.nn import functional as F
from torch.nn.functional import *
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from AgentAwareAttention import *
from mlp import *
from dist import *

class Decoder_recon(nn.Module):
    def __init__(self, args):
        super(Decoder_recon,self).__init__()
        self.model_dim = args.model_dim
        self.dropout = args.dropout
        self.d_model = args.model_dim
        self.dim_feedforward = args.dim_feedforward
        self.nhead = args.nhead
        self.input_size = args.input_size
        self.future_frames = args.pred_seq_len
        self.past_frames = args.obs_seq_len
        self.num_params = args.num_params
        self.input_fc = nn.Linear(self.num_params+args.output_size_recon, self.model_dim)
        # pos encoding from agentformer
        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=True, max_a_len=128, use_agent_enc=False, agent_enc_learn=False)
        self.dec_layers = args.dec_layers_recon
        self.self_attn = AgentAwareAttention(self.d_model, self.nhead, dropout=self.dropout)
        self.multihead_attn = AgentAwareAttention(self.d_model, self.nhead, dropout=self.dropout)
        self.linear1 = Linear(self.d_model, self.dim_feedforward)
        self.dropout0 = Dropout(self.dropout)
        self.linear2 = Linear(self.dim_feedforward, self.d_model)

        self.norm1 = LayerNorm(self.d_model)
        self.norm2 = LayerNorm(self.d_model)
        self.norm3 = LayerNorm(self.d_model)
        self.dropout1 = Dropout(self.dropout)
        self.dropout2 = Dropout(self.dropout)
        self.dropout3 = Dropout(self.dropout)
        self.out_mlp_dim = [512,256]
        self.activation = F.relu
        in_dim = self.model_dim
        
        self.out_mlp = MLP(in_dim, self.out_mlp_dim, 'relu')
        self.output_size_recon = args.output_size_recon
        self.out_fc = nn.Linear(self.out_mlp.out_dim, args.output_size_recon)

    def forward(self,v, z,v_enc,scene_norm, sample_num):
        
        agent_num = v.shape[2]
        mem_agent_mask = torch.zeros((agent_num,agent_num),device=v.device)
        tgt_agent_mask = torch.zeros((agent_num,agent_num),device=v.device)
        self_attn_weights = [None] * self.dec_layers
        cross_attn_weights = [None] * self.dec_layers
        v = v.squeeze(0)
        v = v.repeat_interleave(sample_num,1)
        v_enc = v_enc.repeat_interleave(sample_num,1)

        z = z.view(-1, sample_num, z.shape[-1])
        
        dec_in = v[...,:2]
        dec_in = dec_in.view(-1, sample_num, dec_in.shape[-1])
        in_arr = [dec_in, z]
        dec_in_z = torch.cat(in_arr, dim=-1)
        tf_in = self.input_fc(dec_in_z.view(-1, dec_in_z.shape[-1])).view(dec_in_z.shape[0], -1, self.model_dim)
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=False, t_offset=0)
        mem_mask = generate_mask(tf_in.shape[0], v_enc.shape[0], agent_num, mem_agent_mask).to(tf_in.device)
        tgt_mask = generate_ar_mask(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)
        tgt = tf_in_pos
        for j in range(self.dec_layers):
            tgt2, self_attn_weights[j] = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                key_padding_mask=None, num_agent=agent_num, need_weights=False)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2, cross_attn_weights[j] = self.multihead_attn(tgt, v_enc, v_enc, attn_mask=mem_mask,
                                    key_padding_mask=None, num_agent=agent_num, need_weights=False)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout0(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)


        tf_out = tgt
        out_tmp = tf_out.view(-1, tf_out.shape[-1])
        out_tmp = self.out_mlp(out_tmp)  

        seq_out = self.out_fc(out_tmp).view(tf_out.shape[0], -1, self.output_size_recon)
        norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        seq_out = norm_motion + v[[0]][...,:2] 

        seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])
        
        seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        dec_motion = seq_out + scene_norm
        dec_motion = dec_motion.view(dec_motion.shape[0], dec_motion.shape[1]//sample_num, sample_num, dec_motion.shape[-1])
        dec_motion = dec_motion.permute(2,0,1,3).contiguous() 

        return dec_motion