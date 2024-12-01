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

class F_Encoder(nn.Module):
    def __init__(self, args):
        super(F_Encoder,self).__init__()
        self.model_dim = args.model_dim
        self.dropout = args.dropout
        self.d_model = args.model_dim
        self.dim_feedforward = args.dim_feedforward
        self.nhead = args.nhead
        self.input_size = args.input_size_f
        self.future_frames = args.pred_seq_len
        self.past_frames = args.obs_seq_len
        self.num_params = args.num_params
        self.input_fc = nn.Linear(self.input_size, self.model_dim)
        # pos encoding from agentformer
        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=True, max_a_len=128, use_agent_enc=False, agent_enc_learn=False)
        self.dec_layers = args.dec_layers
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
        self.output_size = args.output_size
        self.out_fc = nn.Linear(self.out_mlp.out_dim, self.output_size)

    def forward(self,v,v_enc,scene_norm, sample_num):
        
        agent_num = v.shape[2]
        mem_agent_mask = torch.zeros((agent_num,agent_num),device=v.device)
        tgt_agent_mask = torch.zeros((agent_num,agent_num),device=v.device)
        self_attn_weights = [None] * self.dec_layers
        cross_attn_weights = [None] * self.dec_layers
        tf_in = self.input_fc(v.view(-1, v.shape[-1])).view(-1, 1, self.model_dim)
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=False, t_offset=0)
        mem_mask = generate_mask(tf_in.shape[0], v_enc.shape[0], agent_num, mem_agent_mask).to(tf_in.device)
        tgt_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)
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
        out_tmp = tf_out.view(v.shape[1], -1, self.model_dim)
        out_tmp = torch.mean(out_tmp, dim=0)
        h = self.out_mlp(out_tmp)  

        return h