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

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder,self).__init__()
        self.model_dim = args.model_dim
        self.dropout = args.dropout
        self.d_model = args.model_dim
        self.dim_feedforward = args.dim_feedforward
        self.nhead = args.nhead
        self.input_size = args.input_size
        
        self.input_fc = nn.Linear(args.input_size, self.model_dim)
        # pos encoding from agentformer
        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=True, max_a_len=128, use_agent_enc=False, agent_enc_learn=False)
        self.en_layers = args.en_layers
        self.self_attn = AgentAwareAttention(self.d_model, self.nhead, dropout=self.dropout)
        self.linear1 = Linear(self.d_model, self.dim_feedforward)
        self.dropout0 = Dropout(self.dropout)
        self.linear2 = Linear(self.dim_feedforward, self.d_model)

        self.norm1 = LayerNorm(self.d_model)
        self.norm2 = LayerNorm(self.d_model)
        self.dropout1 = Dropout(self.dropout)
        self.dropout2 = Dropout(self.dropout)

        self.activation = F.relu

    def forward(self,v):
        agent_num = v.shape[2]
        v = self.input_fc(v.view(-1, v.shape[-1])).view(-1, 1, self.model_dim)
        v_pos = self.pos_encoder(v, num_a=agent_num, agent_enc_shuffle=None)
        src_agent_mask = torch.zeros((agent_num,agent_num),device=v.device)
        src_mask = generate_mask(v.shape[0], v.shape[0], agent_num, src_agent_mask).to(v.device)
        for mod in range(self.en_layers):
            v_pos2 = self.self_attn(v_pos, v_pos, v_pos, attn_mask=src_mask,
                              key_padding_mask=None, num_agent=agent_num)[0]
            v_pos = v_pos + self.dropout1(v_pos2)
            v_pos = self.norm1(v_pos)
            v_pos2 = self.linear2(self.dropout0(self.activation(self.linear1(v_pos))))
            v_pos = v_pos + self.dropout2(v_pos2)
            v_pos = self.norm2(v_pos)
        v = v_pos.view(-1, agent_num, self.model_dim)
        v_pooled = torch.mean(v, dim=0)
        return v_pos, v_pooled