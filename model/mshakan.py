from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from model.RevIN import RevIN
from model.KANS.hahn import HahnPolynomials


class HahnKANBlock(nn.Module):
    def __init__(self, dim, len):
        super().__init__()
        self.intrapatch_kan = HahnPolynomials(dim, dim, 3, 1, 1, 7)
        self.interpatch_kan = HahnPolynomials(len, len, 3, 1, 1, 7)

    def forward(self, x):
        x = self.intrapatch_kan(x)
        x = x.permute(0, 2, 1)
        x = self.interpatch_kan(x)
        x = x.permute(0, 2, 1)
        return x


class Flatten_Head(nn.Module):
    def __init__(self, nf, target_window):
        super().__init__()
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, 336)
        self.linear2 = nn.Linear(336, target_window)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class BackBone(nn.Module):
    def __init__(self, patch_num, patch_len, d_model=128):
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.seq_len = q_len
        self.W_pos = nn.Parameter(torch.randn(1, q_len, d_model))
        self.encoder = nn.ModuleList([HahnKANBlock(d_model, self.patch_num) for i in range(5)])
    
    def forward(self, x) -> Tensor:                                      # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)                                        # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = u + self.W_pos[:, :self.seq_len, :]

        z = u
        for layer in self.encoder:    
            x = layer(z)                                                 # z: [bs * nvars x patch_num x d_model]
            z = z + x

        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))               # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)                                        # z: [bs x nvars x d_model x patch_num]
        
        return z


class Model(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, 
                 d_model=128, padding_patch=None, revin=True, affine=True, subtract_last=False,
                 patch_len_2=8, stride_2=4):  # Added Scale 2 parameters for Multi-Scale Patching
        
        super().__init__()
        
        self.revin = revin
        if self.revin: 
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        self.padding_patch = padding_patch
        self.n_vars = c_in
        
        # ---------------------------------------------------------
        # BRANCH 1: Primary Scale (Macro/Mid-Term Trend)
        # ---------------------------------------------------------
        self.patch_len_1 = patch_len
        self.stride_1 = stride
        patch_num_1 = int((context_window - self.patch_len_1) / self.stride_1 + 1)
        if padding_patch == 'end':
            self.padding_patch_layer1 = nn.ReplicationPad1d((0, self.stride_1)) 
            patch_num_1 += 1
        
        self.backbone1 = BackBone(patch_num=patch_num_1, patch_len=self.patch_len_1, d_model=d_model)


        # BRANCH 2: Secondary Scale (Short-Term Tactical Noise)
        self.patch_len_2 = patch_len_2
        self.stride_2 = stride_2
        patch_num_2 = int((context_window - self.patch_len_2) / self.stride_2 + 1)
        if padding_patch == 'end':
            self.padding_patch_layer2 = nn.ReplicationPad1d((0, self.stride_2))
            patch_num_2 += 1
            
        self.backbone2 = BackBone(patch_num=patch_num_2, patch_len=self.patch_len_2, d_model=d_model)

        # MULTI-SCALE MERGE HEAD
     
        # The flattened features combine the outputs of both patch resolutions
        self.head_nf = (d_model * patch_num_1) + (d_model * patch_num_2)

        # Unused conv layers from the original implementation are removed for clarity
        self.head = Flatten_Head(self.head_nf, target_window)
        
    
    def forward(self, z):                                                       # z: [bs x nvars x seq_len]
        if self.revin: 
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

      
        # Process Branch 1
       
        z1 = z
        if self.padding_patch == 'end':
            z1 = self.padding_patch_layer1(z1)
        z1 = z1.unfold(dimension=-1, size=self.patch_len_1, step=self.stride_1) # [bs x nvars x patch_num_1 x patch_len_1]
        z1 = z1.permute(0, 1, 3, 2)                                             # [bs x nvars x patch_len_1 x patch_num_1]
        feat1 = self.backbone1(z1)                                              # [bs x nvars x d_model x patch_num_1]

        
        # Process Branch 2
        
        z2 = z
        if self.padding_patch == 'end':
            z2 = self.padding_patch_layer2(z2)
        z2 = z2.unfold(dimension=-1, size=self.patch_len_2, step=self.stride_2) # [bs x nvars x patch_num_2 x patch_len_2]
        z2 = z2.permute(0, 1, 3, 2)                                             # [bs x nvars x patch_len_2 x patch_num_2]
        feat2 = self.backbone2(z2)                                              # [bs x nvars x d_model x patch_num_2]

      
        # Concatenate and Predict
       
        multi_scale_features = torch.cat([feat1, feat2], dim=-1)                # [bs x nvars x d_model x (patch_num_1 + patch_num_2)]
        
        z_out = self.head(multi_scale_features)                                 # [bs x nvars x target_window] 
        
        if self.revin: 
            z_out = z_out.permute(0, 2, 1)
            z_out = self.revin_layer(z_out, 'denorm')
            
        return z_out