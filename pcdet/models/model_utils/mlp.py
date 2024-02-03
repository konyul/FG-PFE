
import argparse
import copy
import math

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn
import time
from .cbam import ZBAM
import torch_scatter

class bin_shuffle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_channels, in_channels//2, bias=False),
            nn.BatchNorm1d(in_channels//2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(in_channels//2, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU())

    def forward(self, x):
        return self.conv(x)



class conv1d(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class ZcCBAM(nn.Module):
    def __init__(self,
                 model_cfg):
        super().__init__()
        self.in_channels = model_cfg.input_channel
        self.out_channels = model_cfg.output_channel
        self.num_bins = model_cfg.num_bins
        self.zbam = ZBAM(model_cfg.output_channel)
        self.bin_shuffle = bin_shuffle((self.in_channels)*model_cfg.num_bins, model_cfg.output_channel)
        self.up_dimension = conv1d(input_dim = 5, hidden_dim = int(model_cfg.output_channel/2), output_dim = model_cfg.output_channel, num_layers = 2)
        self.residual = model_cfg.get("z_residual", False)

    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['v_feat_unq_coords'], data_dict['v_feat_unq_inv'], data_dict['v_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1] + 1))
        src[unq_inv, voxel_coords[:, 1], 1:] = voxels
        src[:, :, 0] = -1
        src[unq_inv, voxel_coords[:, 1], 0] = voxel_coords[:,0].float()
        occupied_mask = unq_cnt >=2 
        if 'mask_position' in data_dict:
            occupied_mask = data_dict['mask_position']
            occupied_mask = occupied_mask[data_dict['zfilter']]
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        data_dict['mlp_bxyz'] = src[:,:,:4]
        src = src[:,:,1:]
        src = self.up_dimension(src)
        src = src.permute(0,2,1).contiguous()
        src = self.zbam(src)
        data_dict['mlp_feat'] = src.permute(0,2,1).contiguous()
        N,P,C = src.shape
        src = src.view(N,P*C)
        src = self.bin_shuffle(src)
        if self.residual:
            pfe_src = data_dict['x_mean'][occupied_mask]
            src = torch.cat([src, pfe_src],dim=-1)
        return src, occupied_mask, None, None

class HCBAM(ZcCBAM):
    def __init__(self,
                 model_cfg):
        super().__init__(model_cfg=model_cfg)
        self.feature_fusion = model_cfg.get("feature_fusion",'sum')
        self.bin_shuffle_t = bin_shuffle((5)*10, model_cfg.output_channel)
        self.residual = model_cfg.get("z_residual", False)
    
    def binning_t(self, data_dict):
        voxels, voxel_coords = data_dict['toxel_features'], data_dict['toxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['t_feat_unq_coords'], data_dict['t_feat_unq_inv'], data_dict['t_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], 10, voxels.shape[1] + 1))
        src[unq_inv, voxel_coords[:, 1], 1:] = voxels
        src[:, :, 0] = -1
        src[unq_inv, voxel_coords[:, 1], 0] = voxel_coords[:,0].float()
        occupied_mask = unq_cnt >=2
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src_t, occupied_mask_t = self.binning_t(data_dict)
        src = src[occupied_mask]
        data_dict['mlp_bxyz'] = src[:,:,:4]
        src = src[..., 1:]
        src = self.up_dimension(src)
        src = src.permute(0,2,1).contiguous()
        src = self.zbam(src)
        data_dict['mlp_feat'] = src.permute(0,2,1).contiguous()
        N, C, Z = src.shape
        src = src.view(N, Z*C)
        src = self.bin_shuffle(src)
        src_t = src_t[occupied_mask_t]
        data_dict['mlp_bxyt'] = src_t[:,:,:4]
        src_t = src_t[..., 1:].contiguous()
        data_dict['mlp_feat_t'] = src_t
        N,T,C = src_t.shape
        src_t = src_t.view(N,T*C)
        src_t = self.bin_shuffle_t(src_t)
        return src, occupied_mask, src_t, occupied_mask_t


def build_mlp(model_cfg, model_name='ZCONV'):
    model_dict = {
        'HCBAM': HCBAM
}
    model_class = model_dict[model_name]

    model = model_class(model_cfg
                        )
    return model
