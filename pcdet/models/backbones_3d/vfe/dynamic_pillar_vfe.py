import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from pcdet.models.model_utils.senet import SENet
from pcdet.models.model_utils.mlp import build_mlp
from pcdet.models.model_utils.dsp import build_dsp
class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class DynamicPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]
        
        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)
        # features = self.linear1(features)
        # features_max = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        # features = torch.cat([features, features_max[unq_inv, :]], dim=1)
        # features = self.linear2(features)
        # features = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        
        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                   (unq_coords % self.scale_xy) // self.scale_y,
                                   unq_coords % self.scale_y,
                                   torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                   ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['voxel_features'] = batch_dict['pillar_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict


class DynamicPillarVFESimple2D(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        # self.use_cluster_xyz = self.model_cfg.get('USE_CLUSTER_XYZ', True)
        if self.use_absolute_xyz:
            num_point_features += 3
        # if self.use_cluster_xyz:
        #     num_point_features += 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size[:2]).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        features = [f_center]
        if self.use_absolute_xyz:
            features.append(points[:, 1:])
        else:
            features.append(points[:, 4:])

        # if self.use_cluster_xyz:
        #     points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        #     f_cluster = points_xyz - points_mean[unq_inv, :]
        #     features.append(f_cluster)

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        pillar_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        pillar_coords = pillar_coords[:, [0, 2, 1]]

        batch_dict['pillar_features'] = features
        batch_dict['pillar_coords'] = pillar_coords
        return batch_dict


class FineGrainedPFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.use_cluster_xyz = self.model_cfg.get('USE_CLUSTER_XYZ', False)
        self.use_relative_xyz = self.model_cfg.get('USE_RELATIVE_XYZ', True)
        if self.use_relative_xyz:
            num_point_features += 3
        if self.use_absolute_xyz:
            num_point_features += 3
        if self.use_cluster_xyz:
            num_point_features += 3
        if self.with_distance:
            num_point_features += 1
        self.num_point_features = num_point_features
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        self.zpillar = model_cfg.get("ZPILLAR", None)
        self.numbins = int(8 / voxel_size[2])
        model_cfg.ZPILLAR_CFG.update({"num_bins": self.numbins})
        self.zpillar_model = build_mlp(model_cfg.ZPILLAR_CFG, model_name = self.zpillar)
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        self.dsp = model_cfg.get("DSP", None)
        model_cfg.DSP_CFG.update({"num_point_features":num_point_features,"use_cluster_xyz":self.use_cluster_xyz, \
            "point_cloud_range": self.point_cloud_range, "voxel_size": self.voxel_size,\
                "grid_size": self.grid_size, "x_offset": self.x_offset, "y_offset": self.y_offset, "z_offset": self.z_offset,\
                    "voxel_x": self.voxel_x, "voxel_y": self.voxel_y})
        self.dsp_model = build_dsp(model_cfg.DSP_CFG, model_name=self.dsp)
        self.point_cloud_range_t = torch.tensor([[0.01]]).cuda()
        self.temp_size = torch.tensor([[0.05]]).cuda()
        self.grid_t = torch.tensor(10).cuda()
        self.scale_xyt = grid_size[0] * grid_size[1] * self.grid_t
        self.scale_yt = grid_size[1] * self.grid_t
        self.scale_t = self.grid_t
        self.mlp = nn.Sequential(
                    nn.Linear(96, 32, bias=False),
                    nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                    nn.ReLU())
        self.senet = SENet(96, 8)
        self.double_flip = model_cfg.get("DOUBLE_FLIP", False)


    def get_output_feature_dim(self):
        return self.num_filters[-1]
    
    def dyn_voxelization(self, points, point_coords, batch_dict):
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                        point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        batch_dict['v_unq_inv'] = unq_inv
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_features_coords'] = voxel_coords.contiguous()
        return batch_dict
    
    def dyn_voxelization_t(self, points, point_coords, batch_dict):
        merge_coords = points[:, 0].int() * self.scale_xyt + \
                        point_coords[:, 0] * self.scale_yt + \
                        point_coords[:, 1] * self.scale_t + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyt,
                                    (unq_coords % self.scale_xyt) // self.scale_yt,
                                    (unq_coords % self.scale_yt) // self.scale_t,
                                    unq_coords % self.scale_t), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        batch_dict['t_unq_inv'] = unq_inv
        batch_dict['toxel_features'] = points_mean.contiguous()
        batch_dict['toxel_features_coords'] = voxel_coords.contiguous()
        return batch_dict
    
    def forward(self, batch_dict, **kwargs):
        if self.double_flip:
            batch_dict['batch_size'] = batch_dict['batch_size']*4
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)
        points_coords_3d = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        points_coords = points_coords_3d[:,:2]
        points_coords_t = torch.round((points[:, [5]] - self.point_cloud_range_t) / self.temp_size).int()
        points_coords_t[points_coords_t[:,-1]>9] = 9
        points_coords_t = torch.cat([points_coords, points_coords_t],dim=-1)
        mask3d = ((points_coords_3d >= 0) & (points_coords_3d < self.grid_size)).all(dim=1)
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        zfilter = (mask3d==mask)
        points_coords_3d = points_coords_3d[mask3d]
        points3d = points[mask3d]
        points_coords_t = points_coords_t[mask]
        points = points[mask]
        points_coords = points_coords[mask]

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)
        points_xyz = points[:, [1, 2, 3]].contiguous()
        zfilter = torch_scatter.scatter_max(zfilter.int(), unq_inv, dim=0)[0]
        batch_dict['zfilter'] = zfilter = zfilter.bool()
        batch_dict['points'] = points
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        
        features = [f_center]
        if self.use_absolute_xyz:
            features.append(points[:, 1:])
        else:
            features.append(points[:, 4:])
            
        if self.use_cluster_xyz:
            points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
            f_cluster = points_xyz - points_mean[unq_inv, :]
            features.append(f_cluster)
        else:
            f_cluster = None

        if self.use_relative_xyz:
            f_relative = points_xyz - self.point_cloud_range[:3]
            features.append(f_relative)
        
        features = torch.cat(features, dim=-1)
        features = self.dsp_model(batch_dict, f_center, unq_inv, f_cluster, f_relative)
        batch_dict = self.dyn_voxelization(points3d, points_coords_3d, batch_dict)
        batch_dict = self.dyn_voxelization_t(points, points_coords_t, batch_dict)
        batch_dict['pillar_merge_coords'] = merge_coords
        batch_dict['unq_inv'] = unq_inv
        batch_dict['point_cloud_range'] = self.point_cloud_range
        batch_dict['voxel_size'] = self.voxel_size
        batch_dict['grid_size'] = self.grid_size
        voxel_features, voxel_features_coords = batch_dict['voxel_features'], batch_dict['voxel_features_coords']
        v_feat_coords = voxel_features_coords[:, 0] * self.scale_xy + voxel_features_coords[:, 3] * self.scale_y + voxel_features_coords[:, 2]
        v_feat_unq_coords, v_feat_unq_inv, v_feat_unq_cnt = torch.unique(v_feat_coords, return_inverse=True, return_counts=True, dim=0)
        batch_dict['v_feat_unq_coords'] = v_feat_unq_coords
        batch_dict['v_feat_unq_inv'] = v_feat_unq_inv
        batch_dict['voxel_features'] = voxel_features
        batch_dict['v_feat_unq_cnt'] = v_feat_unq_cnt
        toxel_features, toxel_features_coords = batch_dict['toxel_features'], batch_dict['toxel_features_coords']
        t_feat_coords = toxel_features_coords[:, 0] * self.scale_xy + toxel_features_coords[:, 3] * self.scale_y + toxel_features_coords[:, 2]
        t_feat_unq_coords, t_feat_unq_inv, t_feat_unq_cnt = torch.unique(t_feat_coords, return_inverse=True, return_counts=True, dim=0)
        batch_dict['t_feat_unq_coords'] = t_feat_unq_coords
        batch_dict['t_feat_unq_inv'] = t_feat_unq_inv
        batch_dict['toxel_features'] = toxel_features
        batch_dict['t_feat_unq_cnt'] = t_feat_unq_cnt
        z_pillar_feat, occupied_mask, t_pillar_feat, occupied_mask_t = self.zpillar_model(batch_dict)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords * 0),
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        voxel_coords = voxel_coords[:, [0, 1, 3, 2]]
        pillar_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        pillar_coords = pillar_coords[:, [0, 2, 1]]
        p_num = features.shape[0]
        
        t_feat = t_pillar_feat.new_zeros((p_num,t_pillar_feat.shape[1]))
        t_feat[occupied_mask_t] = t_pillar_feat
        
        p_feat = z_pillar_feat.new_zeros((p_num,z_pillar_feat.shape[1]))
        z_feat = p_feat[zfilter]
        z_feat[occupied_mask] = z_pillar_feat
        p_feat[zfilter] = z_feat
        
        features = torch.cat([features,p_feat,t_feat],dim=1)
        residual = features
        features = self.senet(features)
        features += residual
        features = self.mlp(features)
        batch_dict['pillar_features'] = features
        batch_dict['pillar_coords'] = pillar_coords
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict