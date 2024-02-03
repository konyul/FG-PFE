import torch
from torch import nn
import torch_scatter
import time
class DSP(nn.Module):
    def __init__(self,
                 model_cfg):
        super().__init__()
        num_point_features = model_cfg.num_point_features
        self.point_cloud_range = model_cfg.get("point_cloud_range")
        self.voxel_size = model_cfg.get("voxel_size")
        self.grid_size = model_cfg.get("grid_size")
        self.x_offset = model_cfg.get("x_offset")
        self.y_offset = model_cfg.get("y_offset")
        self.z_offset = model_cfg.get("z_offset")
        self.voxel_x = model_cfg.get("voxel_x")
        self.voxel_y = model_cfg.get("voxel_y")
        self.consecutive = model_cfg.get("consecutive", False)
        self.parallel = model_cfg.get("parallel", False)
        self.z_unlimit = model_cfg.get("z_unlimit")
        self.use_cluster_xyz = model_cfg.get("use_cluster_xyz")
        self.voxel_channel = model_cfg.get("voxel_channel", False)
        self.residual = model_cfg.get("residual", False)
        self.reidx = model_cfg.get("reidx",False)
        channel_list = [[num_point_features,32],\
                        [num_point_features+32*2,self.voxel_channel],\
                        [num_point_features+self.voxel_channel*2,self.voxel_channel],\
                        [self.voxel_channel*2, 32]]
        if self.consecutive:
            self.blocks = nn.ModuleList()
            for module_idx in range(len(self.consecutive) + 1):
                if module_idx == len(self.consecutive):
                    module_idx = -1
                input_features = channel_list[module_idx][0]
                output_features = channel_list[module_idx][1]
                self.blocks.append(nn.Sequential(
                    nn.Linear(input_features, output_features, bias=False),
                    nn.BatchNorm1d(output_features, eps=1e-3, momentum=0.01),
                    nn.ReLU()))
        elif self.parallel:
            self.blocks = nn.ModuleList()
            for module_idx in range(2):
                output_features = self.voxel_channel
                input_features = num_point_features if module_idx == 0 else output_features*2*len(self.parallel)
                self.blocks.append(nn.Sequential(
                    nn.Linear(input_features, output_features, bias=False),
                    nn.BatchNorm1d(output_features, eps=1e-3, momentum=0.01),
                    nn.ReLU()))

    def forward(self, batch_dict, f_center, unq_inv, f_cluster=None, f_relative=None):
        points = batch_dict['points']
        points_coords_3d = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        if self.consecutive:
            features = None
            for module_idx, module in enumerate(self.consecutive):
                func = getattr(self, module)
                if module == 'standard':
                    features = func(points, f_center, f_cluster, f_relative, unq_inv, features, module_idx)    
                else:
                    features = func(points, points_coords_3d, features, batch_dict, module_idx)
                    if self.residual and module_idx == self.residual[0]:
                        residual_feature = features
                    if self.residual and module_idx == self.residual[1]:
                        features = torch.add(residual_feature,features)
            final_features = features
        if self.parallel:
            final_features = []
            for module_idx, module in enumerate(self.parallel):
                func = getattr(self, module)
                if module == 'standard':
                    final_features.append(func(points, f_center, f_cluster, f_relative, unq_inv, None, 0, batch_dict))
                else:
                    final_features.append(func(points, points_coords_3d, None, batch_dict, 0))
            final_features = torch.cat(final_features, dim=1)
        output = self.blocks[-1](final_features)
        batch_dict['dsp_feat'] = output
        output = torch_scatter.scatter_max(output, unq_inv, dim=0)[0]
        return output
    
    def gen_feat(self, points, f_center, f_cluster, f_relative, unq_inv, batch_dict, prev_features=None, idx=False, type=None):
        features = [f_center]
        features.append(points[:, 1:])
        features.append(f_cluster)
        features.append(f_relative)
        features = torch.cat(features,dim=-1).contiguous()
        features = self.blocks[idx](features)
        x_mean = torch_scatter.scatter_mean(features, unq_inv, dim=0)
        if type == 'standard':
            batch_dict['x_mean'] = x_mean
        features = torch.cat([features, x_mean[unq_inv, :]], dim=1)
        return features

    def upsample(self, points, points_coords_3d, features, batch_dict, module_idx=None):
        downsample_level = 1/2
        voxel_size = self.voxel_size[[0, 1]] * downsample_level
        grid_size = (self.grid_size / downsample_level).long()
        scale_xy = grid_size[0] * grid_size[1]
        scale_y = grid_size[1]
        voxel_x = voxel_size[0]
        voxel_y = voxel_size[1]
        x_offset = voxel_x / 2 + self.point_cloud_range[0]
        y_offset = voxel_y / 2 + self.point_cloud_range[1]
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / voxel_size).int()
        if self.z_unlimit:
            mask = ((points_coords >= 0) & (points_coords < grid_size[[0, 1]])).all(dim=1)
        else:
            points_z = points_coords_3d[:,-1:]
            mask = ((points_coords >= 0) & (points_coords < grid_size[[0, 1]]) & (points_z >= 0 ) & (points_z < self.grid_size[-1])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * scale_xy + \
                    points_coords[:, 0] * scale_y + \
                    points_coords[:, 1]

        _, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)

        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_x + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_y + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_cluster_xyz:
            points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
            f_cluster = points_xyz - points_mean[unq_inv, :]
        else:
            f_cluster = None
        features = self.gen_feat(points, f_center, f_cluster, unq_inv, features, module_idx)
        return features
    
    def downsample(self, points, points_coords_3d, features, batch_dict, module_idx=None):
        downsample_level = 2
        voxel_size = self.voxel_size[[0, 1]] * downsample_level
        grid_size = (self.grid_size / downsample_level).long()
        scale_xy = grid_size[0] * grid_size[1]
        scale_y = grid_size[1]
        voxel_x = voxel_size[0]
        voxel_y = voxel_size[1]
        x_offset = voxel_x / 2 + self.point_cloud_range[0]
        y_offset = voxel_y / 2 + self.point_cloud_range[1]
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / voxel_size).int()
        if self.z_unlimit:
            mask = ((points_coords >= 0) & (points_coords < grid_size[[0, 1]])).all(dim=1)
        else:
            points_z = points_coords_3d[:,-1:]
            mask = ((points_coords >= 0) & (points_coords < grid_size[[0, 1]]) & (points_z >= 0 ) & (points_z < self.grid_size[-1])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * scale_xy + \
                    points_coords[:, 0] * scale_y + \
                    points_coords[:, 1]

        _, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)

        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_x + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_y + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_cluster_xyz:
            points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
            f_cluster = points_xyz - points_mean[unq_inv, :]
        else:
            f_cluster = None
        features = self.gen_feat(points, f_center, f_cluster, unq_inv, features, module_idx)
        return features

    def shift(self, points, points_coords_3d, features, batch_dict, module_idx=None):
        if 'shift_data' not in batch_dict:
            shifted_point_cloud_range = self.point_cloud_range[[0,1]] + self.voxel_size[[0,1]] / 2
            points_coords = (torch.floor((points[:, [1, 2]] - shifted_point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]) + 1).int()
            #points_z = points_coords_3d[:,-1:]
            #mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]] + 1) & (points_z >= 0 ) & (points_z < self.grid_size[-1])).all(dim=1)
            mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]] + 1)).all(dim=1)
            points = points[mask]
            points_coords = points_coords[mask]
            points_xyz = points[:, [1, 2, 3]].contiguous()
            shifted_scale_xy = (self.grid_size[0] + 1) * (self.grid_size[1] + 1)
            shifted_scale_y = (self.grid_size[1] + 1)
            merge_coords = points[:, 0].int() * shifted_scale_xy + \
                        points_coords[:, 0] * shifted_scale_y + \
                        points_coords[:, 1]

            _, unq_inv, _ = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

            f_center = torch.zeros_like(points_xyz)
            
            f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
            f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
            f_center[:, 2] = points_xyz[:, 2] - self.z_offset
            
            if self.use_cluster_xyz:
                points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
                f_cluster = points_xyz - points_mean[unq_inv, :]
            else:
                f_cluster = None
            f_relative = points_xyz - torch.cat([shifted_point_cloud_range,self.point_cloud_range[2:3]],dim=0)
            batch_dict['shift_data'] = [points, f_center, f_cluster, unq_inv]
        else:
            points, f_center, f_cluster, unq_inv = batch_dict['shift_data']
        features = self.gen_feat(points, f_center, f_cluster, f_relative, unq_inv, batch_dict, features, module_idx)
        return features

    def standard(self, points, f_center, f_cluster, f_relative, unq_inv, features, module_idx, batch_dict):
        features = self.gen_feat(points, f_center, f_cluster, f_relative, unq_inv, batch_dict, features, module_idx, 'standard')
        return features
    
    def to_dense_batch(self, x, pillar_idx, max_points, max_pillar_idx):
        r"""
        Point sampling according to pillar index with constraint amount
        """
        
        # num_points in pillars (0 for empty pillar)
        num_nodes = torch_scatter.scatter_add(pillar_idx.new_ones(x.size(0)), pillar_idx, dim=0,
                                dim_size=max_pillar_idx)
        cum_nodes = torch.cat([pillar_idx.new_zeros(1), num_nodes.cumsum(dim=0)])

        # check if num_points in pillars exceed the predefined num_points value
        filter_nodes = False
        if num_nodes.max() > max_points:
            filter_nodes = True
        tmp = torch.arange(pillar_idx.size(0), device=x.device) - cum_nodes[pillar_idx]
        if filter_nodes:
            mask = tmp < max_points
            x = x[mask]
            pillar_idx = pillar_idx[mask]
        return x, pillar_idx


def build_dsp(model_cfg, model_name='DSP'):
    model_dict = {
        'DSP': DSP
}
    model_class = model_dict[model_name]

    model = model_class(model_cfg
                        )
    return model
