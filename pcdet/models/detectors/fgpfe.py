from .detector3d_template import Detector3DTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.models.backbones_3d.focal_sparse_conv.focal_sparse_utils import FocalLoss
import torch
import torch.nn as nn
class FGPFE(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_cfg = model_cfg
        self.guide_loss = self.model_cfg.get('LOSS', False)
        if self.guide_loss:
            self.waymo = self.model_cfg['LOSS'].get('waymo',False)
            self.loss_type = self.model_cfg['LOSS'].get('loss_type',False)
            self.enlarged_bbox = self.model_cfg['LOSS'].get('enlarged_bbox',False)
            self.dsp_loss = self.model_cfg['LOSS'].get('dsp_loss',False)
            self.mlp_loss = self.model_cfg['LOSS'].get('mlp_loss',False)
            self.zbam_loss = self.model_cfg['LOSS'].get('zbam_loss',False)
            self.all_loss = self.model_cfg['LOSS'].get('all_loss',False)
            self.weight = self.model_cfg['LOSS']['weight']
            in_channel = self.model_cfg['VFE']['NUM_FILTERS'][0]
            if self.loss_type in ['bce', 'gaussian', 'focal']:
                self.mlp = nn.Sequential(nn.Linear(in_channel, 1, bias=False),
                                        nn.Sigmoid())
            if self.loss_type in ['bce', 'gaussian']:
                self.loss = nn.BCELoss()
            elif self.loss_type == 'focal':
                self.loss = FocalLoss()
            self.offset = -53.9625
            self.voxel_size = 0.075

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_module_loss(self, batch_dict):
        if self.waymo:
            gt_boxes = batch_dict['gt_boxes'][:,:,:-1]
        else:
            gt_boxes = batch_dict['gt_boxes'][:,:,:-3]
        B = len(batch_dict['frame_id'])
        
        # for dsp
        if self.dsp_loss:
            dsp_feat = batch_dict['dsp_feat']
            dsp_xyz = batch_dict['points'][:,[1,2,3]]
            dsp_batch = batch_dict['points'][:,0]
        
        # for mlp
        if self.mlp_loss:
            mlp_feat = batch_dict['mlp_feat']
            mlp_xyz = batch_dict['mlp_bxyz'][:,:,[1,2,3]]
            mlp_batch = batch_dict['mlp_bxyz'][:,:,0]

        # for zbam
        if self.zbam_loss:
            zbam_feat = batch_dict['zbam_feat']
            zbam_xyz = batch_dict['zbam_bxyz'][:,:,[1,2,3]]
            zbam_batch = batch_dict['zbam_bxyz'][:,:,0]
        
        if self.all_loss:
            all_feat = batch_dict['all_feat'].squeeze(-1)
            all_xyz = batch_dict['all_coords'][:,1:]
            all_xyz = all_xyz * self.voxel_size + self.offset
            all_xyz = all_xyz[:,[1,0]]
            all_batch = batch_dict['all_coords'][:,0]

        valid_dsp_xyz = []
        valid_all_xyz = []
        valid_mlp_xyz = []
        valid_mlp_feat = []
        valid_zbam_xyz = []
        valid_zbam_feat = []
        if self.enlarged_bbox:
            gt_boxes[:,:,3:6] *= self.enlarged_bbox
        for b in range(B):
            batch_gt_boxes = gt_boxes[b:b+1]
            batch_gt_boxes_mask = (batch_gt_boxes.sum(-1)!=0)
            batch_gt_boxes = batch_gt_boxes[batch_gt_boxes_mask][None,:]
            # for dsp
            if self.dsp_loss:
                mask = dsp_batch==b
                batch_dsp_xyz = dsp_xyz[mask]
                if self.loss_type == 'bce':
                    valid_batch_dsp_xyz = roiaware_pool3d_utils.points_in_boxes_gpu(batch_dsp_xyz[None,:], batch_gt_boxes)[0]
                elif self.loss_type =='focal':
                    valid_batch_dsp_xyz = roiaware_pool3d_utils.points_in_boxes_gpu(batch_dsp_xyz[None,:], batch_gt_boxes)[0]
                    valid_batch_dsp_xyz = (valid_batch_dsp_xyz>=0)
                elif self.loss_type == 'gaussian':
                    valid_batch_dsp_xyz = roiaware_pool3d_utils.points_in_gaussian_boxes_gpu(batch_dsp_xyz[None,:], batch_gt_boxes)[0]
                valid_dsp_xyz.append(valid_batch_dsp_xyz)
            
            if self.all_loss:
                mask = all_batch==b
                batch_all_xyz = all_xyz[mask]
                if self.loss_type in ['bce', 'focal']:
                    valid_batch_all_xyz = roiaware_pool3d_utils.points_in_boxes_gpu(batch_all_xyz[None,:], batch_gt_boxes)[0]
                elif self.loss_type == 'gaussian':
                    temp_gt_boxes = batch_gt_boxes.clone()
                    temp_gt_boxes[:,:,[2]] = -1 
                    temp_gt_boxes[:,:,[5]] = 8
                    all_z = torch.full((batch_all_xyz.shape[0],1),0).cuda()
                    batch_all_xyz = torch.cat([batch_all_xyz,all_z],dim=-1)
                    valid_batch_all_xyz = roiaware_pool3d_utils.points_in_gaussian_boxes_gpu(batch_all_xyz[None,:], temp_gt_boxes)[0]
                valid_all_xyz.append(valid_batch_all_xyz)
            
            # for mlp
            if self.mlp_loss:
                mask = mlp_batch==b
                batch_mlp_xyz = mlp_xyz[mask]
                batch_mlp_feat = mlp_feat[mask]
                if self.loss_type == 'bce':
                    valid_batch_mlp_xyz = roiaware_pool3d_utils.points_in_boxes_gpu(batch_mlp_xyz[None,:], batch_gt_boxes)[0]
                elif self.loss_type == 'focal':
                    valid_batch_mlp_xyz = roiaware_pool3d_utils.points_in_boxes_gpu(batch_mlp_xyz[None,:], batch_gt_boxes)[0]
                    valid_batch_mlp_xyz = (valid_batch_mlp_xyz>=0)
                elif self.loss_type == 'gaussian':
                    valid_batch_mlp_xyz = roiaware_pool3d_utils.points_in_gaussian_boxes_gpu(batch_mlp_xyz[None,:], batch_gt_boxes)[0]
                valid_mlp_xyz.append(valid_batch_mlp_xyz)
                valid_mlp_feat.append(batch_mlp_feat)
                
            # for zbam
            if self.zbam_loss:
                mask = zbam_batch==b
                batch_zbam_xyz = zbam_xyz[mask]
                batch_zbam_feat = zbam_feat[mask]
                if self.loss_type in ['bce', 'focal']:
                    valid_batch_zbam_xyz = roiaware_pool3d_utils.points_in_boxes_gpu(batch_zbam_xyz[None,:], batch_gt_boxes)[0]
                elif self.loss_type == 'gaussian':
                    valid_batch_zbam_xyz = roiaware_pool3d_utils.points_in_gaussian_boxes_gpu(batch_zbam_xyz[None,:], batch_gt_boxes)[0]
                valid_zbam_xyz.append(valid_batch_zbam_xyz)
                valid_zbam_feat.append(batch_zbam_feat)
            
        loss = 0
        # for dsp
        if self.dsp_loss:
            valid_dsp_xyz = torch.cat(valid_dsp_xyz).to(dsp_feat.dtype)
            if self.loss_type == 'bce':
                valid_dsp_target = (valid_dsp_xyz != -1).float()[:,None]
                dsp_attn = self.mlp(dsp_feat)
            elif self.loss_type == 'focal':
                valid_dsp_target = (valid_dsp_xyz != -1).float()[:,None]
                dsp_attn = self.mlp(dsp_feat)
                mask_voxels_two_classes = torch.cat([1-dsp_attn, dsp_attn], dim=1)
                dsp_attn = mask_voxels_two_classes
                valid_dsp_target = valid_dsp_xyz.long()
                
            elif self.loss_type == 'gaussian':
                valid_dsp_target = valid_dsp_xyz.unsqueeze(-1)
                dsp_attn = self.mlp(dsp_feat)
            dsp_loss = self.loss(dsp_attn, valid_dsp_target)
            loss += dsp_loss
        
        if self.all_loss:
            valid_all_xyz = torch.cat(valid_all_xyz).to(all_feat.dtype)
            if self.loss_type in ['bce', 'focal']:
                valid_all_target = (valid_all_xyz != -1).float()[:,None]
            elif self.loss_type == 'gaussian':
                valid_all_target = valid_all_xyz.unsqueeze(-1)
            all_attn = self.all_mlp(all_feat)
            all_loss = self.loss(all_attn, valid_all_target)
            loss += all_loss

        # for mlp
        if self.mlp_loss:
            valid_mlp_xyz = torch.cat(valid_mlp_xyz).to(mlp_feat.dtype)
            if self.loss_type == 'bce':
                valid_mlp_target = (valid_mlp_xyz != -1).float()[:,None]
                valid_mlp_feat = torch.cat(valid_mlp_feat).to(mlp_feat.dtype)
                mlp_attn = self.mlp(valid_mlp_feat)
            elif self.loss_type == 'focal':
                valid_mlp_target = (valid_mlp_xyz != -1).float()[:,None]
                valid_mlp_feat = torch.cat(valid_mlp_feat).to(mlp_feat.dtype)
                mlp_attn = self.mlp(valid_mlp_feat)
                mask_voxels_two_classes = torch.cat([1-mlp_attn, mlp_attn], dim=1)
                mlp_attn = mask_voxels_two_classes
                valid_mlp_target = valid_mlp_xyz.long()
            elif self.loss_type == 'gaussian':
                valid_mlp_target = valid_mlp_xyz.unsqueeze(-1)
                valid_mlp_feat = torch.cat(valid_mlp_feat).to(mlp_feat.dtype)
                mlp_attn = self.mlp(valid_mlp_feat)
            mlp_loss = self.loss(mlp_attn, valid_mlp_target)
            loss += mlp_loss

        # for zbam
        if self.zbam_loss:
            valid_zbam_xyz = torch.cat(valid_zbam_xyz).to(zbam_feat.dtype)
            if self.loss_type in ['bce', 'focal']:
                valid_zbam_target = (valid_zbam_xyz != -1).float()[:,None]
            elif self.loss_type == 'gaussian':
                valid_zbam_target = valid_zbam_xyz.unsqueeze(-1)
            valid_zbam_feat = torch.cat(valid_zbam_feat).to(zbam_feat.dtype)
            zbam_attn = self.mlp(valid_zbam_feat)
            zbam_loss = self.loss(zbam_attn, valid_zbam_target)
            loss += zbam_loss

        return loss
    
    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        if self.guide_loss:
            guide_loss = self.get_module_loss(batch_dict)
            loss += self.weight * guide_loss
            tb_dict.update({
                'guide_loss' : guide_loss.item(),
                **tb_dict
            })
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict