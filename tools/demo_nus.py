import argparse
import glob
from pathlib import Path


import open3d
from visual_utils import open3d_vis_utils as V
OPEN3D_FLAG = True
import numpy as np
import torch
import cv2

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import pickle
from visual_utils.simple_vis import nuscene_vis
from nuscenes.nuscenes import NuScenes
from pcdet.utils.box_utils import mask_boxes_outside_range_numpy
import shutil
import os
from pathlib import Path
import yaml
from easydict import EasyDict

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', scene_num=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext

        self.nusc = NuScenes(version='v1.0-trainval', dataroot='../data/nuscenes/v1.0-trainval', verbose=True)
        for scene in self.nusc.scene:
            if int(scene_num) == int(scene['name'].split("-")[-1]):
                scene_num = int(scene_num)
                my_scene = scene
        scene_name = my_scene['name']
        os.makedirs(scene_name, exist_ok=True)
        os.makedirs(os.path.join(scene_name,"cam_front_left"), exist_ok=True)
        os.makedirs(os.path.join(scene_name,"cam_front"), exist_ok=True)
        os.makedirs(os.path.join(scene_name,"cam_front_right"), exist_ok=True)
        os.makedirs(os.path.join(scene_name,"lidar"), exist_ok=True)
        os.makedirs(os.path.join(scene_name,"lidar_baseline"), exist_ok=True)

        first_sample_token = my_scene['first_sample_token']
        data_file_list = []        
        sample_token = first_sample_token
        gt_list = []
        for i in range(my_scene['nbr_samples']):
            sample_info = self.nusc.get("sample",sample_token)
            lidar_token = sample_info['data']['LIDAR_TOP']
            lidar_path = self.nusc.get_sample_data(lidar_token)[0]
            data_file_list.append(lidar_path)
            cam_front_left = sample_info['data']['CAM_FRONT_LEFT']
            cam_front = sample_info['data']['CAM_FRONT']
            cam_front_right = sample_info['data']['CAM_FRONT_RIGHT']
            cam_front_left_path = self.nusc.get_sample_data(cam_front_left)[0]
            cam_front_path = self.nusc.get_sample_data(cam_front)[0]
            cam_front_right_path = self.nusc.get_sample_data(cam_front_right)[0]
            stri = str(i).zfill(2)
            shutil.copy(cam_front_left_path,os.path.join(my_scene['name'],"cam_front_left", f'test_{stri}.png'))
            shutil.copy(cam_front_path,os.path.join(my_scene['name'],"cam_front", f'test_{stri}.png'))
            shutil.copy(cam_front_right_path,os.path.join(my_scene['name'],"cam_front_right", f'test_{stri}.png'))
            next_samp_token = sample_info['next']
            annotations = [self.nusc.get('sample_annotation', token) for token in sample_info['anns']]
            num_lidar_pts = np.array([anno['num_lidar_pts'] for anno in annotations])
            num_radar_pts = np.array([anno['num_radar_pts'] for anno in annotations])
            mask = (num_lidar_pts + num_radar_pts > 0) & (num_lidar_pts > 4)
            ref_lidar_path, ref_boxes, _ = self.nusc.get_sample_data(lidar_token)
            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([self.quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)
            assert len(annotations) == len(gt_boxes) == len(velocity)
            gt_boxes = gt_boxes[mask, :]
            gt_boxes = gt_boxes[:,:7]
            gt_boxes[:,-1] = -gt_boxes[:,-1]
            sample_token = next_samp_token
            point_cloud_range= [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            mask = mask_boxes_outside_range_numpy(
                gt_boxes, point_cloud_range, min_num_corners=1, 
                use_center_to_filter=True
            )
            gt_boxes = gt_boxes[mask]
            gt_list.append(gt_boxes)
        self.gt_list = gt_list
        self.sample_file_list = data_file_list
        self.scene_name = scene_name

    def quaternion_yaw(self, q):
        """
        Calculate the yaw angle from a quaternion.
        Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
        It does not work for a box in the camera frame.
        :param q: Quaternion of interest.
        :return: Yaw angle in radians.
        """
        # Project into xy plane.
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        # Measure yaw using arctan.
        yaw = np.arctan2(v[1], v[0])
        return yaw

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
            times = np.zeros([points.shape[0],1])
            points = np.concatenate((points, times),axis=1)

        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    
    parser.add_argument('--cfg_file_baseline', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt_baseline', type=str, default=None, help='specify the pretrained model')
    
    
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--scene_num', type=str, help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_baseline = EasyDict()
    cfg_baseline.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg_baseline.LOCAL_RANK = 0
    cfg_from_yaml_file(args.cfg_file_baseline, cfg_baseline)
    return args, cfg, cfg_baseline


def main():
    args, cfg, cfg_baseline = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger, scene_num=args.scene_num
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    scene_name = demo_dataset.scene_name

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    model_baseline = build_network(model_cfg=cfg_baseline.MODEL, num_class=len(cfg_baseline.CLASS_NAMES), dataset=demo_dataset)
    model_baseline.load_params_from_file(filename=args.ckpt_baseline, logger=logger, to_cpu=True)
    model_baseline.cuda()
    model_baseline.eval()
    
    
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            model.model_cfg.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = 0.4
            model_baseline.model_cfg.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = 0.4
            pred_dicts, _ = model(data_dict)
            pred_dicts_baseline, _ = model_baseline(data_dict)
            gt_boxes = demo_dataset.gt_list[idx]
            pred_dicts[0]['pred_boxes'][:,6] = -pred_dicts[0]['pred_boxes'][:,6]
            pred_dicts_baseline[0]['pred_boxes'][:,6] = -pred_dicts_baseline[0]['pred_boxes'][:,6]
            det = nuscene_vis(data_dict['points'][:, 1:4].detach().cpu().numpy(), pred_dicts[0]['pred_boxes'][:,:7].detach().cpu().numpy(), gt_boxes=gt_boxes) #xyz, dim, theta, vel
            det_baseline = nuscene_vis(data_dict['points'][:, 1:4].detach().cpu().numpy(), pred_dicts_baseline[0]['pred_boxes'][:,:7].detach().cpu().numpy(), gt_boxes=gt_boxes) #xyz, dim, theta, vel
            stridx = str(idx).zfill(2)
            cv2.imwrite(os.path.join(scene_name,'lidar',f'test_{stridx}.png'), det)
            cv2.imwrite(os.path.join(scene_name,'lidar_baseline',f'test_{stridx}.png'), det_baseline)

    logger.info('Demo done.')
    return scene_name


if __name__ == '__main__':
    scene_name = main()