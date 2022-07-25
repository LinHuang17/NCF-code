"""
 dataset class of bop ycbv
"""

import os
import sys
import pdb
import random
import logging
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import PIL
import json
import torch
import pickle
import numpy as np
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data_utils.aug_util import AugmentOp, augment_image
from data_utils.sample_frustum_util import load_trimesh, wks_sampling_sdf_xyz_calc, wks_sampling_eff_csdf_xyz_calc, xyz_mask_calc

from options import BaseOptions
from debug_pyrender_util import *

# log = logging.getLogger('trimesh')
# log.setLevel(40)

class BOP_BP_YCBV(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        # path & state setup
        self.phase = phase
        self.is_train = (self.phase == 'train')

        # 3D->2D projection: 'orthogonal' or 'perspective'
        self.projection_mode = 'perspective'

        # ABBox or Sphere in cam. c.s.
        B_SHIFT = self.opt.bbx_shift
        Bx_SIZE = self.opt.bbx_size // 2
        By_SIZE = self.opt.bbx_size // 2
        Bz_SIZE = self.opt.bbx_size // 2
        self.B_MIN = np.array([-Bx_SIZE, -By_SIZE, -Bz_SIZE])
        self.B_MAX = np.array([Bx_SIZE, By_SIZE, Bz_SIZE])
        # wks box in cam. c.s.
        self.CAM_Bz_SHIFT = self.opt.wks_z_shift
        Cam_Bx_SIZE = self.opt.wks_size[0] // 2
        Cam_By_SIZE = self.opt.wks_size[1] // 2
        Cam_Bz_SIZE = self.opt.wks_size[2] // 2
        self.CAM_B_MIN = np.array([-Cam_Bx_SIZE, -Cam_By_SIZE, -Cam_Bz_SIZE+self.CAM_Bz_SHIFT])
        self.CAM_B_MAX = np.array([Cam_Bx_SIZE, Cam_By_SIZE, Cam_Bz_SIZE+self.CAM_Bz_SHIFT])
        # test wks box in cam. c.s.
        self.TEST_CAM_Bz_SHIFT = self.opt.test_wks_z_shift
        Test_Cam_Bx_SIZE = self.opt.test_wks_size[0] // 2
        Test_Cam_By_SIZE = self.opt.test_wks_size[1] // 2
        Test_Cam_Bz_SIZE = self.opt.test_wks_size[2] // 2
        self.TEST_CAM_B_MIN = np.array([-Test_Cam_Bx_SIZE, -Test_Cam_By_SIZE, -Test_Cam_Bz_SIZE+self.TEST_CAM_Bz_SHIFT])
        self.TEST_CAM_B_MAX = np.array([Test_Cam_Bx_SIZE, Test_Cam_By_SIZE, Test_Cam_Bz_SIZE+self.TEST_CAM_Bz_SHIFT])

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.aug_ops = [
                        AugmentOp('blur', 0.4, [1, self.opt.aug_blur]),
                        AugmentOp('sharpness', 0.3, [0.0, self.opt.aug_sha]),
                        AugmentOp('contrast', 0.3, [0.2, self.opt.aug_con]),
                        AugmentOp('brightness', 0.5, [0.1, self.opt.aug_bri]),
                        AugmentOp('color', 0.3, [0.0, self.opt.aug_col]),
                    ]

        # ycbv train
        # self.obj_id = self.opt.obj_id
        self.obj_id_list = [self.opt.obj_id]
        self.model_dir = self.opt.model_dir
        self.ds_root_dir = self.opt.ds_ycbv_dir
        if self.phase == 'train':
            self.ds_dir = os.path.join(self.ds_root_dir, 'train_pbr')
            start = 0
            end = 50
            self.visib_fract_thresh = self.opt.visib_fract_thresh
        elif self.phase == 'test':
            self.ds_dir = os.path.join(self.ds_root_dir, 'test')
            start = 48
            end = 60
            self.visib_fract_thresh = 0.0
        self.all_gt_info = []
        for folder_id in range(start, end):
            self.scene_gt_dict = {}
            self.scene_gt_info_dict = {}
            self.scene_camera_dict = {}
            with open(os.path.join(self.ds_dir, f'{int(folder_id):06d}/scene_gt_info.json')) as f:
                self.scene_gt_info_dict = json.load(f)

            with open(os.path.join(self.ds_dir, f'{int(folder_id):06d}/scene_camera.json')) as f:
                self.scene_camera_dict = json.load(f)

            with open(os.path.join(self.ds_dir, f'{int(folder_id):06d}/scene_gt.json')) as f:
                self.scene_gt_dict = json.load(f)

            # for data_idx in range(len(self.scene_gt_info_dict)):
            for data_id in self.scene_gt_info_dict.keys():
                len_item = len(self.scene_gt_info_dict[str(data_id)])
                for obj_idx in range(len_item):
                    # self.all_gt_info[str(obj_id)] = []
                    if self.scene_gt_dict[str(data_id)][obj_idx]['obj_id'] in self.obj_id_list:
                        if self.scene_gt_info_dict[str(data_id)][obj_idx]['visib_fract'] > self.visib_fract_thresh:
                            single_annot = {}
                            single_annot['folder_id'] = folder_id
                            single_annot['frame_id'] = int(data_id)
                            single_annot['cam_R_m2c'] = self.scene_gt_dict[str(data_id)][obj_idx]['cam_R_m2c']
                            single_annot['cam_t_m2c'] = self.scene_gt_dict[str(data_id)][obj_idx]['cam_t_m2c']
                            single_annot['obj_id'] = self.scene_gt_dict[str(data_id)][obj_idx]['obj_id']
                            single_annot['cam_K'] = self.scene_camera_dict[str(data_id)]['cam_K']
                            # self.all_gt_info[str(obj_id)].append(single_annot)
                            self.all_gt_info.append(single_annot)
        self.model_mesh_dict = load_trimesh(self.model_dir, self.opt.model_unit)

    def __len__(self):

        return len(self.all_gt_info)

    def get_img_cam(self, frame_id):

        data_gt_info = self.all_gt_info[frame_id]
        folder_id = data_gt_info['folder_id']
        frame_id = data_gt_info['frame_id']
        rgb_parent_path = os.path.join(self.ds_dir, f'{int(folder_id):06d}', 'rgb')
        if self.phase == 'train':
            rgb_path = os.path.join(rgb_parent_path, f'{int(frame_id):06d}.jpg')
        elif self.phase == 'test':
            rgb_path = os.path.join(rgb_parent_path, f'{int(frame_id):06d}.png')

        # shape (H, W, C)/(480, 640, 3)
        render = Image.open(rgb_path).convert('RGB')
        w, h = render.size

        # original camera intrinsic
        K = np.array(data_gt_info['cam_K']).reshape(3, 3)
        camera = dict(K=K.astype(np.float32), aug_K=np.copy(K.astype(np.float32)), resolution=(w, h))

        objects = []
        # annotation for every object in the scene
        # Rotation matrix from model to cam
        R_m2c = np.array(data_gt_info['cam_R_m2c']).reshape(3, 3)
        # translation vector from model to cam
        # unit: mm -> cm
        t_m2c = np.array(data_gt_info['cam_t_m2c'])
        # Rigid Transform class from model to cam/model c.s. 6D pose in cam c.s./extrinsic
        RT_m2c = np.concatenate([R_m2c, t_m2c.reshape(3,1)], axis=1)
        # model to cam: Rigid Transform homo. matrix
        RT_m2c = np.concatenate([RT_m2c, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        obj_id = data_gt_info['obj_id']
        name = f'obj_{int(obj_id):06d}'
        obj = dict(label=name, name=name, RT_m2c=RT_m2c.astype(np.float32))
        objects.append(obj)

        # object name
        objname = objects[0]['name']

        # color aug.
        if self.is_train and self.opt.use_aug:
            render = augment_image(render, self.aug_ops)

        aug_intrinsic = camera['aug_K']
        aug_intrinsic = np.concatenate([aug_intrinsic, np.array([0, 0, 0]).reshape(3, 1)], 1)
        aug_intrinsic = np.concatenate([aug_intrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        extrinsic = objects[0]['RT_m2c']
        calib = torch.Tensor(np.matmul(aug_intrinsic, extrinsic)).float()
        extrinsic = torch.Tensor(extrinsic).float()
        aug_intrinsic = torch.Tensor(aug_intrinsic).float()

        render = self.to_tensor(render)

        # shape (C, H, W), ...
        return {'img': render, 'calib': aug_intrinsic, 'extrinsic': extrinsic, 'aug_intrinsic': aug_intrinsic, 'folder_id': folder_id, 'frame_id': frame_id, 'obj_id': obj_id, 'name': objname}

    def get_item(self, index):

        res = {
            'b_min': self.CAM_B_MIN,
            'b_max': self.CAM_B_MAX,
            'test_b_min': self.TEST_CAM_B_MIN,
            'test_b_max': self.TEST_CAM_B_MAX,
        }

        render_data = self.get_img_cam(index)
        res.update(render_data)
        if self.is_train:
            if self.opt.out_type[:3] == 'eff':
                # efficient conventional-SDF calculation
                sample_data = wks_sampling_eff_csdf_xyz_calc(self.opt,
                                                             # bouding box
                                                             self.B_MAX, self.B_MIN,
                                                             # wks
                                                             self.CAM_B_MAX, self.CAM_B_MIN,
                                                             # model mesh
                                                             self.model_mesh_dict[res['name']].copy(include_cache=False),
                                                             # camera param. & bouding volume
                                                             res['extrinsic'].clone(), res['calib'].clone(), bounding='sphere')
            else:
                # Ray-SDF or conventional-SDF calculation
                sample_data = wks_sampling_sdf_xyz_calc(self.opt,
                                                        # bouding box
                                                        self.B_MAX, self.B_MIN,
                                                        # wks
                                                        self.CAM_B_MAX, self.CAM_B_MIN,
                                                        # model mesh
                                                        self.model_mesh_dict[res['name']].copy(include_cache=False),
                                                        # camera param. & bouding volume
                                                        res['extrinsic'].clone(), res['calib'].clone(), bounding='sphere')
            if self.opt.use_xyz:
                xyz_mask = xyz_mask_calc(sdfs=sample_data['labels'].clone(), xyz_range=self.opt.norm_clamp_dist)
                res.update(xyz_mask)
            res.update(sample_data)
        else:
            norm_xyz_factor = self.opt.bbx_size / 2
            res['norm_xyz_factor'] = torch.tensor(norm_xyz_factor)

        return res

    def __getitem__(self, index):
        return self.get_item(index)


if __name__ == '__main__':

    phase = 'train'
    opt = BaseOptions().parse()
    debug_path = f'/data1/lin/ncf_results/data/ycbv_{opt.out_type}_obj{opt.obj_id}_{phase}'
    os.makedirs(debug_path, exist_ok=True)
    dataset = BOP_BP_YCBV(opt, phase=phase)
    print(f'len. of dataset {len(dataset)}')

    num_debug = 10
    for idx in range(0, len(dataset), len(dataset) // num_debug):
        print(f'Debugging for sample: {idx}')
        res = dataset[idx]

        # debug for rgb, mask, rendering of object model
        # img = np.uint8((np.transpose(res['img'].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
        model_mesh = dataset.model_mesh_dict[res['name']].copy(include_cache=False)
        img = (np.transpose(res['img'].numpy(), (1, 2, 0)) * 0.5 + 0.5)
        save_debug_path = os.path.join(debug_path, f'data_sample{idx}_debug_bop_ycbv.jpeg')
        viz_debug_data(img, model_mesh,
                       res['extrinsic'].numpy(), res['aug_intrinsic'].numpy(),
                       save_debug_path)

        # debug for sampled points with labels: same for each sample
        save_sdf_path = os.path.join(debug_path, f'data_sample{idx}_clamp{opt.norm_clamp_dist}_sdf.ply')
        save_sdf_xyz_path = os.path.join(debug_path, f'data_sample{idx}_clamp{opt.norm_clamp_dist}_xyz.ply')
        save_samples_truncted_sdf(save_sdf_path, res['samples'].numpy().T, res['labels'].numpy().T, thres=opt.norm_clamp_dist)
        save_samples_truncted_sdf(save_sdf_xyz_path, res['xyzs'].numpy().T, res['labels'].numpy().T, thres=opt.norm_clamp_dist)

        # debug for query projection
        save_in_query_path = os.path.join(debug_path, f'data_sample{idx}_in_query.jpeg')
        save_out_query_path = os.path.join(debug_path, f'data_sample{idx}_out_query.jpeg')
        viz_debug_query(opt.out_type, res, save_in_query_path, save_out_query_path)
