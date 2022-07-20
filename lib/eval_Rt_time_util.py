"""
 comprehensive evaluation for:
 SDF, predicted corresopndence, 6D pose
"""

import os
import json
import time
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image

import trimesh

from .geometry import *

from lib.rigid_fit.ransac import RansacEstimator
from lib.rigid_fit.ransac_kabsch import Procrustes

from .sdf import create_grid, eval_sdf_xyz_grid_frustum


def save_bop_results(path, results, version='bop19'):
  """Saves 6D object pose estimates to a file.
  :param path: Path to the output file.
  :param results: Dictionary with pose estimates.
  :param version: Version of the results.
  """
  # See docs/bop_challenge_2019.md for details.
  if version == 'bop19':
    lines = ['scene_id,im_id,obj_id,score,R,t,time']
    for res in results:
      if 'time' in res:
        run_time = res['time']
      else:
        run_time = -1

      lines.append('{scene_id},{im_id},{obj_id},{score},{R},{t},{time}'.format(
        scene_id=res['scene_id'],
        im_id=res['im_id'],
        obj_id=res['obj_id'],
        score=res['score'],
        R=' '.join(map(str, res['R'].flatten().tolist())),
        t=' '.join(map(str, res['t'].flatten().tolist())),
        time=run_time))

    with open(path, 'w') as f:
      f.write('\n'.join(lines))

  else:
    raise ValueError('Unknown version of BOP results.')


def out_of_plane_mask_calc(cam_pts, calib, img_size):
    # deal with out-of-plane cases
    c2i_rot = calib[:3, :3]
    c2i_trans = calib[:3, 3:4]
    img_sample_pts = torch.addmm(c2i_trans, c2i_rot, torch.Tensor(cam_pts.T).float())
    img_sample_uvs = img_sample_pts[:2, :] / img_sample_pts[2:3, :]

    # normalize to [-1,1]
    transforms = torch.zeros([2,3])
    transforms[0,0] = 1 / (img_size[0] // 2)
    transforms[1,1] = 1 / (img_size[1] // 2)
    transforms[0,2] = -1
    transforms[1,2] = -1
    scale = transforms[:2, :2]
    shift = transforms[:2, 2:3]
    img_sample_norm_uvs = torch.addmm(shift, scale, img_sample_uvs)
    in_img = (img_sample_norm_uvs[0,:] >= -1.0) & (img_sample_norm_uvs[0,:] <= 1.0) & (img_sample_norm_uvs[1,:] >= -1.0) & (img_sample_norm_uvs[1,:] <= 1.0)
    not_in_img = torch.logical_not(in_img).numpy()

    return not_in_img


"""
generate 6D rigid pose based on SDF & Corresopndence
calculate eval. time
"""
def eval_Rt_time(opt, net, test_data_loader, save_csv_path):

    with torch.no_grad():
        preds = []
        # for test_idx, test_data in enumerate(test_data_loader):
        for test_idx, test_data in enumerate(tqdm(test_data_loader)):

            # retrieve the data
            # resolution = opt.resolution
            resolution_X = int(opt.test_wks_size[0] / opt.step_size)
            resolution_Y = int(opt.test_wks_size[1] / opt.step_size)
            resolution_Z = int(opt.test_wks_size[2] / opt.step_size)
            image_tensor = test_data['img'].cuda()
            calib_tensor = test_data['calib'].cuda()
            norm_xyz_factor = test_data['norm_xyz_factor'][0].item()

            # get all 3D queries
            # create a grid by resolution
            # and transforming matrix for grid coordinates to real world xyz
            b_min = np.array(test_data['test_b_min'][0])
            b_max = np.array(test_data['test_b_max'][0])
            coords, mat = create_grid(resolution_X, resolution_Y, resolution_Z, b_min, b_max, transform=None)
            # (M=KxKxK, 3)
            coords = coords.reshape([3, -1]).T
            # (M,)
            coords_not_in_img = out_of_plane_mask_calc(coords, test_data['calib'][0], opt.img_size)
            # (M,)
            coords_in_img = np.logical_not(coords_not_in_img)
            # (3, N)
            coords_in_frustum = coords[coords_in_img].T

            # transform for proj.
            transforms = torch.zeros([1,2,3]).cuda()
            transforms[:, 0,0] = 1 / (opt.img_size[0] // 2)
            transforms[:, 1,1] = 1 / (opt.img_size[1] // 2)
            transforms[:, 0,2] = -1
            transforms[:, 1,2] = -1

            # create ransac
            ransac = RansacEstimator(
                                    min_samples=opt.min_samples,
                                    residual_threshold=(opt.res_thresh)**2,
                                    max_trials=opt.max_trials,
                                    )

            eval_start_time = time.time()
            # get 2D feat. maps
            net.filter(image_tensor)
            # Then we define the lambda function for cell evaluation
            def eval_func(points):
                points = np.expand_dims(points, axis=0)
                # points = np.repeat(points, net.num_views, axis=0)
                samples = torch.from_numpy(points).cuda().float()

                transforms = torch.zeros([1,2,3]).cuda()
                transforms[:, 0,0] = 1 / (opt.img_size[0] // 2)
                transforms[:, 1,1] = 1 / (opt.img_size[1] // 2)
                transforms[:, 0,2] = -1
                transforms[:, 1,2] = -1
                net.query(samples, calib_tensor, transforms=transforms)
                # shape (B, 1, N) -> (N)
                eval_sdfs = net.preds[0][0]
                # shape (B, 3, N) -> (3, N)
                eval_xyzs = net.xyzs[0]
                return eval_sdfs.detach().cpu().numpy(), eval_xyzs.detach().cpu().numpy()
            # (N), (3, N), all the predicted dfs and xyzs
            pred_sdfs, pred_xyzs = eval_sdf_xyz_grid_frustum(coords_in_frustum, eval_func, num_samples=opt.num_in_batch)
            # norm_xyz_factor = max(opt.bbx_size) / 2
            pred_xyzs = pred_xyzs * norm_xyz_factor
            # get sdf & xyz within clamping distance
            pos_anchor_mask = (abs(pred_sdfs) < opt.norm_clamp_dist)
            est_cam_pts = coords_in_frustum[:, pos_anchor_mask]
            est_model_pts = pred_xyzs[:, pos_anchor_mask]
            # mask_sdfs = pred_sdfs[pos_anchor_mask]

            # estimate 6D pose with RANSAC-based kabsch or procruste
            ret = ransac.fit(Procrustes(), [est_model_pts.T, est_cam_pts.T])
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time

            # est. RT
            RT_m2c_est = ret["best_params"]
            R_m2c_est = RT_m2c_est[:3, :3]
            t_m2c_est = RT_m2c_est[:3, 3:4]

            scene_id = int(test_data['folder_id'][0])
            im_id = int(test_data['frame_id'][0])
            obj_id = int(test_data['obj_id'][0])
            pred = dict(scene_id=scene_id,
                        im_id=im_id,
                        obj_id=obj_id,
                        score=1,
                        R=np.array(R_m2c_est).reshape(3, 3),
                        t=np.array(t_m2c_est),
                        time=eval_time)
            preds.append(pred)
        save_bop_results(save_csv_path, preds)
