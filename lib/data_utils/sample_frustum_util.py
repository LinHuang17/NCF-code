"""
 utils for trimesh loading
           sampling surface, bouding volume, camera space
           SDF calc. via trimesh or pysdf
           xyz_correspondence & its mask gen.
"""
import os
import pdb
import random
import logging

import torch
import numpy as np

import trimesh
from trimesh.ray import ray_pyembree

from pysdf import SDF

log = logging.getLogger('trimesh')
log.setLevel(40)


def load_trimesh(model_dir, model_unit):
    files = os.listdir(model_dir)
    mesh_dict = {}
    for idx, filename in enumerate(files):
        if filename[-4:] == '.ply':
            # load mesh in model space
            model_mesh = trimesh.load(os.path.join(model_dir, filename), process=False)
            # m -> mm unit if orig. ycbv
            if model_unit == 'meter':
                # m -> mm unit
                model_mesh.vertices = model_mesh.vertices * 1000
            key = filename[:-4]
            mesh_dict[key] = model_mesh

    return mesh_dict


def sampling_in_ball(num_points, dimension, radius=1):

    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(dimension,num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)

    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random(num_points) ** (1/dimension)

    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T


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


# ray-SDF or conventional-SDF
def wks_sampling_sdf_xyz_calc(opt, bmax, bmin, cam_bmax, cam_bmin, model_mesh, extrinsic, calib, bounding):
    # if not self.is_train:
    #     random.seed(1991)
    #     np.random.seed(1991)
    #     torch.manual_seed(1991)

    # extrinsic to transform from model to cam. space
    m2c_rot = extrinsic.numpy()[:3, :3]
    m2c_trans = extrinsic.numpy()[:3, 3:4]
    # (N, 3)
    cam_vert_pts = (m2c_rot.dot(model_mesh.vertices.T) + m2c_trans.reshape((3, 1))).T
    # load mesh in cam. space
    cam_mesh = trimesh.Trimesh(vertices=cam_vert_pts, faces=model_mesh.faces, process=False)
    # (1) sampling in surface with gaussian noise
    surf_ratio = float(opt.sample_ratio) / 8
    surface_points_cam, _ = trimesh.sample.sample_surface(cam_mesh, int(surf_ratio * opt.num_sample_inout))
    # with gaussian noise
    sigma = opt.sigma_ratio * opt.clamp_dist
    noisy_surface_points_cam = surface_points_cam + np.random.normal(scale=sigma, size=surface_points_cam.shape)

    # (2) sampling in tight sphere: add random points within image space
    # 16:1=1250/16:0.5=625 in tight sphere
    bd_length = bmax - bmin
    zero_rot = np.identity(3)
    wks_ratio = opt.sample_ratio // 4
    if bounding == 'abb':
        bounding_points_model = np.random.rand(opt.num_sample_inout // wks_ratio, 3) * bd_length + bmin
    elif bounding == 'sphere':
        radius = bd_length.max() / 2
        bounding_points_model = sampling_in_ball(opt.num_sample_inout // wks_ratio, 3, radius=radius)
    # (N, 3)
    bounding_points_trans = (zero_rot.dot(bounding_points_model.T) + m2c_trans.reshape((3, 1))).T

    # (3) sampling in 3D frustum inside the 3D genearl workspace in front of the camera
    # 16:1=1250/16:0.5=625 in 3D workspace
    wks_sample_flag = True
    frustum_points_trans_list = []
    wks_length = cam_bmax - cam_bmin
    while wks_sample_flag:
        # (N, 3)
        wks_points_trans = np.random.rand((opt.num_sample_inout // wks_ratio) * 10, 3) * wks_length + cam_bmin
        # filter out pts not in camera frustum
        # (N,)
        wks_not_in_img = out_of_plane_mask_calc(wks_points_trans, calib, opt.img_size)
        # (N,)
        wks_in_img = np.logical_not(wks_not_in_img)
        frustum_points_trans_list = frustum_points_trans_list + wks_points_trans[wks_in_img].tolist()
        if len(frustum_points_trans_list) >= (opt.num_sample_inout // wks_ratio):
            wks_sample_flag = False
            frustum_points_trans = np.array(frustum_points_trans_list[:(opt.num_sample_inout // wks_ratio)])

    # (N, 3): combine all 21250 points
    sample_points_cam = np.concatenate([noisy_surface_points_cam, bounding_points_trans, frustum_points_trans], 0)
    np.random.shuffle(sample_points_cam)

    inside = cam_mesh.contains(sample_points_cam)
    inside_points = sample_points_cam[inside]
    outside_points = sample_points_cam[np.logical_not(inside)]

    nin = inside_points.shape[0]
    inside_points = inside_points[
                    :opt.num_sample_inout // 2] if nin > opt.num_sample_inout // 2 else inside_points
    outside_points = outside_points[
                        :opt.num_sample_inout // 2] if nin > opt.num_sample_inout // 2 else outside_points[
                                                                                            :(opt.num_sample_inout - nin)]
    # (N, 3)
    cam_sample_pts = np.concatenate([inside_points, outside_points], 0)

    # trimesh-based ray-SDF
    if opt.out_type == 'rsdf':
        # (N, 1)
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1).T

        ray_mesh_emb = ray_pyembree.RayMeshIntersector(cam_mesh, scale_to_box=False)
        cam_sample_pt_sdf = np.zeros(cam_sample_pts.shape[0])
        ray_origins = np.zeros_like(cam_sample_pts)
        delta_vect = (cam_sample_pts - ray_origins)
        norm_delta = np.expand_dims(np.linalg.norm(delta_vect, axis=1), axis=1)
        unit_ray_dir = delta_vect / norm_delta

        # intersect = ray_mesh_emb.intersects_any(ray_origins, unit_ray_dir)
        _, hit_index_ray, hit_locations = ray_mesh_emb.intersects_id(ray_origins, unit_ray_dir, multiple_hits=True, return_locations=True)
        # intersect mask
        hit_unique_idx_ray = np.unique(hit_index_ray)
        hit_ray_mask = np.zeros(cam_sample_pts.shape[0], dtype=bool)
        hit_ray_mask[hit_unique_idx_ray] = True
        for idx, pt in enumerate(cam_sample_pts):
            if hit_ray_mask[idx]:
                min_df = np.inf
                hit_idx_list = (np.where(np.array(hit_index_ray) == idx)[0]).tolist()
                for hit_idx in hit_idx_list:
                    cur_df = np.linalg.norm((hit_locations[hit_idx] - pt))
                    if cur_df < min_df:
                        min_df = cur_df
                if labels[idx]:
                    cam_sample_pt_sdf[idx] = -min_df
                else:
                    cam_sample_pt_sdf[idx] = min_df
            else:
                cam_sample_pt_sdf[idx] = 100 * opt.clamp_dist
    # pysdf-based conventional-SDF
    if opt.out_type == 'csdf':
        sdf_calc_func = SDF(cam_mesh.vertices, cam_mesh.faces)
        cam_sample_pt_sdf = (-1) * sdf_calc_func(cam_sample_pts)

    # shape (N, 1)
    sdfs = np.expand_dims(cam_sample_pt_sdf, axis=1)

    # deal with out-of-plane cases
    not_in_img = out_of_plane_mask_calc(cam_sample_pts, calib, opt.img_size)
    sdfs[not_in_img] = 100 * opt.clamp_dist

    norm_sdfs = sdfs / (opt.clamp_dist / opt.norm_clamp_dist)

    # obtain for xyz of correspondence in model space
    inverse_ext = torch.inverse(extrinsic)
    c2m_rot = inverse_ext[:3, :3]
    c2m_trans = inverse_ext[:3, 3:4]
    # (3, N)
    model_sample_pts = torch.addmm(c2m_trans, c2m_rot, torch.Tensor(cam_sample_pts.T).float()).float()
    norm_xyz_factor = opt.bbx_size / 2
    norm_model_sample_pts = model_sample_pts / norm_xyz_factor
    # (3, N)
    cam_sample_pts = torch.Tensor(cam_sample_pts.T).float()
    # (1, N)
    norm_sdfs = torch.Tensor(norm_sdfs.T).float()

    del model_mesh
    del cam_mesh

    return {
        'samples': cam_sample_pts,
        'labels': norm_sdfs,
        'xyzs': norm_model_sample_pts,
        'norm_xyz_factor': torch.tensor(norm_xyz_factor)
    }


# efficient conventional-SDF
def wks_sampling_eff_csdf_xyz_calc(opt, bmax, bmin, cam_bmax, cam_bmin, model_mesh, extrinsic, calib, bounding):
    # if not self.is_train:
    #     random.seed(1991)
    #     np.random.seed(1991)
    #     torch.manual_seed(1991)

    # extrinsic to transform from model to cam. space
    m2c_rot = extrinsic.numpy()[:3, :3]
    m2c_trans = extrinsic.numpy()[:3, 3:4]
    # (N, 3)
    cam_vert_pts = (m2c_rot.dot(model_mesh.vertices.T) + m2c_trans.reshape((3, 1))).T
    # load mesh in cam. space
    cam_mesh = trimesh.Trimesh(vertices=cam_vert_pts, faces=model_mesh.faces, process=False)
    # (1) sampling in surface with gaussian noise
    surf_ratio = float(opt.sample_ratio) / 8
    surface_points_cam, _ = trimesh.sample.sample_surface(cam_mesh, int(surf_ratio * opt.num_sample_inout))
    # with gaussian noise
    sigma = opt.sigma_ratio * opt.clamp_dist
    noisy_surface_points_cam = surface_points_cam + np.random.normal(scale=sigma, size=surface_points_cam.shape)

    # (2) sampling in tight sphere: add random points within image space
    # 16:1=1250/16:0.5=625 in tight sphere
    bd_length = bmax - bmin
    zero_rot = np.identity(3)
    wks_ratio = opt.sample_ratio // 4
    if bounding == 'abb':
        bounding_points_model = np.random.rand(opt.num_sample_inout // wks_ratio, 3) * bd_length + bmin
    elif bounding == 'sphere':
        radius = bd_length.max() / 2
        bounding_points_model = sampling_in_ball(opt.num_sample_inout // wks_ratio, 3, radius=radius)
    # (N, 3)
    bounding_points_trans = (zero_rot.dot(bounding_points_model.T) + m2c_trans.reshape((3, 1))).T

    # (3) sampling in 3D frustum inside the 3D genearl workspace in front of the camera
    # 16:1=1250/16:0.5=625 in 3D workspace
    wks_sample_flag = True
    frustum_points_trans_list = []
    wks_length = cam_bmax - cam_bmin
    while wks_sample_flag:
        # (N, 3)
        wks_points_trans = np.random.rand((opt.num_sample_inout // wks_ratio) * 10, 3) * wks_length + cam_bmin
        # filter out pts not in camera frustum
        # (N,)
        wks_not_in_img = out_of_plane_mask_calc(wks_points_trans, calib, opt.img_size)
        # (N,)
        wks_in_img = np.logical_not(wks_not_in_img)
        frustum_points_trans_list = frustum_points_trans_list + wks_points_trans[wks_in_img].tolist()
        if len(frustum_points_trans_list) >= (opt.num_sample_inout // wks_ratio):
            wks_sample_flag = False
            frustum_points_trans = np.array(frustum_points_trans_list[:(opt.num_sample_inout // wks_ratio)])

    # (N, 3): combine all 21250 points
    sample_points_cam = np.concatenate([noisy_surface_points_cam, bounding_points_trans, frustum_points_trans], 0)
    np.random.shuffle(sample_points_cam)

    # pysdf-based conventional-SDF
    sdf_calc_func = SDF(cam_mesh.vertices, cam_mesh.faces)
    sample_points_cam_sdf = (-1) * sdf_calc_func(sample_points_cam)

    inside = (sample_points_cam_sdf < 0)
    inside_points = sample_points_cam[inside]
    outside_points = sample_points_cam[np.logical_not(inside)]
    inside_points_sdf = sample_points_cam_sdf[inside]
    outside_points_sdf = sample_points_cam_sdf[np.logical_not(inside)]

    nin = inside_points.shape[0]
    inside_points = inside_points[
                    :opt.num_sample_inout // 2] if nin > opt.num_sample_inout // 2 else inside_points
    outside_points = outside_points[
                        :opt.num_sample_inout // 2] if nin > opt.num_sample_inout // 2 else outside_points[
                                                                                            :(opt.num_sample_inout - nin)]
    inside_points_sdf = inside_points_sdf[
                    :opt.num_sample_inout // 2] if nin > opt.num_sample_inout // 2 else inside_points_sdf
    outside_points_sdf = outside_points_sdf[
                        :opt.num_sample_inout // 2] if nin > opt.num_sample_inout // 2 else outside_points_sdf[
                                                                                            :(opt.num_sample_inout - nin)]
    # (N, 3)
    cam_sample_pts = np.concatenate([inside_points, outside_points], 0)
    cam_sample_pt_sdf = np.concatenate([inside_points_sdf, outside_points_sdf], 0)

    # shape (N, 1)
    sdfs = np.expand_dims(cam_sample_pt_sdf, axis=1)

    # deal with out-of-plane cases
    not_in_img = out_of_plane_mask_calc(cam_sample_pts, calib, opt.img_size)
    sdfs[not_in_img] = 100 * opt.clamp_dist

    norm_sdfs = sdfs / (opt.clamp_dist / opt.norm_clamp_dist)

    # obtain for xyz of correspondence in model space
    inverse_ext = torch.inverse(extrinsic)
    c2m_rot = inverse_ext[:3, :3]
    c2m_trans = inverse_ext[:3, 3:4]
    # (3, N)
    model_sample_pts = torch.addmm(c2m_trans, c2m_rot, torch.Tensor(cam_sample_pts.T).float()).float()
    norm_xyz_factor = opt.bbx_size / 2
    norm_model_sample_pts = model_sample_pts / norm_xyz_factor
    # (3, N)
    cam_sample_pts = torch.Tensor(cam_sample_pts.T).float()
    # (1, N)
    norm_sdfs = torch.Tensor(norm_sdfs.T).float()

    del model_mesh
    del cam_mesh

    return {
        'samples': cam_sample_pts,
        'labels': norm_sdfs,
        'xyzs': norm_model_sample_pts,
        'norm_xyz_factor': torch.tensor(norm_xyz_factor)
    }


def xyz_mask_calc(sdfs, xyz_range):

    # shape (1, num_sample_inout)
    return {'xyz_mask': (abs(sdfs) < xyz_range).float()}