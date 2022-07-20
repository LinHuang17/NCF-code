from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
 debugging utils
"""

import os
import sys
import pdb
import code
import json
import random
import pickle
import warnings
import datetime
import subprocess

import PIL
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# import pyrender

import trimesh
# import transforms3d as t3d


# """
# pyrender-based rendering
# """
# class Renderer(object):
#     """
#     Render mesh using PyRender for visualization.
#     in m unit by default
#     """
#     def __init__(self, alight_color, dlight_color, dlight_int=2.0, bg='black', im_width=640, im_height=480):

#         self.im_width = im_width
#         self.im_height = im_height
#         # light initialization
#         self.alight_color = alight_color
#         self.dlight_int = dlight_int
#         self.dlight_color = dlight_color
#         # blending coe for bg
#         if bg == 'white':
#             self.bg_color = [1.0, 1.0, 1.0]
#         elif bg == 'black':
#             self.bg_color = [0.0, 0.0, 0.0]

#         # render creation
#         self.renderer = pyrender.OffscreenRenderer(self.im_width, self.im_height)
#         # renderer_flags = pyrender.constants.RenderFlags.DEPTH_ONLY
#         # renderer_flags = pyrender.constants.RenderFlags.FLAT
#         # renderer_flags = pyrender.constants.RenderFlags.RGBA

#         # light creation
#         self.direc_light = pyrender.DirectionalLight(color=self.dlight_color, intensity=self.dlight_int)

#     def render(self, cam_intr, cam_pose, tri_mesh):

#         # scene creation
#         self.scene = pyrender.Scene(ambient_light=self.alight_color, bg_color=self.bg_color)

#         # camera
#         K = np.copy(cam_intr)
#         fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
#         # fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
#         camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)

#         # Object->Camera to Camera->Object.
#         # camera_pose = np.linalg.inv(camera_pose)

#         # OpenCV to OpenGL coordinate system.
#         camera_pose = self.opencv_to_opengl_transformation(cam_pose)

#         # create mesh node
#         tri_mesh.vertices *= 0.001  # To meters.
#         mesh = pyrender.Mesh.from_trimesh(tri_mesh)

#         # add mesh
#         self.scene.add(mesh)
#         # add direc_light
#         self.scene.add(self.direc_light, pose=camera_pose)
#         # Create a camera node and add
#         camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
#         self.scene.add_node(camera_node)

#         # render
#         color, _ = self.renderer.render(self.scene)
#         # color, _ = renderer.render(scene, flags=renderer_flags)
#         self.scene.remove_node(camera_node)
#         color = np.uint8(color)

#         return color

#     def opencv_to_opengl_transformation(self, trans):
#         """Converts a transformation from OpenCV to OpenGL coordinate system.

#         :param trans: A 4x4 transformation matrix.
#         """
#         yz_flip = np.eye(4, dtype=np.float64)
#         yz_flip[1, 1], yz_flip[2, 2] = -1, -1
#         trans = trans.dot(yz_flip)
#         return trans

"""
 save point cloud
"""
def save_samples_rgb(fname, points, rgb):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param rgb: [N, 3] array of rgb values in the range [0~1]
    :return:
    '''
    to_save = np.concatenate([points, rgb * 255], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

def save_samples_truncted_sdf(fname, points, sdf, thres):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param sdf: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (sdf <= -thres).reshape([-1, 1]) * 255
    g = (sdf >= thres).reshape([-1, 1]) * 255
    b = (abs(sdf) < thres).reshape([-1, 1]) * 255
    # b = np.zeros(r.shape)
    # pdb.set_trace()
    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

"""
 save mesh
"""
def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()


"""
 viz img, mask, rendering
"""
def viz_debug_data(img, model_mesh, extrinsic, aug_intrinsic, save_debug_path):

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(img)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_debug_path, dpi=100)


"""
viz query projection for debugging
"""
def viz_debug_query(out_type, res, save_in_query_path, save_out_query_path):

    # from RGB order to opencv BGR order
    img = np.uint8((np.transpose(res['img'].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
    img_cp = np.copy(img)
    rot = res['calib'][:3, :3]
    trans = res['calib'][:3, 3:4]

    # draw points inside
    # pts = torch.addmm(trans, rot, sample_data['samples'])  # [3, N]
    if out_type[-3:] == 'sdf':
        pts = torch.addmm(trans, rot, res['samples'][:, res['labels'][0] < 0])  # [3, N]
    uv = pts[:2, :] / pts[2:3, :]
    uvz = torch.cat([uv, pts[2:3, :]], 0)
    # draw projected queries
    img = np.ascontiguousarray(img, dtype=np.uint8)
    for pt in torch.transpose(uvz, 0, 1):
        img = cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0,0,255), -1)
    cv2.imwrite(save_in_query_path, img)

    # draw points outside
    if out_type[-3:] == 'sdf':
        pts = torch.addmm(trans, rot, res['samples'][:, res['labels'][0] > 0])  # [3, N]
    uv = pts[:2, :] / pts[2:3, :]
    uvz = torch.cat([uv, pts[2:3, :]], 0)
    # draw projected queries
    img_cp = np.ascontiguousarray(img_cp, dtype=np.uint8)
    for pt in torch.transpose(uvz, 0, 1):
        img_cp = cv2.circle(img_cp, (int(pt[0]), int(pt[1])), 2, (0,255,0), -1)
    cv2.imwrite(save_out_query_path, img_cp)

def viz_debug_query_forward(out_type, res, save_in_query_path, save_out_query_path):

    # from RGB order to opencv BGR order
    img = np.uint8((np.transpose(res['img'].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
    img_cp = np.copy(img)

    # draw points inside
    if out_type[-3:] == 'sdf':
        uv = (res['samples'][:, res['labels'][0] < 0])  # [2, N]
    # draw projected queries
    img = np.ascontiguousarray(img, dtype=np.uint8)
    for pt in torch.transpose(uv, 0, 1):
        img = cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0,0,255), -1)
    cv2.imwrite(save_in_query_path, img)

    # draw points outside
    if out_type[-3:] == 'sdf':
        uv = (res['samples'][:, res['labels'][0] > 0])  # [2, N]
    # draw projected queries
    img_cp = np.ascontiguousarray(img_cp, dtype=np.uint8)
    for pt in torch.transpose(uv, 0, 1):
        img_cp = cv2.circle(img_cp, (int(pt[0]), int(pt[1])), 2, (0,255,0), -1)
    cv2.imwrite(save_out_query_path, img_cp)

"""
 Meter for recording
"""
class AverageMeter(object):
    """
     refer to https://github.com/bearpaw/pytorch-pose
     Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
