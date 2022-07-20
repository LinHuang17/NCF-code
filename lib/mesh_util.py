from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure


def reconstruction(opt, net, calib_tensor,
                   resolution, b_min, b_max, thresh=0.5,
                   use_octree=False, num_samples=10000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).cuda().float()

        transforms = torch.zeros([1,2,3]).cuda()
        transforms[:, 0,0] = 1 / (opt.img_size[0] // 2)
        transforms[:, 1,1] = 1 / (opt.img_size[1] // 2)
        transforms[:, 0,2] = -1 
        transforms[:, 1,2] = -1
        net.query(samples, calib_tensor, transforms=transforms)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, thresh)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1

