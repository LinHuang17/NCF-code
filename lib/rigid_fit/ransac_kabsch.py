"""Estimate a rigid transform between 2 point clouds.
"""

import numpy as np
from .ransac import RansacEstimator

import pdb

def gen_data(N=100, frac=0.1):
  # create a random rigid transform
  transform = np.eye(4)
  # transform[:3, :3] = RotationMatrix.random()
  transform[:3, :3] = np.array([-0.52573111, 0.85065081, 0.0, 0.84825128, 0.52424812, -0.07505775, -0.06384793, -0.03946019, -0.99717919]).reshape(3,3)
  transform[:3, 3] = 2 * np.random.randn(3) + 1

  # create a random source point cloud
  src_pc = 5 * np.random.randn(N, 3) + 2
  dst_pc = Procrustes.transform_xyz(src_pc, transform)

  # corrupt
  rand_corrupt = np.random.choice(np.arange(len(src_pc)), replace=False, size=int(frac*N))
  dst_pc[rand_corrupt] += np.random.uniform(-10, 10, (int(frac*N), 3))

  return src_pc, dst_pc, transform, rand_corrupt


def transform_from_rotm_tr(rotm, tr):
  transform = np.eye(4)
  transform[:3, :3] = rotm
  transform[:3, 3] = tr
  return transform

class Procrustes:
  """Determines the best rigid transform [1] between two point clouds.

  References:
    [1]: https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
  """
  def __init__(self, transform=None):
    self._transform = transform

  def __call__(self, xyz):
    return Procrustes.transform_xyz(xyz, self._transform)

  @staticmethod
  def transform_xyz(xyz, transform):
    """Applies a rigid transform to an (N, 3) point cloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1))])  # homogenize 3D pointcloud
    xyz_t_h = (transform @ xyz_h.T).T  # apply transform
    return xyz_t_h[:, :3]

#   def estimate(self, X, Y):
#     # find centroids
#     X_c = np.mean(X, axis=0)
#     Y_c = np.mean(Y, axis=0)

#     # shift
#     X_s = X - X_c
#     Y_s = Y - Y_c

#     # compute SVD of covariance matrix
#     cov = Y_s.T @ X_s
#     u, _, vt = np.linalg.svd(cov)

#     # determine rotation
#     rot = u @ vt
#     if np.linalg.det(rot) < 0.:
#       vt[2, :] *= -1
#       rot = u @ vt

#     # determine optimal translation
#     trans = Y_c - rot @ X_c

#     self._transform = transform_from_rotm_tr(rot, trans)

  def estimate(self, X, Y):
    # find centroids
    X_c = np.mean(X, axis=0)
    Y_c = np.mean(Y, axis=0)

    # shift
    X_s = X - X_c
    Y_s = Y - Y_c

    # Computation of the covariance matrix
    C = np.dot(np.transpose(Y_s), X_s)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    rot = np.dot(V, W)

    # determine optimal translation
    trans = Y_c - rot @ X_c

    self._transform = transform_from_rotm_tr(rot, trans)

  def residuals(self, X, Y):
    """L2 distance between point correspondences.
    """
    Y_est = self(X)
    sum_sq = np.sum((Y_est - Y)**2, axis=1)
    return sum_sq

  @property
  def params(self):
    return self._transform


if __name__ == "__main__":
  src_pc, dst_pc, transform_true, rand_corrupt = gen_data(frac=0.2)

  # estimate without ransac, i.e. using all
  # point correspondences
  naive_model = Procrustes()
  naive_model.estimate(src_pc, dst_pc)
  transform_naive = naive_model.params
  mse_naive = np.sqrt(naive_model.residuals(src_pc, dst_pc).mean())
  print("mse naive: {}".format(mse_naive))


  # estimate with RANSAC
  ransac = RansacEstimator(
    min_samples=3,
    # 5, 10, 20
    residual_threshold=(10)**2,
    max_trials=100,
  )
  ret = ransac.fit(Procrustes(), [src_pc, dst_pc])
  transform_ransac = ret["best_params"]


  inliers_ransac = ret["best_inliers"]
  mse_ransac = np.sqrt(Procrustes(transform_ransac).residuals(src_pc, dst_pc).mean())
  print("mse ransac all: {}".format(mse_ransac))
  mse_ransac_inliers = np.sqrt(
    Procrustes(transform_ransac).residuals(src_pc[inliers_ransac], dst_pc[inliers_ransac]).mean())
  print("mse ransac inliers: {}".format(mse_ransac_inliers))
  pdb.set_trace()