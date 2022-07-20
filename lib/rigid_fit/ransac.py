"""A simple RANSAC class implementation.
References:
[1] : https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/fit.py
[2] : https://github.com/scikit-learn/scikit-learn/blob/e5698bde9/sklearn/linear_model/_ransac.py
"""

import numpy as np


class RansacEstimator:
  """Random Sample Consensus.
  """
  def __init__(self, min_samples=None, residual_threshold=None, max_trials=100):
    """Constructor.

    Args:
      min_samples: The minimal number of samples needed to fit the model
        to the data. If `None`, we assume a linear model in which case
        the minimum number is one more than the feature dimension.
      residual_threshold: The maximum allowed residual for a sample to
        be classified as an inlier. If `None`, the threshold is chosen
        to be the median absolute deviation of the target variable.
      max_trials: The maximum number of trials to run RANSAC for. By
        default, this value is 100.
    """
    self.min_samples = min_samples
    self.residual_threshold = residual_threshold
    self.max_trials = max_trials

  def fit(self, model, data):
    """Robustely fit a model to the data.

    Args:
      model: a class object that implements `estimate` and
        `residuals` methods.
      data: the data to fit the model to. Can be a list of
        data pairs, such as `X` and `y` in the case of
        regression.

    Returns:
      A dictionary containing:
        best_model: the model with the largest consensus set
          and lowest residual error.
        inliers: a boolean mask indicating the inlier subset
          of the data for the best model.
    """
    best_model = None
    best_inliers = None
    best_num_inliers = 0
    best_residual_sum = np.inf

    if not isinstance(data, (tuple, list)):
      data = [data]
    num_data, num_feats = data[0].shape

    for trial in range(self.max_trials):
      # randomly select subset
      rand_subset_idxs = np.random.choice(
        np.arange(num_data), size=self.min_samples, replace=False)
      rand_subset = [d[rand_subset_idxs] for d in data]

      # estimate with model
      model.estimate(*rand_subset)

      # compute residuals
      residuals = model.residuals(*data)
      # residuals_sum = residuals.sum()
      inliers = residuals <= self.residual_threshold
      num_inliers = np.sum(inliers)

      # decide if better
      # if (best_num_inliers < num_inliers) or (best_residual_sum > residuals_sum):
      if (best_num_inliers < num_inliers):
        best_num_inliers = num_inliers
        # best_residual_sum = residuals_sum
        best_inliers = inliers

    # refit model using all inliers for this set
    if best_num_inliers == 0:
      data_inliers = data
    else:
      data_inliers = [d[best_inliers] for d in data]
    model.estimate(*data_inliers)

    ret = {
      "best_params": model.params,
      "best_inliers": best_inliers,
    }
    return ret