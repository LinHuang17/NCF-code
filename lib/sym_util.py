# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Parameters of the BOP datasets."""

import os
import pdb
import json

import math
import numpy as np

def load_json(path, keys_to_int=False):
  """Loads content of a JSON file.

  :param path: Path to the JSON file.
  :return: Content of the loaded JSON file.
  """
  # Keys to integers.
  def convert_keys_to_int(x):
    return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

  with open(path, 'r') as f:
    if keys_to_int:
      content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
    else:
      content = json.load(f)

  return content

def get_obj_params(models_path, dataset_name):
  """Returns parameters of object models for the specified dataset.

  :param models_path: Path to a folder with models.
  :param dataset_name: Name of the dataset for which to return the parameters.
  :return: Dictionary with object model parameters for the specified dataset.
  """
  # Object ID's.
  obj_ids = {
    'lm': list(range(1, 16)),
    'lmo': [1, 5, 6, 8, 9, 10, 11, 12],
    'tudl': list(range(1, 4)),
    'tyol': list(range(1, 22)),
    'ruapc': list(range(1, 15)),
    'icmi': list(range(1, 7)),
    'icbin': list(range(1, 3)),
    'itodd': list(range(1, 29)),
    'hbs': [1, 3, 4, 8, 9, 10, 12, 15, 17, 18, 19, 22, 23, 29, 32, 33],
    'hb': list(range(1, 34)),  # Full HB dataset.
    'ycbv': list(range(1, 22)),
    'hope': list(range(1, 29)),
  }[dataset_name]

  # ID's of objects with ambiguous views evaluated using the ADI pose error
  # function (the others are evaluated using ADD). See Hodan et al. (ECCVW'16).
  symmetric_obj_ids = {
    'lm': [3, 7, 10, 11],
    'lmo': [10, 11],
    'tudl': [],
    'tyol': [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21],
    'ruapc': [8, 9, 12, 13],
    'icmi': [1, 2, 6],
    'icbin': [1],
    'itodd': [2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 17, 18, 19, 23, 24, 25, 27, 28],
    'hbs': [10, 12, 18, 29],
    'hb': [6, 10, 11, 12, 13, 14, 18, 24, 29],
    'ycbv': [1, 13, 14, 16, 18, 19, 20, 21],
    'hope': None,  # Not defined yet.
  }[dataset_name]

  # Both versions of the HB dataset share the same directory.
  if dataset_name == 'hbs':
    dataset_name = 'hb'

  p = {
    # ID's of all objects included in the dataset.
    'obj_ids': obj_ids,

    # ID's of objects with symmetries.
    'symmetric_obj_ids': symmetric_obj_ids,

    # Path to a file with meta information about the object models.
    'models_info_path': os.path.join(models_path, 'models_info.json')
  }

  return p

def unit_vector(data, axis=None, out=None):
  """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

  >>> v0 = numpy.random.random(3)
  >>> v1 = unit_vector(v0)
  >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
  True
  >>> v0 = numpy.random.rand(5, 4, 3)
  >>> v1 = unit_vector(v0, axis=-1)
  >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
  >>> numpy.allclose(v1, v2)
  True
  >>> v1 = unit_vector(v0, axis=1)
  >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
  >>> numpy.allclose(v1, v2)
  True
  >>> v1 = numpy.empty((5, 4, 3))
  >>> unit_vector(v0, axis=1, out=v1)
  >>> numpy.allclose(v1, v2)
  True
  >>> list(unit_vector([]))
  []
  >>> list(unit_vector([1]))
  [1.0]

  """
  if out is None:
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
      data /= math.sqrt(np.dot(data, data))
      return data
  else:
    if out is not data:
      out[:] = np.array(data, copy=False)
    data = out
  length = np.atleast_1d(np.sum(data * data, axis))
  np.sqrt(length, length)
  if axis is not None:
    length = np.expand_dims(length, axis)
  data /= length
  if out is None:
    return data

def rotation_matrix(angle, direction, point=None):
  """Return matrix to rotate about axis defined by point and direction.

  >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
  >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
  True
  >>> angle = (random.random() - 0.5) * (2*math.pi)
  >>> direc = numpy.random.random(3) - 0.5
  >>> point = numpy.random.random(3) - 0.5
  >>> R0 = rotation_matrix(angle, direc, point)
  >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
  >>> is_same_transform(R0, R1)
  True
  >>> R0 = rotation_matrix(angle, direc, point)
  >>> R1 = rotation_matrix(-angle, -direc, point)
  >>> is_same_transform(R0, R1)
  True
  >>> I = numpy.identity(4, numpy.float64)
  >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
  True
  >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
  ...                                               direc, point)))
  True

  """
  sina = math.sin(angle)
  cosa = math.cos(angle)
  direction = unit_vector(direction[:3])
  # rotation matrix around unit vector
  R = np.diag([cosa, cosa, cosa])
  R += np.outer(direction, direction) * (1.0 - cosa)
  direction *= sina
  R += np.array([[0.0, -direction[2], direction[1]],
                    [direction[2], 0.0, -direction[0]],
                    [-direction[1], direction[0], 0.0]])
  M = np.identity(4)
  M[:3, :3] = R
  if point is not None:
    # rotation not around origin
    point = np.array(point[:3], dtype=np.float64, copy=False)
    M[:3, 3] = point - np.dot(R, point)
  return M

def get_symmetry_transformations(model_info, max_sym_disc_step):
  """Returns a set of symmetry transformations for an object model.

  :param model_info: See files models_info.json provided with the datasets.
  :param max_sym_disc_step: The maximum fraction of the object diameter which
    the vertex that is the furthest from the axis of continuous rotational
    symmetry travels between consecutive discretized rotations.
  :return: The set of symmetry transformations.
  """
  # Discrete symmetries.
  trans_disc = [{'R': np.eye(3), 't': np.array([[0, 0, 0]]).T}]  # Identity.
  if 'symmetries_discrete' in model_info:
    for sym in model_info['symmetries_discrete']:
      sym_4x4 = np.reshape(sym, (4, 4))
      R = sym_4x4[:3, :3]
      t = sym_4x4[:3, 3].reshape((3, 1))
      trans_disc.append({'R': R, 't': t})

  # Discretized continuous symmetries.
  trans_cont = []
  if 'symmetries_continuous' in model_info:
    for sym in model_info['symmetries_continuous']:
      axis = np.array(sym['axis'])
      offset = np.array(sym['offset']).reshape((3, 1))

      # (PI * diam.) / (max_sym_disc_step * diam.) = discrete_steps_count
      discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))

      # Discrete step in radians.
      discrete_step = 2.0 * np.pi / discrete_steps_count

      for i in range(1, discrete_steps_count):
        R = rotation_matrix(i * discrete_step, axis)[:3, :3]
        t = -R.dot(offset) + offset
        trans_cont.append({'R': R, 't': t})

  # Combine the discrete and the discretized continuous symmetries.
  trans = []
  for tran_disc in trans_disc:
    if len(trans_cont):
      for tran_cont in trans_cont:
        R = tran_cont['R'].dot(tran_disc['R'])
        t = tran_cont['R'].dot(tran_disc['t']) + tran_cont['t']
        trans.append({'R': R, 't': t})
    else:
      trans.append(tran_disc)

  return trans


if __name__ == '__main__':
  # PARAMETERS.
  ################################################################################
  p = {
    # See dataset_params.py for options.
    'dataset': 'ycbv',

    # See misc.get_symmetry_transformations().
    'max_sym_disc_step': 0.01,

    # Folder containing the BOP datasets.
    'models_path': '/mnt/data0/lin/bop_datasets/ycbv/models',

  }
  ################################################################################

  # Load dataset parameters.
  obj_params = get_obj_params(p['models_path'], p['dataset'])

  # Load meta info about the models (including symmetries).
  models_info = load_json(obj_params['models_info_path'], keys_to_int=True)

#   for obj_id in obj_params['obj_ids']:
  import torch
  sym_pool=[]
  obj_id = 13
  sym_poses = get_symmetry_transformations(models_info[obj_id], p['max_sym_disc_step'])
  for sym_pose in sym_poses:
      Rt = np.concatenate([sym_pose['R'], sym_pose['t'].reshape(3,1)], axis=1)
      Rt = np.concatenate([Rt, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
      sym_pool.append(torch.Tensor(Rt))
  pdb.set_trace()
