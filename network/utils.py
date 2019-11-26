import os
import cv2
import numpy as np

from glob import glob
from natsort import natsorted
from matplotlib import colors
from matplotlib import pyplot as plt
from keras.backend import int_shape, variable
from keras.layers.convolutional import ZeroPadding3D, Cropping3D
from scipy.ndimage import distance_transform_edt as distance


def create_paths(name, mode, dataset, modality):
  x = { 'train': [], 'valid': [], 'test': [] }
  y = { 'train': [], 'valid': [], 'test': [] }

  modality_map = dict(fl='FLAIR', t1='T1', all='*')
  modality = modality_map[modality]

  dataset_map = dict(sm='small', md='medium', lg='large')
  dataset = dataset_map[dataset]

  for group in ['train', 'valid']:
    x[group] = glob(os.path.join('input', name, modality, mode, dataset, group, '*', 'x', '*'))
    y[group] = glob(os.path.join('input', name, modality, mode, dataset, group, '*', 'y', '*'))
  
  x['test'] = natsorted(glob(os.path.join('input', name, 'FLAIR', mode, 'large', 'test', '*', 'x', '*')))
  y['test'] = natsorted(glob(os.path.join('input', name, 'FLAIR', mode, 'large', 'test', '*', 'y', '*')))

  return x['train'], y['train'], x['valid'], y['valid'], x['test'], y['test']


def calculate_weights(name, mode, dataset, modality, shapes):
  _, y_train_paths, _, _, _, _ = create_paths(name, mode, dataset, modality)

  class_counter = { 'backgr': 0, 'struct': 0 }
  for y_train_path in y_train_paths:
    data = load_data(y_train_path, mode, shapes)
    nonzeros = np.count_nonzero(data)
    class_counter['backgr'] += data.size - nonzeros
    class_counter['struct'] += nonzeros
  class_share = { key: value / sum(class_counter.values()) for key, value in class_counter.items() }
  class_weights = { 'backgr': class_share['struct'], 'struct': class_share['backgr'] }
  
  return class_weights


def load_data(filename, mode, shapes):
  data = np.array([])
  if mode is '2d':
    data = np.array(cv2.imread(filename, cv2.IMREAD_GRAYSCALE)).reshape(*shapes[mode], 1)
  if mode is '3d':
    data = np.load(filename).reshape(*shapes[mode], 1)

  return data.astype(np.float32) * 1 / 255


def cubify(arr, newshape):
  oldshape = np.array(arr.shape)
  repeats = (oldshape / newshape).astype(int)
  tmpshape = np.column_stack([repeats, newshape]).ravel()
  order = np.arange(len(tmpshape))
  order = np.concatenate([order[::2], order[1::2]])
  # newshape must divide oldshape evenly or else ValueError will be raised
  return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)


def uncubify(arr, oldshape):
  _, newshape = arr.shape[0], arr.shape[1:]
  oldshape = np.array(oldshape)    
  repeats = (oldshape / newshape).astype(int)
  tmpshape = np.concatenate([repeats, newshape])
  order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
  return arr.reshape(tmpshape).transpose(order).reshape(oldshape)


def calc_fit_pad(width, height, depth, n_layers):
  divider = 2 ** n_layers
  w_pad, h_pad, d_pad = 0, 0, 0

  w_rest = width % divider
  h_rest = height % divider
  d_rest = depth % divider

  if w_rest:
    w_pad = (divider - w_rest) // 2
  if h_rest:
    h_pad = (divider - h_rest) // 2
  if d_rest:
    d_pad = (divider - d_rest) // 2

  return w_pad, h_pad, d_pad


def pad_to_fit(inputs, n_layers=4):
  width = int_shape(inputs)[1]
  height = int_shape(inputs)[2]
  depth = int_shape(inputs)[3]

  w_pad, h_pad, d_pad = calc_fit_pad(width, height, depth, n_layers)

  x = ZeroPadding3D((w_pad, h_pad, d_pad))(inputs)
  return x


def crop_to_fit(inputs, outputs, n_layers=4):
  width = int_shape(inputs)[1]
  height = int_shape(inputs)[2]
  depth = int_shape(inputs)[3]

  w_pad, h_pad, d_pad = calc_fit_pad(width, height, depth, n_layers)

  x = Cropping3D((w_pad, h_pad, d_pad))(outputs)
  return x


def extract_generator(generator):
  batches_x, batches_y = zip(*generator.__iter__())

  x = [x for batch_x in batches_x for x in batch_x]
  y = [y for batch_y in batches_y for y in batch_y]

  return np.array(x), np.array(y)


def squeeze_all(*items):
  return [np.squeeze(item) for item in items]


def uncubify_all(*items):
  return [uncubify(item, (256, 256, )) for item in items]


def prepare_visualisation(x, y, y_pred, y_comb, slices):

  rows, cols = len(slices), 5

  fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 60))
  [ax.set_axis_off() for ax in axs.ravel()]

  for i, _slice in enumerate(slices):
    axs[i, 0].imshow(x[_slice], cmap='gray')
    axs[i, 0].set_title('Image')

    axs[i, 1].imshow(y[_slice], cmap='gray')
    axs[i, 1].set_title('Mask')

    axs[i, 2].imshow(y_pred[_slice], cmap='gray')
    axs[i, 2].set_title('Predicted Mask')

    cmap = colors.ListedColormap(['black', 'red', 'yellow', 'green'])
    norm = colors.Normalize(vmin=0, vmax=3)

    axs[i, 3].imshow(y_comb[_slice], cmap=cmap, norm=norm)
    axs[i, 3].set_title('Combined Mask')

    axs[i, 4].imshow(x[_slice], cmap='gray')
    axs[i, 4].imshow(y_comb[_slice], cmap=cmap, norm=norm, alpha=0.7)
    axs[i, 4].set_title('Layered Scan')

  return fig


alpha = variable(1, dtype='float32')


def alpha_coef(alpha):
  def _alpha_coef(y_true, y_pred):
    return alpha
  
  return _alpha_coef


def update_alpha(value):

  return np.clip(value - 0.01, 0.01, 1)


def calc_conf_matrix(y, y_pred):
  y_comb = 2 * y + y_pred

  return [np.count_nonzero(y_comb == i) for i in [3, 2, 1, 0]]


def calc_metrics(y, y_pred, epsilon=1e-4):
  tp, fn, fp, tn = calc_conf_matrix(y, y_pred)

  prec = tp / (tp + fp + epsilon)
  rec = tp / (tp + fn + epsilon)
  f1 = 2 * prec * rec / (prec + rec + epsilon)

  return prec, rec, f1
