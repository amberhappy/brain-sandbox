import os
import re
import cv2
import random
import shutil
import numpy as np

from glob import glob

from inn_pipeline.dataset import NiftiDataset
from network.utils import cubify


class MyExtractor():
  def __init__(
    self,
    name,
    path,
    scan_paths,
    mask_path,
    labels,
    modes,
    extensions,
    shapes,
    extraction_sizes,
    dirs
  ):
    self.name = name
    self.path = path
    self.scan_paths = scan_paths
    self.mask_path = mask_path
    self.labels = labels
    self.modes = modes
    self.extensions = extensions
    self.shapes = shapes
    self.extraction_sizes = extraction_sizes
    self.dirs = dirs


  def __nii_paths(self, scan_path):
    '''Preparing paths to nifti files; HLN-12 set is filtered out because it doesn't contain labels'''
    nii_x_paths = [p for p in glob(os.path.join(*self.path, '*', *scan_path)) if not re.match('.*HLN-12.*', p)]
    nii_y_paths = [p for p in glob(os.path.join(*self.path, '*', *self.mask_path)) if not re.match('.*HLN-12.*', p)]

    return nii_x_paths, nii_y_paths


  def __nii_to_array(self, nii_x_path, nii_y_path):
    '''Converting nifti file to numpy array in axial view'''
    x = NiftiDataset(nii_x_path, grayscale_conversion = True).axial_view
    y = NiftiDataset(nii_y_path, grayscale_conversion = False).axial_view

    return x, y


  def __leave_masks_only(self, x, y):
    '''Getting rid of zero masks'''
    x, y = zip(*[(xx, yy) for xx, yy in zip(x, y) if np.any(yy)])
    
    return np.array(x), np.array(y)


  def __leave_some_zero_masks(self, x, y):
    '''Leave some zero masks in dataset'''
    x_nonzeros, y_nonzeros = zip(*[(xx, yy) for xx, yy in zip(x, y) if np.any(yy)])
    x_zeros, y_zeros = zip(*[(xx, yy) for xx, yy in zip(x, y) if not np.any(yy)])
    
    number_of_zero_masks = int(len(x_nonzeros) / 4)
    
    to_shuffle = list(zip(x_zeros, y_zeros))
    random.shuffle(to_shuffle)
    x_zeros, y_zeros = zip(*to_shuffle)
    
    x_zeros, y_zeros = x_zeros[:number_of_zero_masks], y_zeros[:number_of_zero_masks]
    
    x, y = np.append(x_nonzeros, x_zeros, axis=0), np.append(y_nonzeros, y_zeros, axis=0)
          
    return np.array(x), np.array(y)


  def __segment(self, y):
    '''Filtering brain atlas image based on given labels'''
    for yy in y:
      masks = []
      for l in self.labels:
        masks.append(yy == l)
      yy[~np.all(masks, axis = 0)] = 0.0
      yy[np.any(masks, axis = 0)] = 1.0
    
    return y


  def __resize(self, x, y):
    '''Changing shape of x and y data'''
    x = np.array([cv2.resize(xx, tuple(self.shapes['2d'][::-1])) for xx in x])
    y = np.array([cv2.resize(yy, tuple(self.shapes['2d'][::-1])) for yy in y])
    
    return x, y


  def __binarize(self, y):
    '''Binarizing image'''
    y = np.array([cv2.threshold(yy, yy.max()/2, 1.0, cv2.THRESH_BINARY)[1] for yy in y])
    y = y * 255
    y = y.astype(np.uint8)
    
    return y


  def __custom_cubify(self, x, y):
    '''Dividing volume into cubes'''

    return cubify(x, tuple(self.shapes['3d'])), cubify(y, tuple(self.shapes['3d']))


  def __save(self, path, f_name, item, ext):
    '''Saving file in path, create dir if specified path does not exits'''
    if not os.path.isdir(path):
      os.makedirs(path)
    if ext is 'png':  
      cv2.imwrite(os.path.join(path, f_name), item)
    if ext is 'npy':
      np.save(os.path.join(path, f_name), item)


  def __save_xy_in_path(self, x, y, path, dir_name, size, ext):
    '''Saving sequence of x, y data'''
    #if size == 'medium':
    #  x, y = self.__leave_some_zero_masks(x, y)
    if size == 'small':
      x, y = self.__leave_masks_only(x, y)
    for i, (xx, yy) in enumerate(zip(x, y)):
      x_f_name = "{}_{}.{}".format(dir_name, i, ext)
      y_f_name = "{}_{}.{}".format(dir_name, i, ext)
      self.__save(os.path.join(path, dir_name, 'x'), x_f_name, xx, ext)
      self.__save(os.path.join(path, dir_name, 'y'), y_f_name, yy, ext)
    print('Saving data in path:', os.path.join(path, dir_name))


  def __serve_saving(self, x, y, dir_name, modality, mode, size, ext):
    '''Serving data saving'''
    path = None
    for group in ['train', 'valid', 'test']:
      if dir_name in self.dirs[group]:
        path = os.path.join('input', self.name, modality, mode, size, group)
    if path:
      x, y = self.__custom_cubify(x, y) if mode is '3d' else (x, y)
      self.__save_xy_in_path(x, y, path, dir_name, size, ext)


  def extract_dataset(self):
    if os.path.isdir(os.path.join('input', self.name)):
      shutil.rmtree(os.path.join('input', self.name))
    
    for scan_path in self.scan_paths:
      for size in self.extraction_sizes:
        for nii_x_path, nii_y_path in zip(*self.__nii_paths(scan_path)):
          dir_name = nii_x_path.split('/')[3]
          
          x, y = self.__nii_to_array(nii_x_path, nii_y_path)
          y = self.__segment(y)
          x, y = self.__resize(x, y)
          y = self.__binarize(y)

          modality = scan_path[1].split('.')[0]
          for mode, extension in zip(self.modes, self.extensions):
            self.__serve_saving(x, y, dir_name, modality, mode, size, extension)

    print('Dataset extracted successfully')
