import random
import numpy as np

from scipy.ndimage import rotate, shift
from skimage.transform import AffineTransform, warp


def augmentator(x, y, mode):
  rotate_values = [0, 5, 10, 15]
  shift_values = [0, 0.1, 0.2, 0.3]
  shear_values = [0, 0.05, 0.1, 0.15]

  rotate_range = rotate_values[mode]
  shift_range = shift_values[mode]
  shear_range = shear_values[mode]

  random_rotate = random.uniform(-rotate_range, rotate_range)
  random_shift = random.uniform(-shift_range, shift_range)
  shift_x = round(x.shape[0] * random_shift)
  shift_y = round(x.shape[0] * random_shift)
  random_shear = random.uniform(-shear_range, shear_range)
  shear_tf = AffineTransform(shear=random_shear)

  x = rotate(x, random_rotate, order=0, reshape=False)
  y = rotate(y, random_rotate, order=0, reshape=False)

  x = shift(x, (shift_x, shift_y, 0), order=0)
  y = shift(y, (shift_x, shift_y, 0), order=0)

  x = warp(x, inverse_map=shear_tf)
  y = warp(y, inverse_map=shear_tf)

  y = (y > 0.5).astype(np.uint8)

  return x, y
