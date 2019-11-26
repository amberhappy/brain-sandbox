import random
import numpy as np

from keras.utils import Sequence

from network.augment import augmentator
from network.utils import load_data


class DataSequence(Sequence):
  def __init__(self, x_paths, y_paths, shapes, mode='2d', batch_size=16, augment=False, shuffle=True):
    self.x_paths, self.y_paths = x_paths, y_paths
    self.shapes = shapes
    self.mode = mode
    self.batch_size = batch_size
    self.augment = augment
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    return int(np.ceil(len(self.x_paths) / float(self.batch_size)))

  def __getitem__(self, idx):
    batch_x = self.x_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
    batch_y = self.y_paths[idx * self.batch_size : (idx + 1) * self.batch_size]

    batch_x = np.array([load_data(x, self.mode, self.shapes) for x in batch_x])
    batch_y = np.array([load_data(y, self.mode, self.shapes) for y in batch_y])

    if self.augment:
      batch_x, batch_y = zip(*[augmentator(x, y, self.augment) for x, y in zip(batch_x, batch_y)])

    return np.array(batch_x), np.array(batch_y)

  def on_epoch_end(self):
    if self.shuffle:
      to_shuffle = list(zip(self.x_paths, self.y_paths))
      random.shuffle(to_shuffle)
      self.x_paths, self.y_paths = zip(*to_shuffle)
