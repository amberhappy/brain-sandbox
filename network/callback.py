from keras import backend as K
from keras.callbacks import Callback


class AlphaScheduler(Callback):
  def __init__(self, alpha, update_fn):
    self.alpha = alpha
    self.update_fn = update_fn
  
  def on_epoch_end(self, epoch, logs=None):
    updated_alpha = self.update_fn(K.get_value(self.alpha))
    K.set_value(self.alpha, updated_alpha)
