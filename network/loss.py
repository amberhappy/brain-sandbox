import tensorflow as tf
from keras import backend as K
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


def get(name, alpha):
  '''Getter of loss function'''
  loss = dict(
    bin='binary_crossentropy',
    dcl=dice_loss,
    gdl=generalised_dice_loss,
    sfl=surface_loss,
    sdl=surface_dice_loss(alpha),
    sgl=surface_generalised_loss(alpha)
  )
  
  return loss[name]


def dice(y_true, y_pred, smooth=1.):
  '''Creator of dice coefficent score'''
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
  '''Creator of dice coefficent loss'''
  return 1 - dice(y_true, y_pred)


def extract_classes(y):
  '''Creator of dictionary with classes'''
  return { 'backgr': 1 - y, 'struct': y }


def weighted_dice(class_weights):
  '''Creator of weighted dice coefficent score'''
  def _weighted_dice(y_true, y_pred):
    y_true, y_pred = [extract_classes(y) for y in [y_true, y_pred]]

    dice_score = 0.0
    for class_name in ['backgr', 'struct']:
      dice_score += class_weights[class_name] * dice(y_true[class_name], y_pred[class_name])
    
    return dice_score
  return _weighted_dice


def weighted_dice_loss(class_weights):
  '''Creator of weighted dice coefficent loss'''
  def _weighted_dice_loss(y_true, y_pred):
    return 1 - weighted_dice(class_weights)(y_true, y_pred)
  return _weighted_dice_loss


def generalised_dice(y_true, y_pred):
  '''Creator of generalised dice coefficent score'''
  y_true, y_pred = [extract_classes(y) for y in [K.flatten(y_true), K.flatten(y_pred)]]

  nom, den = 0.0, 0.0
  for class_name in ['backgr', 'struct']:
    class_weight = 1 / K.square(K.sum(y_true[class_name]))
    nom += class_weight * K.sum(y_true[class_name] * y_pred[class_name])
    den += class_weight * K.sum(y_true[class_name] + y_pred[class_name])

  return 2 * nom / den


def generalised_dice_loss(y_true, y_pred):
  '''Creator of generalised dice coefficent loss'''
  return 1 - generalised_dice(y_true, y_pred)


def calc_dist_map(seg):
  '''Creator of dist map'''
  res = np.zeros_like(seg)
  posmask = seg.astype(np.bool)

  if posmask.any():
    negmask = ~posmask
    res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

  return res


def calc_dist_map_batch(y_true):
  '''Calculator of dist map'''
  y_true_numpy = y_true.numpy()
    
  return np.array([calc_dist_map(y) for y in y_true_numpy]).astype(np.float32)


def surface_loss(y_true, y_pred):
  '''Creator of surface loss'''
  y_true_dist_map = tf.py_function(calc_dist_map_batch, [y_true], tf.float32)
  multipled = y_pred * y_true_dist_map

  return K.mean(multipled)


def surface_dice_loss(alpha):
  '''Creator of combined loss (surface and dice)'''
  def _surface_dice_loss(y_true, y_pred):
    return alpha * dice_loss(y_true, y_pred) + (1 - alpha) * surface_loss(y_true, y_pred)
  return _surface_dice_loss


def surface_generalised_loss(alpha):
  '''Creator of combined loss (surface and generalised dice)'''
  def _surface_generalised_loss(y_true, y_pred):
    return alpha * generalised_dice_loss(y_true, y_pred) + (1 - alpha) * surface_loss(y_true, y_pred)
  return _surface_generalised_loss
