from keras.optimizers import Adam

from keras_radam import RAdam

def get(name):
  '''Getter of optimizer function'''
  optimizer = dict(
    adm=Adam,
    rdm=RAdam
  )

  return optimizer[name]
