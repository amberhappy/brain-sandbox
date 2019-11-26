import segmentation_models as sm
from keras.layers import Input, Conv2D
from keras.models import Model

def efficientnet(shape, n_filters):

  BACKBONE = 'efficientnetb7'

  inputs = Input(shape=(None, None, 1))
  layer = Conv2D(3, (1, 1))(inputs)
  outputs = sm.Unet(BACKBONE)(layer)

  model = Model(inputs=[inputs], outputs=[outputs])

  return model
