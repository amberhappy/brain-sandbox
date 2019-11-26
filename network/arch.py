from network.archs.efficientnet import efficientnet
from network.archs.unet import unet
from network.archs.resunet import resunet
from network.archs.unet_3d import unet_3d
from network.archs.resunet_3d import resunet_3d


def get(name):
  arch = dict(
    e2d=efficientnet,
    u2d=unet,
    r2d=resunet,
    u3d=unet_3d,
    r3d=resunet_3d
  )

  return arch[name]
