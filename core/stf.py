# spatio-temporal fusion

import numpy as np
import torch
from torch import nn
from skimage.transform import resize


def flatten(in_volume, in_images, out_size):
    # in_volume.shape    == (volume_channels, volume_depth, volume_height, volume_width)
    # in_images[i].shape == (image_channels, image_height, image_width)
    # result.shape       == (sum_of_channels, out_size, out_size)

    assert (
        (isinstance(in_volume, np.ndarray) and (len(in_volume.shape) == 4) and (np.prod(in_volume.shape) > 0))
        and (
            (isinstance(in_images, np.ndarray) and (len(in_images.shape) == 3))
            or (isinstance(in_images, list) and (len(in_images) > 0))
        )
    ), 'invalid input'

    def fit(img):
        assert (isinstance(img, np.ndarray) and (len(img.shape) == 3) and (np.prod(img.shape) > 0))
        _, h, w = img.shape
        if (h != out_size) or (w != out_size):
            img = np.moveaxis(img, 0, -1)
            img = resize(img, (out_size, out_size), anti_aliasing=True)
            img = np.moveaxis(img, -1, 0)
        return img

    result = in_volume

    c, d, h, w = result.shape
    if (h != out_size) or (w != out_size):
        result = np.moveaxis(result, 0, -1)
        result = resize(result, (out_size, out_size, out_size), anti_aliasing=True)
        result = np.moveaxis(result, -1, 0)
    result = np.reshape(result, (c*d, h, w))

    if isinstance(in_images, np.ndarray):
        in_images = fit(in_images)
        result    = np.concatenate((result, in_images), axis=0)
    else:
        for img in in_images:
            img    = fit(img)
            result = np.concatenate((result, img), axis=0)

    return result


class Fuse(nn.Module):
    def __init__(self, in_size, in_channels, out_channels):
        super().__init__()

        # in  .shape == (in_channels         , in_size, in_size)
        # out .shape == (in_size*out_channels, in_size, in_size)
        self.out_shape = (out_channels, in_size, in_size, in_size)
        self.xform     = nn.Conv2d(in_channels, in_size*out_channels, 1)

    def forward(self, x):
        x = self.xform(x)
        x = torch.reshape(x, self.out_shape)
        return x
