from torch import nn

class SLAMTransform(nn.Module):
    def __init__(self, image_shape, out_channels):
        super().__init__()

        # image_shape == (channels, height, width)
        self.xform = nn.Conv2d(image_shape[0], out_channels, 1)

    def forward(self, x):
        return self.xform(x)
