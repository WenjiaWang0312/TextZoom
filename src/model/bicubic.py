import torch
import torch.nn as nn
import torch.nn.functional as F


class BICUBIC(object):
    def __init__(self, scale_factor=2):
        super(BICUBIC).__init__()
        self.scale_factor = scale_factor

    def __call__(self, x):
        out = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=True)
        return out
