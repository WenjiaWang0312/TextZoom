import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from IPython import embed
from torchvision import transforms


class GradientPriorLoss(nn.Module):
    def __init__(self, ):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss()

    def forward(self, out_images, target_images):
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)
        return self.func(map_out, map_target)

    @staticmethod
    def gradient_map(self, x):
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
        return xgrad


if __name__ == '__main__':
    embed()
