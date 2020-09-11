import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from IPython import embed
from torchvision import transforms


class ImageLoss(nn.Module):
    def __init__(self, gradient=True, loss_weight=[20, 1e-4]):
        super(ImageLoss, self).__init__()
        self.mse = nn.MSELoss()
        if gradient:
            self.GPLoss = GradientPriorLoss()
        self.gradient = gradient
        self.loss_weight = loss_weight

    def forward(self, out_images, target_images):
        if self.gradient:
            loss = self.loss_weight[0] * self.mse(out_images, target_images) + \
                   self.loss_weight[1] * self.GPLoss(out_images[:, :3, :, :], target_images[:, :3, :, :])
        else:
            loss = self.loss_weight[0] * self.mse(out_images, target_images)
        return loss


class GradientPriorLoss(nn.Module):
    def __init__(self, ):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss()

    def forward(self, out_images, target_images):
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)
        return self.func(map_out, map_target)

    @staticmethod
    def gradient_map(x):
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)+1e-6, 0.5)
        return xgrad


if __name__ == '__main__':
    im1=Image.open('../tt.jpg')
    im1=transforms.ToTensor()(im1)
    im1=im1.unsqueeze(0)
    im2 = Image.open('../tt1.jpg')
    im2 = transforms.ToTensor()(im2)
    im2 = im2.unsqueeze(0)
    embed()
