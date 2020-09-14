import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as d_sets
from torch.utils.data import DataLoader as d_loader
import matplotlib.pyplot as plt
from PIL import Image
from IPython import embed
import sys
sys.path.append('./')
from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead


class SRCNN(nn.Module):
    def __init__(self, scale_factor=2, in_planes=3, STN=False, height=32, width=128):
        super(SRCNN, self).__init__()
        self.upscale_factor = scale_factor
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, in_planes, kernel_size=5, padding=2)
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height, width]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=3,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        if self.stn:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upscale_factor)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out


if __name__=='__main__':
    embed()