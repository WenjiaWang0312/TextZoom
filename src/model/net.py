import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed

sys.path.append('./')
sys.path.append('../')
from .recognizer.tps_spatial_transformer import TPSSpatialTransformer
from .recognizer.stn_head import STNHead
from . import AttentionalImageLoss


class TextZoom(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=False, rrb_nums=6):
        super(TextZoom, self).__init__()
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.rrb_nums = rrb_nums
        for i in range(rrb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(64))

        setattr(self, 'block%d' % (rrb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64)
                ))
        block_ = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (rrb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height//scale_factor, width//scale_factor]
        tps_outputsize = [height//scale_factor, width//scale_factor]
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
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}
        for i in range(self.rrb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.rrb_nums + 3)] = getattr(self, 'block%d' % (self.rrb_nums + 3)) \
            ((block['1'] + block[str(self.rrb_nums + 2)]))
        output = torch.tanh(block[str(self.rrb_nums + 3)])
        return output


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.PReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.PReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        x, _ = self.gru(x)
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


if __name__ == '__main__':
    img = torch.zeros(7, 3, 16, 64)
    embed()
