import torch
import torch.nn as nn
import math
from IPython import embed


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output,identity_data)
        return output


class EDSR(nn.Module):
    def __init__(self, scale_factor=2):
        super(EDSR, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = MeanShift(rgb_mean, -1)

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, 32)

        self.conv_mid = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        upscale_block = []
        for i in range(int(math.log2(scale_factor))):
            upscale_block.append(
                nn.Conv2d(in_channels=256, out_channels=256*4, kernel_size=3, stride=1, padding=1, bias=False),
            )
            upscale_block.append(nn.PixelShuffle(2))
        self.upscale = nn.Sequential(
            *upscale_block
        )

        self.conv_output = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.sub_mean(x)
        out = self.conv_input(out)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out,residual)
        out = self.upscale(out)
        out = self.conv_output(out)
        out = self.add_mean(out)
        return out


if __name__=='__main__':
    embed()