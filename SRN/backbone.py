import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=2 if stride == (1, 1) else kernel_size,
                              dilation=2 if stride == (1, 1) else 1,
                              stride=stride,
                              padding=(kernel_size - 1) // 2,
                              groups=groups)

        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x

class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, is_first=False):
        super(ShortCut, self).__init__()
        self.use_conv = True

        if in_channels != out_channels or stride != 1 or is_first == True:
            if stride == (1, 1):
                self.conv = ConvBNLayer(in_channels, out_channels, 1, 1)
            else:
                self.conv = ConvBNLayer(in_channels, out_channels, 1, stride)
        else:
            self.use_conv = False

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()
        self.conv0 = nn.Sequential(ConvBNLayer(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=1),
                                   nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(ConvBNLayer(in_channels=out_channels,
                                               out_channels=out_channels,
                                               kernel_size=3,
                                               stride=stride),
                                   nn.ReLU(inplace=True))

        self.conv2 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels * 4,
                                 kernel_size=1)

        self.short = ShortCut(in_channels=in_channels,
                              out_channels=out_channels * 4,
                              stride=stride,
                              is_first=False)

        self.out_channels = out_channels * 4

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.short(x)
        y = F.relu(y)
        return y


class ResNetFPN(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super(ResNetFPN, self).__init__()
        self.depth = [3, 4, 6, 3]
        stride_list = [(2, 2), (2, 2), (1, 1), (1, 1)]
        num_filters = [64, 128, 256, 512]
        self.F = []
        self.conv = nn.Sequential(ConvBNLayer(in_channels=in_channels,
                                              out_channels=64,
                                              kernel_size=7,
                                              stride=2),
                                  nn.ReLU(inplace=True))

        self.block_list = nn.ModuleList()
        in_ch = 64

        for block in range(len(self.depth)):
            for i in range(self.depth[block]):
                self.block_list.append(BottleneckBlock(in_channels=in_ch,
                                                       out_channels=num_filters[block],
                                                       stride=stride_list[block] if i == 0 else 1))

                in_ch = num_filters[block] * 4
            self.F.append(self.block_list)

        out_ch_list = [in_ch // 4, in_ch // 2, in_ch]
        self.base_block = nn.ModuleList()
        self.conv_trans = []
        self.bn_block = []
        for i in [-2, -3]:
            in_channels = out_ch_list[i + 1] + out_ch_list[i]

            self.base_block.append(nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_ch_list[i],
                                             kernel_size=1,
                                             bias=True))

            self.base_block.append(nn.Conv2d(in_channels=out_ch_list[i],
                                             out_channels=out_ch_list[i],
                                             kernel_size=3,
                                             padding=1,
                                             bias=True))


            self.base_block.append(nn.Sequential(nn.BatchNorm2d(num_features=out_ch_list[i]),
                                                 nn.ReLU()))

        self.base_block.append(nn.Conv2d(in_channels=out_ch_list[i],
                                         out_channels=512,
                                         kernel_size=1,
                                         bias=True))


        self.out_channels = 512

    def forward(self, x):
        x = self.conv(x)
        fpn_list = []
        F = []
        for i in range(len(self.depth)):
            fpn_list.append(np.sum(self.depth[:i + 1]))

        for i, block in enumerate(self.block_list):
            x = block(x)
            for number in fpn_list:
                if i + 1 == number:
                    F.append(x)

        base = F[-1]

        j = 0
        for i, block in enumerate(self.base_block):
            if i % 3 == 0 and i < 6:
                j = j + 1
                b, c, w, h = F[-j - 1].shape
                if [w, h] == list(base.shape[2:]):
                    base = base
                else:
                    base = self.conv_trans[j - 1](base)
                    base = self.bn_block[j - 1](base)
                base = torch.cat([base, F[-j - 1]], dim =1)
            base = block(base)
        return base