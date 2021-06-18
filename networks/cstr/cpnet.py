import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from .cbam import CBAM
from .nonlocalblock import NonLocalBlock
from ..dcn import DeformableConv2d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels= in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CPNet(nn.Module):
    def __init__(self):
        self.inplanes = 96
        super(CPNet, self).__init__()
        layers = [1,4,7,5,3]

        # different model config between ImageNet and CIFAR 

        self.conv1_1 = ConvModule(1, 48, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = ConvModule(48, 96, kernel_size=3, stride=1, padding=1)  
 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(BasicBlock, 192, layers[0], stride=1, att_type="CBAM")
        self.conv2 = ConvModule(192, 192, kernel_size=3, stride=1, padding=1)

        self.nonlocal3 = NonLocalBlock(in_channels=192)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer3 = self._make_layer(BasicBlock, 384, layers[1], stride=1, att_type="CBAM")
        self.conv3 = ConvModule(384, 384, kernel_size=3, stride=1, padding=1)


        self.nonlocal4 = NonLocalBlock(in_channels=384)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=(2, 2))
        self.layer4_1 = self._make_layer(BasicBlock, 768, layers[2], stride=1, att_type="CBAM")
        self.conv4 = ConvModule(768, 768, kernel_size=3, stride=1, padding=1)
        self.layer4_2 = self._make_layer(BasicBlock, 768, layers[3], stride=1, att_type="CBAM")

        self.nonlocal5 = NonLocalBlock(in_channels=768)
        self.conv5_1 = ConvModule(768, 768, kernel_size=2, stride=(2,1))
        self.layer5 = self._make_layer(BasicBlock, 768, layers[4], stride=1, att_type="CBAM")
        self.conv5_2 = DeformableConv2d(768, 768, kernel_size=3, stride=1, padding=1)
 

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1_1(x)
        x = self.conv1_2(x)

        x = self.pool2(x)
        x = self.layer2(x)
        x = self.conv2(x)

        x = self.nonlocal3(x)
        x = self.pool3(x)
        x = self.layer3(x)
        x = self.conv3(x)

        x = self.nonlocal4(x)
        x = self.pool4(x)
        x = self.layer4_1(x)
        x = self.conv4(x)
        x = self.layer4_2(x)

        x = self.nonlocal5(x)
        x = self.conv5_1(x)
        x = self.layer5(x)
        x = self.conv5_2(x)
        return x