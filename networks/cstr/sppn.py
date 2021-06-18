import torch
import torch.nn as nn
from .cpnet import ConvModule

class SPPN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SPPN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.batch_max_length = 100
        
        self.convs = nn.ModuleList()
        for i in range(self.batch_max_length):
            conv = ConvModule(in_channels,num_classes,1,stride=1,padding=0)
            self.convs.append(conv)
        
    def forward(self, x):
        x = self.pool(x)
        
        outs = []
        for conv in self.convs:
            out = conv(x)
            out = out.contiguous().view(-1,1,self.num_classes)
            outs.append(out)
        outs = torch.cat(outs, dim=1)
        return outs
        