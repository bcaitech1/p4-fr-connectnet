import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels= in_channels, out_channels=in_channels//2, kernel_size=1, stride= 1)
        self.conv2 = nn.Conv2d(in_channels= in_channels, out_channels=in_channels//2, kernel_size=1, stride= 1)
        self.conv3 = nn.Conv2d(in_channels= in_channels, out_channels=in_channels//2, kernel_size=1, stride= 1)
        self.conv4 = nn.Conv2d(in_channels= in_channels//2, out_channels=in_channels, kernel_size=1, stride= 1)

    def forward(self, x):
        b,c,h,w = x.size()
        
        x1 = self.conv1(x).view(b,h*w,-1)
        x2 = self.conv2(x).view(b,-1,h*w)

        x3 = torch.matmul(x1,x2)
        x3 = F.softmax(x3)

        y = self.conv3(x)
        y = y.view(b,h*w,-1)

        y = torch.matmul(x3,y)
        y = y.view(b,-1,h,w)

        y = self.conv4(y)
        
        return y