import torch
import torch.nn as nn
from .cpnet import CPNet
from .fpn import FPN
from .sppn import SPPN

class CSTR(nn.Module):
    def __init__(self):
        super(CSTR, self).__init__()
        self.backbone = CPNet()
        self.fpn = FPN([384,768,768], 512, 3)
        self.sppn = SPPN(512,243)

        self.criterion = (
            nn.CrossEntropyLoss()
        )  # without ignore_index=train_dataset.token_to_id[PAD]
    
    def forward(self, x):
        x, features = self.backbone(x)
        x = self.fpn(features[2:])
        x = self.sppn(x)
        return x       