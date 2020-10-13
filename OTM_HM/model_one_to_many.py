import math
import torch
import numpy as np
from torch import nn
from PIL import JpegPresets


def softRound(x):
    return x
#    r = torch.round(x).detach_()
#    return r + (x - r) ** 3

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )

    def forward(self, i):
        return 0.01*self.conv0(i)+i

class One_to_many(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.PReLU(init=0.1),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.PReLU(init=0.1)
        )

        self.conv1 = ResBlock()
        self.conv2 = ResBlock()
        self.conv3 = ResBlock()
        self.conv4 = ResBlock()
        self.conv5 = ResBlock()
        self.conv6 = ResBlock()
        self.conv7 = ResBlock()
        self.conv8 = ResBlock()
        self.conv9 = ResBlock()
        self.conv10 = ResBlock()
        self.conv11 = ResBlock()
        self.conv12 = ResBlock()
        self.conv13 = ResBlock()
        self.conv14 = ResBlock()
        self.conv15 = ResBlock()

        self.conv16 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.PReLU(init=0.1)
        )

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key and '.0.' in key:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0


    def forward(self, inputs):
        i = inputs
        x_in = self.conv0(i)

        x = self.conv1(x_in)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
 
        x = self.conv16(x) 

        return i+0.1*x

class dmcnnLoss(nn.Module):
    def __init__(self, theta=0.618, lambd=0.9):
        super(dmcnnLoss, self).__init__()
        self.MSE1 = nn.MSELoss()

    def forward(self, x, target):
        MSE_value = self.MSE1(x, target)
        return  MSE_value
