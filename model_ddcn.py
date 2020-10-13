import math
import torch
import numpy as np
from torch import nn
from PIL import JpegPresets
import torch.nn.functional as F

def softRound(x):
    return x
#    r = torch.round(x).detach_()
#    return r + (x - r) ** 3

def initDctCoeff(blocksize):
    dm = []
    for i in range(blocksize):
        col = []
        for j in range(blocksize):
            if i == 0:
                c = math.sqrt(1 / blocksize)
            else:
                c = math.sqrt(2 / blocksize)
            col.append(c * math.cos(math.pi * (j + 0.5) * i / blocksize))
        dm.append(col)
    dm = torch.FloatTensor(dm)
    DCT = nn.Parameter(dm, requires_grad=False)
    iDCT = nn.Parameter(dm.t(), requires_grad=False)
    return DCT, iDCT

class dctLayer(nn.Module):
    def __init__(self, blocksize):
        super().__init__()
        self.DCT, self.iDCT = initDctCoeff(blocksize)
        self.blocksize = blocksize

    def forward(self, inputs):
        b, c, w, h = inputs.shape
        nw, nh = w // self.blocksize, h // self.blocksize

        x = softRound(inputs * 255)
        x = x - 128
        x = x.contiguous().view(b, c, nw, self.blocksize, nh, self.blocksize)
        x = x.transpose(3, 4)
        x = self.DCT @ x @ self.iDCT
        x = x.transpose(3, 4)
        x = x.contiguous().view(b, c, w, h)

        return x

class iDctLayer(nn.Module):
    def __init__(self, blocksize, alpha=0):
        super().__init__()
        self.DCT, self.iDCT = initDctCoeff(blocksize)
        self.blocksize = blocksize

    def forward(self, inputs, target):
        b, c, w, h = inputs.shape
        nw, nh = w // self.blocksize, h // self.blocksize

        x = inputs.contiguous().view(b, c, nw, self.blocksize, nh, self.blocksize)
        x = x.transpose(3, 4)
        x = self.iDCT @ x @ self.DCT
        x = x.transpose(3, 4)
        x = x.contiguous().view(b, c, w, h)
        x = x + 128
        x = x / 255

        return x

class dctSubnet(nn.Module):
    def __init__(self, C, blocksize):
        super().__init__()
        self.DCTlayer = dctLayer(blocksize)
        self.conv0 = nn.Sequential(
            nn.Conv2d(C, 32, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.iDCTlayer = iDctLayer(blocksize)

    def forward(self, inputs):
        x = self.DCTlayer(inputs)
        dct_in = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        dct_out = x + 0.1*dct_in
        x = self.iDCTlayer(dct_out, inputs)
        return x, dct_out

class DDCN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dct_branch8 = dctSubnet(C, 8)

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(init=0.1)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
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
        d8, _ = self.dct_branch8(i)

        x_in = self.conv0(i)
        x = self.conv1(x_in)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)+0.1*x_in

        x = torch.cat([x, 0.1*d8], dim=1)

        x_in2 = self.conv6(x)
        x = self.conv7(x_in2)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)+0.1*x_in2
        x = self.conv11(x)

        return i+0.1*x
