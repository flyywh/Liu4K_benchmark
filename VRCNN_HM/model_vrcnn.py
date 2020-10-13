import math
import torch
import numpy as np
from torch import nn
from PIL import JpegPresets
import common

class VSRCNN(torch.nn.Module):
    def __init__(self, n_channels, d=56, s=12, m=4):
        super(VSRCNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=5, stride=1, padding=2), nn.ReLU())

        self.conv2_part1 = torch.nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.conv2_part2 = torch.nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), nn.ReLU())

        self.conv3_part1 = torch.nn.Sequential(nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv3_part2 = torch.nn.Sequential(nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0), nn.ReLU())

        self.conv4 = nn.Conv2d(in_channels=48, out_channels=n_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        y0 = self.conv1(x)

        y1_p1 = self.conv2_part1(y0)
        y1_p2 = self.conv2_part2(y0)
        y1_all = torch.cat([y1_p1, y1_p2], 1)

        y2_p1 = self.conv3_part1(y1_all)
        y2_p2 = self.conv3_part2(y1_all)
        y2_all = torch.cat([y2_p1, y2_p2], 1)

        out = self.conv4(y2_all) + x

        return out

    def weight_init(self):
        '''
        Initial the weights.
        :return:
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0.0, 0.2)
                m.weight.data.normal_(0.0, sqrt(2/m.out_channels/m.kernel_size[0]/m.kernel_size[0])) # MSRA
                # nn.init.xavier_normal(m.weight) # Xavier
                if m.bias is not None:
                    m.bias.data.zero_()


