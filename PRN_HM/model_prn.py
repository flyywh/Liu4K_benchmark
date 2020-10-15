
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable
import SI
class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x
        
class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate,PredRDB):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(2*nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
    self.conv_feat_1x1 = nn.Conv2d(2*nChannels_, nChannels_, kernel_size=1, padding=0, bias=False)
    self.feature = 0
    self.PredRDB = PredRDB
    # self.pred = PredRDB.info
    # self.info
  def forward(self, x):
    out = self.dense_layers(x)
    if self.PredRDB != 0:
        out = torch.cat((self.PredRDB.feature,out),1)
    else:
        out = torch.cat((out,out),1)
    self.feature = self.conv_feat_1x1(out) 
    out = self.conv_1x1(out)

    out = out + x
    return out
# Residual Dense Network
class PRN(nn.Module):
    def __init__(self):
        super(PRN, self).__init__()
        nChannel = 1
        nDenselayer = 6
        nFeat = 64
        scale = 1
        growthRate = 32
        self.nRDB = 10
        nRDB = self.nRDB

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs
        # self.RDB0 = RDB(nFeat,nDenselayer,growthRate)
        # self.RDBs = [RDB(nFeat,nDenselayer,growthRate) for i in range(0,20)]
        RDBs = []
        for i in range(0,nRDB):
            if i != 0:
                RDBs.append(RDB(nFeat,nDenselayer,growthRate,RDBs[i-1]))
            else:
                RDBs.append(RDB(nFeat,nDenselayer,growthRate,0))
        self.RDBs = nn.Sequential(*RDBs)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*nRDB, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat*scale*scale, kernel_size=3, padding=1, bias=True)
        self.upsample = sub_pixel(scale)
        # conv 
        self.conv3 = nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True)

        self.SI0 = SI.SI()
        self.SI1 = SI.SI()
        self.SI2 = SI.SI()
        self.SI3 = SI.SI()
    def forward(self, x,cuave0,cuave1,cuave2,cuave3):
        nRDB = self.nRDB
        F_  = self.conv1(x)
        F_0 = self.conv2(F_)
        cu0 = self.SI0(cuave0)
        cu1 = self.SI1(cuave1)
        cu2 = self.SI2(cuave2)
        cu3 = self.SI3(cuave3)
        F = [F_0]
        for i in range(0,nRDB):
            tmp = self.RDBs[i](F[i])
            if i == 1:
                tmp = tmp + cu0
            elif i == 3:
                tmp = tmp + cu1
            elif i == 5:
                tmp = tmp + cu2
            elif i == 7:
                tmp = tmp + cu3
            F.append(tmp)

        FF = F[1]
        for i in range(2,nRDB+1):
            FF = torch.cat((FF,F[i]),1)
        # print(FF.shape)
        # F_2 = self.RDB2(F_1)
        # F_3 = self.RDB3(F_2)     
        # FF = torch.cat((F_1,F_2,F_3), 1)
        FdLF = self.GFF_1x1(FF)         
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        us = self.conv_up(FDF)
        us = self.upsample(us)

        output = self.conv3(us)


        return output
