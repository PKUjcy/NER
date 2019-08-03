# -*- coding: utf-8 -*-
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math
from .basic_module import BasicModule
from torchvision import models
from torchvision.models.vgg import VGG
class BaseNet(BasicModule):
    def __init__(self, num_init_features = 512, num_classes=10000):
        super(BaseNet, self).__init__()
        # input [N, C, H, W]
        # First convolution
        #频带卷积结构，输入nx23,经过75x12的卷积核，得到[C=64, H=12, W=n-d+1]的特征
        #再经过12x1的池化操作得到[64, 1, n-d+1 x1 ]
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels= num_init_features,
                                kernel_size=(12, 75), stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=(12,1), stride=(1,1))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=num_init_features, out_channels=1024,
                                kernel_size=(1,3), stride=1,bias=False)),
            ('norm1', nn.BatchNorm2d(1024)),
            ('relu1', nn.ReLU(inplace=True)),
            #('pool0', nn.AvgPool2d(kernel_size=(12,1), stride=(1,1))),
        ]))
        #self.spp = SPP([32,16,10,8,6,4,2,1])
        self.fc0 = nn.Linear(1024, 2048)
        self.fc1 = nn.Linear(2048, num_classes)
    def forward(self, x):
        # input [N, C, H, W] (W = 396)
        x = self.features(x) # [N, 512, 1, W - 75 + 1]
        x = self.conv(x) #  [N, 1024, 1, W - 75 +1 - 3 + 1]
        x = F.avg_pool2d(x, kernel_size = x.size()[2:]).view(x.size()[0], -1) # [N, 1024]
        feature = F.relu(self.fc0(x))
        x = self.fc1(feature)

        return x, feature

class CQTBaseNet(BasicModule):
    def __init__(self):
        super().__init__()
        # input N, C, 72, L
        # First convolution
        #频带卷积结构，输入nx84,经过75x12的卷积核，得到[N, C, 61, L]的特征
        #再经过12x1的池化操作得到[64, 1, n-d+1 x1 ]
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels= 32,kernel_size=(36, 75),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12,3), 
                                stride=(1,1), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), 
                                stride=(1,1), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(1,3), bias=False)),
            ('norm0', nn.BatchNorm2d(256)),
            ('relu0', nn.ReLU(inplace=True)),
            
            ('conv1', nn.Conv2d(in_channels=256, out_channels=512,kernel_size=(1,3), bias=False)),
            ('norm1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),
            
            ('conv2', nn.Conv2d(in_channels=512, out_channels=1024,kernel_size=(1,3), bias=False)),
            ('norm2', nn.BatchNorm2d(1024)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,32))),
        ]))
        #self.spp = SPP([32,16,10,8,6,4,2,1])
        self.fc0 = nn.Linear(1024*32, 300)
        self.fc1 = nn.Linear(300, 10000)
    def forward(self, x):
        # input [N, C, H, W] (W = 396)
        N = x.size()[0]
        x = self.features(x) # [N, 128, 1, W - 75 + 1]
        x32 = self.conv(x) #  [N, 256, 1, W - 75 +1 - 3 + 1]
        #x = SPP(x, [32,16,10,8,6,4,2,1]) # [N, 256, 1, sum()=79]
        x = x32.view(N,-1)
        feature = self.fc0(x)
        x = self.fc1(feature)
        return x, feature,x32