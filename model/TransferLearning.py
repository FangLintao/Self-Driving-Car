#!/usr/bin/env python
# coding: utf-8

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class TLearning(nn.Module):
    def __init__(self):
        super(TLearning,self).__init__()
        self.ResNet = models.resnet50(pretrained=True, progress=True)
        
        self.fc1 = nn.Linear(in_features =1000, out_features = 512, bias=True)
        self.fc2 = nn.Linear(in_features = 512, out_features = 256, bias=True)
        self.fc3 = nn.Linear(in_features = 256, out_features = 64, bias=True)
        self.fc4 = nn.Linear(in_features = 64, out_features = 1, bias=True)
    def forward(self, Input):
        image = self.ResNet(Input)
        # input size = (1,3,224,224)-(Batches, Channels, Height, Width)
        # output size = (1, 1000)-(Batches, Feature)
        image = F.relu(self.fc1(image))
        # input size = (1,1000)-(batches, features)
        # output size = (1,512)-(batches, features)
        image = F.relu(self.fc2(image))
        # input size = (1,512)-(batches, features)
        # output size = (1,256)-(batches, features)
        image = F.relu(self.fc3(image))
        # input size = (1,256)-(batches, features)
        # output size = (1,64)-(batches, features)
        angle = self.fc4(image)
        # input size = (1,64)-(batches, features
        # output size = (1,1)-(batches, features)
        return angle

