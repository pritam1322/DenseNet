import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False)
        self.batchnorm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(0.1)

    def forward(self, x):
        out = self.conv(self.relu(self.batchnorm(x)))
        out = torch.cat((x,out),1)


class dense_blocK(nn.Module):    #Bottleneck Layer
    def __init__(self, in_channels, growthRate, **kwargs):
        super(dense_blocK, self).__init__()
        self.in_channels = in_channels
        interChannels = 4 * growthRate
        self.conv1 = nn.Conv2d(in_channels, interChannels, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(in_channels, interChannels, kernel_size = 3, bias = False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.relu = nn.ReLU(0.1)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        out = torch.cat((x,out),1)
        return out

                                        
class Transition_layer(nn.Module):          # Transition layer is used for reducing the resolution layers.
    def __init__(self, in_channels, out_channels):
        super(Transition_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.avg_pool_layer = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.avg_pool_layer(out)
        return out

class DenseNet(nn.Module):
    def __init__(self, growthRate,  depth, reduction , num_class, bottleneck=True):
        super(DenseNet, self).__init__()
        num_blocks = (depth -4) // 3
        if bottleneck :
            num_class //= 2

        in_channels = 2*growthRate
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, stride=2,padding=1)
        self.dense1 = self.make_layers(in_channels, growthRate, num_blocks, bottleneck)
        in_channels += num_blocks*growthRate
        out_channels = int(math.floor(in_channels*reduction))
        self.tran1 = Transition_layer(in_channels, out_channels)


        in_channels = out_channels
        self.dense2 = self.make_layers(in_channels, growthRate, num_blocks, bottleneck)
        in_channels += num_blocks*growthRate
        out_channels = int(math.floor(in_channels*reduction))
        self.tran2 = Transition_layer(in_channels, out_channels)

        in_channels = out_channels
        self.dense3 = self.make_layers(in_channels, growthRate, num_blocks, bottleneck)
        in_channels += num_blocks*growthRate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, num_class)

    def make_layers(self, in_channels, out_channels, num_blocks, bottleneck):
        layers = []
        for i in range(num_blocks):
            if bottleneck:
                layers.append(dense_blocK(in_channels, out_channels))
            else :
                layers.append(CNN(in_channels, out_channels))
            in_channels += out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tran1(self.dense1(x))
        x = self.tran2(self.dense2(x))
        x = self.dense3(x)
        out =  torch.squeeze(F.avg_pool2d(F.relu(self.bn1(x)), 8))
        out = F.log_softmax(self.fc(out))
        return out

model = DenseNet(32, 100, 0.5, 10)
print(model)
