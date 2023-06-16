import torch
import torch.nn as nn
import torch.nn.functional as F
BN_MOMENTUM = 0.2
class FC3BasicBlock(nn.Module):
    def __init__(self, inchannels, outchannels, stride=1, padding=0):
        super(FC3BasicBlock, self).__init__()

        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM) # default 0.2
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)

        return x