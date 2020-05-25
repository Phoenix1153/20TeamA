import torch.nn as nn
import torch.nn.functional as F
from . import binaryfunction
import torch
import math


class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float()
        self.t = torch.tensor([0.1]).float()

    def forward(self, input):
        bw = self.weight
        a = input
        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        ba = binaryfunction.BinaryQuantize().apply(a, self.k, self.t)
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output