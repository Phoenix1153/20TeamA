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
        w = self.weight
        a = input
        bw = w
        sa1 = a.abs().mean()
        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        ba1 = binaryfunction.BinaryQuantize().apply(a, self.k, self.t)
        ba1 = ba1 * sa1
        a2 = a - ba1
        sa2 = a2.abs().mean()
        ba2 = binaryfunction.BinaryQuantize().apply(a2, self.k, self.t)
        ba2 = ba2 * sa2 + ba1
        output = F.conv2d(ba2, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output