import torch.nn as nn
import torch.nn.functional as F
from . import binaryfunction
import torch
import math
import pdb


class BCWNConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BCWNConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        bw = self.weight
        ba = input
        sw = bw.abs().view(bw.size(0), -1).mean(-1).view(bw.size(0), 1, 1, 1).detach()
        sa = ba.abs().view(ba.size(0), -1).mean(-1).view(ba.size(0), 1, 1, 1).detach()
        bw = binaryfunction.BinaryQuantize().apply(bw)
        ba = binaryfunction.BinaryQuantize().apply(ba)
        bw = bw * sw
        ba = ba * sa
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output
