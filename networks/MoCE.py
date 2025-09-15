import numbers
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.arch_utils import *


class Gating(nn.Module):
    def __init__(self, in_c, num_experts):
        super(Gating, self).__init__()
        self.body = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  Rearrange('b c 1 1 -> b c'),
                                  nn.Linear(in_c, num_experts * 2),
                                  nn.GELU(),
                                  nn.Linear(num_experts * 2, num_experts))

    def forward(self, x):
        return self.body(x)


class ConvOP(nn.Module):

    def __init__(self, dim, kernel_size, pad):
        super(ConvOP, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim, bias=False),
                                  nn.GELU(),
                                  nn.Conv2d(dim, dim, kernel_size=1, bias=False))

    def forward(self, x):
        return self.body(x) + x


class DilConv(nn.Module):
    def __init__(self, dim, kernel_size, dilation):
        super(DilConv, self).__init__()
        pad_h = int((kernel_size[0] - 1) // 2 * dilation)
        pad_w = int((kernel_size[1] - 1) // 2 * dilation)
        self.op = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=(pad_h, pad_w), dilation=dilation, groups=dim, bias=False),
                                nn.GELU(),
                                nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False),)

    def forward(self, x):
        return self.op(x) + x


class FreBlock(nn.Module):
    def __init__(self, dim):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(nn.Conv2d(dim, dim,1,1,0),
                                        nn.LeakyReLU(0.1,inplace=True),
                                        nn.Conv2d(dim, dim,1,1,0))
        self.processpha = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Conv2d(dim, dim, 1, 1, 0))

    def forward(self,x):
        _, _, H, W = x.shape
        u = x.clone()
        x = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_freq_spatial + u


class OpLayer(nn.Module):

    def __init__(self, dim):
        super(OpLayer, self).__init__()
        self.Op = nn.ModuleList()

        self.Op.append(ConvOP(dim, kernel_size=(3, 3), pad=(1, 1)))  # Small Rectangular
        self.Op.append(ConvOP(dim, kernel_size=(21, 21), pad=(10, 10)))  # Large Rectangular
        self.Op.append(DilConv(dim, kernel_size=(9, 9), dilation=8))  # Ultra Sparse Rectangular

        self.Op.append(ConvOP(dim, kernel_size=(21, 1), pad=(10, 0)))  # Horizon
        self.Op.append(ConvOP(dim, kernel_size=(1, 21), pad=(0, 10)))  # Vertical
        self.Op.append(FreBlock(dim))  # Frequency
        self.weight = Gating(in_c=dim, num_experts=len(self.Op))

        self.out_ = nn.Conv2d(dim * len(self.Op), dim, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.weight(x)
        weights = F.softmax(out, dim=1)
        states = []
        for i, expert in enumerate(self.Op):
            out = expert(x) * weights[:, i: i + 1, None, None]
            states.append(out)
        return self.out_(torch.cat(states, dim=1))


class MoCE(nn.Module):
    """
    Mixture of Convolutional Experts
    """
    def __init__(self, dim):
        super(MoCE, self).__init__()
        self.norm = LayerNorm(dim, 'WithBias')
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()

        self.layers = nn.Sequential(OpLayer(dim=dim))
        self.out_ = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        u = x.clone()
        x = self.act(self.proj_conv(self.norm(x)))
        out = self.layers(x)
        return self.out_(out) + u
