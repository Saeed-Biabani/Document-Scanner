import torch
from torch import nn
import torch.nn.functional as F
from .Layers import *


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel):
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv = nn.Sequential(*[
            ConvLayer(mid_channel, out_channel),        
            ConvLayer(out_channel, out_channel),        
        ])

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.concat((x, skip), 1)

        return self.conv(x)


def InitDecoder(kernels):
    list_ = nn.ModuleList()
    
    for kernel_indx in range(len(kernels)-1):
        in_ = kernels[kernel_indx]
        out = kernels[kernel_indx+1]
        list_.append(Decoder(in_, out, in_+in_//2))

    return list_