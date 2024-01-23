import torch
from torch import nn
from .Layers import *

class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder, self).__init__()
        
        self.spatial_conv_1 = ConvLayer(in_channel, out_channel, (1, 9))
        self.spatial_conv_2 = ConvLayer(in_channel, out_channel, (9, 1))
        self.out = ConvLayer(out_channel*2, out_channel)
    
    def forward(self, x):
        __1 = self.spatial_conv_1(x)
        __2 = self.spatial_conv_2(x)
        
        x = torch.concat((__1, __2), dim = 1)
        
        return self.out(x)


def InitEncoder(kernels):
    list_ = nn.ModuleList()

    for kernel_indx in range(len(kernels)-1):
        list_.append(Encoder(kernels[kernel_indx], kernels[kernel_indx+1]))
    
    return list_