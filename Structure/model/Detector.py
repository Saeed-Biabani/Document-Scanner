from torch import nn
from .Encoder import *
from .Decoder import *


class Detector(nn.Module):
    def __init__(self, config : dict):
        super(Detector, self).__init__()

        self.encoder_seq = InitEncoder(config["enc_channel"])
        self.decoder_seq = InitDecoder(config["dec_channel"])
        
        self.out = nn.Conv2d(1, 1, 1, padding = "same")

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder_seq:
            x = layer(x)
            skip_connections.append(x)
            x = nn.AvgPool2d(2)(x)

        for layer, skip in zip(self.decoder_seq, skip_connections[::-1]):
            x = layer(x, skip)

        return self.out(x)



def GetModel(opt):
    return Detector(opt["model_config"])