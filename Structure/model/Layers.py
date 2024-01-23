from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel = (3, 3), stride = 1, padding = "same"):
        super(ConvLayer, self).__init__()

        self.layer = nn.Sequential(*[
            nn.Conv2d(in_channel, out_channel, kernel, stride = stride, padding = padding, bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
        ])

    def forward(self, x):
        return self.layer(x)