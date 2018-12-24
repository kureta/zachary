from collections import OrderedDict

from torch import nn


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SeparableConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConvTranspose1d, self).__init__()

        self.conv1 = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, groups=in_channels,
                                        bias=bias)
        self.pointwise = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                            bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Flatten(nn.Module):
    def __init__(self, axis=0):
        super(Flatten, self).__init__()
        self.axis = axis

    def forward(self, x):
        return x.squeeze(self.axis)


class Expand(nn.Module):
    def __init__(self, axis=0):
        super(Expand, self).__init__()
        self.axis = axis

    def forward(self, x):
        return x.unsqueeze(self.axis)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        channels = [513, 256, 128, 32]

        layers = OrderedDict([
            ('init', nn.Conv2d(in_channels=513, out_channels=513, kernel_size=(3, 2), padding=(1, 0))),
            ('relu_00', nn.SELU()),
            ('squee', Flatten(-1)),
            ('conv1d_01', SeparableConv1d(channels[0], channels[1], kernel_size=3, padding=1)),
            ('relu_01', nn.SELU()),
            ('conv1d_02', SeparableConv1d(channels[1], channels[2], kernel_size=3, padding=1)),
            ('relu_02', nn.SELU()),
            ('conv1d_03', SeparableConv1d(channels[2], channels[3], kernel_size=3, padding=1)),
            # ('sigmoid_01', nn.Sigmoid()),
        ])

        self.block = nn.Sequential(layers)

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        channels = [32, 128, 256, 513]

        layers = OrderedDict([
            ('conv1d_01', SeparableConvTranspose1d(channels[0], channels[1], kernel_size=3, padding=1)),
            ('relu_01', nn.SELU()),
            ('conv1d_02', SeparableConvTranspose1d(channels[1], channels[2], kernel_size=3, padding=1)),
            ('relu_02', nn.SELU()),
            ('conv1d_03', SeparableConvTranspose1d(channels[2], channels[3], kernel_size=3, padding=1)),
            ('relu_03', nn.SELU()),
            ('unsquee', Expand(-1)),
            ('finit', nn.ConvTranspose2d(in_channels=513, out_channels=513, kernel_size=(3, 2), padding=(1, 0))),
            ('sigmoid_01', nn.Sigmoid()),
        ])

        self.block = nn.Sequential(layers)

    def forward(self, x):
        return self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        layers = OrderedDict([
            ('encoder', self.encoder),
            ('decoder', self.decoder),
        ])

        self.block = nn.Sequential(layers)

    def forward(self, x):
        return self.block(x)
