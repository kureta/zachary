from torch import nn


class Flatten(nn.Module):
    def __init__(self, axis=-1):
        super(Flatten, self).__init__()
        self.axis = axis

    def forward(self, x):
        return x.squeeze(self.axis)


class Expand(nn.Module):
    def __init__(self, axis=-1):
        super(Expand, self).__init__()
        self.axis = axis

    def forward(self, x):
        return x.unsqueeze(self.axis)


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        nn.ELU(True),
        nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        nn.ELU(True),
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.AvgPool1d(kernel_size=2, stride=2))


def deconv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ELU(True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ELU(True),
        Expand(-1),
        nn.UpsamplingNearest2d(scale_factor=(2, 1)),
        Flatten(-1)
    )


def unet_block(in_channels, out_channels, name, transposed=False, batch_norm=True, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module(f'{name}.relu', nn.ReLU(inplace=True))
    else:
        block.add_module(f'{name}.leakyrelu', nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module(f'{name}.conv', nn.Conv1d(in_channels, out_channels, 4, 2, 1, bias=False))
    else:
        block.add_module(f'{name}.tconv', nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1, bias=False))
    if batch_norm:
        block.add_module(f'{name}.bn', nn.BatchNorm1d(out_channels))
    if dropout:
        block.add_module(f'{name}.expand', Expand(-1))
        block.add_module(f'{name}.dropout', nn.Dropout2d(0.5, inplace=True))
        block.add_module(f'{name}.flatten', Flatten(-1))
    return block
