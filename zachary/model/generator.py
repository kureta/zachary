import torch
from torch import nn

from zachary.model.modules import unet_block


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nf):
        super(Generator, self).__init__()

        # input is 256 x 256
        layer_idx = 1
        name = f'layer{layer_idx}'
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv1d(in_channels, nf, 4, 2, 1, bias=False))
        # input is 128 x 128
        layer_idx += 1
        name = f'layer{layer_idx}'
        layer2 = unet_block(nf, nf * 2, name, transposed=False, batch_norm=True, relu=False, dropout=False)
        # input is 64 x 64
        layer_idx += 1
        name = f'layer{layer_idx}'
        layer3 = unet_block(nf * 2, nf * 4, name, transposed=False, batch_norm=True, relu=False, dropout=False)
        # input is 32
        layer_idx += 1
        name = f'layer{layer_idx}'
        layer4 = unet_block(nf * 4, nf * 8, name, transposed=False, batch_norm=True, relu=False, dropout=False)
        # input is 16
        layer_idx += 1
        name = f'layer{layer_idx}'
        layer5 = unet_block(nf * 8, nf * 8, name, transposed=False, batch_norm=True, relu=False, dropout=False)
        # input is 8
        layer_idx += 1
        name = f'layer{layer_idx}'
        layer6 = unet_block(nf * 8, nf * 8, name, transposed=False, batch_norm=True, relu=False, dropout=False)
        # input is 4
        layer_idx += 1
        name = f'layer{layer_idx}'
        layer7 = unet_block(nf * 8, nf * 8, name, transposed=False, batch_norm=True, relu=False, dropout=False)
        # input is 2 x  2
        layer_idx += 1
        name = f'layer{layer_idx}'
        layer8 = unet_block(nf * 8, nf * 8, name, transposed=False, batch_norm=False, relu=False, dropout=False)

        # NOTE: decoder
        # input is 1
        name = f'dlayer{layer_idx}'
        d_inc = nf * 8
        dlayer8 = unet_block(d_inc, nf * 8, name, transposed=True, batch_norm=True, relu=True, dropout=True)

        # import pdb; pdb.set_trace()
        # input is 2
        layer_idx -= 1
        name = f'dlayer{layer_idx}'
        d_inc = nf * 8 * 2
        dlayer7 = unet_block(d_inc, nf * 8, name, transposed=True, batch_norm=True, relu=True, dropout=True)
        # input is 4
        layer_idx -= 1
        name = f'dlayer{layer_idx}'
        d_inc = nf * 8 * 2
        dlayer6 = unet_block(d_inc, nf * 8, name, transposed=True, batch_norm=True, relu=True, dropout=True)
        # input is 8
        layer_idx -= 1
        name = f'dlayer{layer_idx}'
        d_inc = nf * 8 * 2
        dlayer5 = unet_block(d_inc, nf * 8, name, transposed=True, batch_norm=True, relu=True, dropout=False)
        # input is 16
        layer_idx -= 1
        name = f'dlayer{layer_idx}'
        d_inc = nf * 8 * 2
        dlayer4 = unet_block(d_inc, nf * 4, name, transposed=True, batch_norm=True, relu=True, dropout=False)
        # input is 32
        layer_idx -= 1
        name = f'dlayer{layer_idx}'
        d_inc = nf * 4 * 2
        dlayer3 = unet_block(d_inc, nf * 2, name, transposed=True, batch_norm=True, relu=True, dropout=False)
        # input is 64
        layer_idx -= 1
        name = f'dlayer{layer_idx}'
        d_inc = nf * 2 * 2
        dlayer2 = unet_block(d_inc, nf, name, transposed=True, batch_norm=True, relu=True, dropout=False)
        # input is 128
        layer_idx -= 1
        name = f'dlayer{layer_idx}'
        dlayer1 = nn.Sequential()
        d_inc = nf * 2
        dlayer1.add_module(f'{name}.relu', nn.ReLU(inplace=True))
        dlayer1.add_module(f'{name}.tconv', nn.ConvTranspose1d(d_inc, out_channels, 4, 2, 1, bias=False))
        dlayer1.add_module(f'{name}.tanh', nn.Tanh())

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8
        self.dlayer8 = dlayer8
        self.dlayer7 = dlayer7
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        dout8 = self.dlayer8(out8)
        dout8_out7 = torch.cat([dout8, out7], 1)
        dout7 = self.dlayer7(dout8_out7)
        dout7_out6 = torch.cat([dout7, out6], 1)
        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1
