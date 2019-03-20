from torch import nn

from zachary.model.modules import conv_block, deconv_block


class Discriminator(nn.Module):
    def __init__(self, n_channels, n_features, hidden_size):
        super(Discriminator, self).__init__()

        # 256
        self.conv1 = nn.Sequential(nn.Conv1d(n_channels, n_features, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(True))
        # 256
        self.conv2 = conv_block(n_features, n_features)
        # 128
        self.conv3 = conv_block(n_features, n_features * 2)
        # 64
        self.conv4 = conv_block(n_features * 2, n_features * 3)
        # 32
        self.encode = nn.Conv1d(n_features * 3, hidden_size, kernel_size=1, stride=1, padding=0)
        self.decode = nn.Conv1d(hidden_size, n_features, kernel_size=1, stride=1, padding=0)
        # 32
        self.deconv4 = deconv_block(n_features, n_features)
        # 64
        self.deconv3 = deconv_block(n_features, n_features)
        # 128
        self.deconv2 = deconv_block(n_features, n_features)
        # 256
        self.deconv1 = nn.Sequential(nn.Conv1d(n_features, n_features, kernel_size=3, stride=1, padding=1),
                                     nn.ELU(True),
                                     nn.Conv1d(n_features, n_features, kernel_size=3, stride=1, padding=1),
                                     nn.ELU(True),
                                     nn.Conv1d(n_features, n_channels, kernel_size=3, stride=1, padding=1),
                                     nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.encode(out4)
        dout5 = self.decode(out5)
        dout4 = self.deconv4(dout5)
        dout3 = self.deconv3(dout4)
        dout2 = self.deconv2(dout3)
        dout1 = self.deconv1(dout2)
        return dout1
