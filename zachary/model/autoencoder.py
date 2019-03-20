from torch import nn


class Codec(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(Codec, self).__init__()

        if in_channels > out_channels:
            features = [in_channels // (2 ** n) for n in range(n_layers + 1)]
        else:
            features = [in_channels * (2 ** n) for n in range(n_layers + 1)]

        self.model = nn.Sequential()

        # Except the last pair
        for idx, (in_ch, out_ch) in enumerate(zip(features, features[1:])):
            num = idx + 1
            self.model.add_module(f'conv{num}', nn.Linear(in_ch, out_ch))
            self.model.add_module(f'leaky_relu{num}', nn.LeakyReLU())
            self.model.add_module(f'dropout{num}', nn.Dropout(0.2))

        self.model.add_module(f'conv{n_layers + 1}', nn.Linear(features[-1], out_channels))

    def forward(self, x):
        return self.model(x)


class Autoencoder(nn.Module):
    def __init__(self, channels, z_channels, n_layers):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            Codec(channels, z_channels, n_layers),
            nn.Tanh()
        )
        self.decoder = Codec(z_channels, channels, n_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))
