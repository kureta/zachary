import torch.nn as nn


def initialize_model(model):
    for m in model.modules():
        classname = m.__class__.__name__

        if 'Conv' in classname:
            try:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.normal_(m.bias, mean=0, std=0.001)
            except AttributeError:
                pass

        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight, mean=0, std=0.02)
            nn.init.normal_(m.bias, mean=0, std=0.001)
