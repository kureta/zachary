import torch
import torch.nn as nn


def get_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("No CUDA! Machine slow! Ugh!")
    return device


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_model(model):
    for m in model.modules():
        classname = m.__class__.__name__

        if 'Conv' in classname:
            try:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.normal_(m.bias, mean=0, std=0.02)
            except AttributeError:
                pass

        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight, mean=0, std=0.02)
            nn.init.normal_(m.bias, mean=0, std=0.001)
