import torch


def get_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("No CUDA! Machine slow! Ugh!")
    return device


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
