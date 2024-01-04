import torch
from torch import nn
from copy import deepcopy


def create_layer_clones(module: nn.Module, N: int):
    """
    Create N copies of the module.
    :param module: A PyTorch nn.Module subclass.
    :param N: The number of copies to make.
    :return: A new list of modules.
    """
    return nn.ModuleList([deepcopy(module) for _ in range(N)])
