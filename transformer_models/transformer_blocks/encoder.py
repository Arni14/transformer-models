import torch

from transformer_models.utils import create_layer_clones
from transformer_models.transformer_blocks.layer_norm import LayerNorm


class Encoder(torch.nn.Module):

    def __init__(self, encoder_layer: torch.nn.Module, num_copies: int):
        super(Encoder, self).__init__()
        self.layers = create_layer_clones(encoder_layer, num_copies)
        self.norm = LayerNorm(encoder_layer.size)
