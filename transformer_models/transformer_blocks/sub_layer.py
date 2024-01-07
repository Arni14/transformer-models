from torch import nn

from transformer_models.transformer_blocks.layer_norm import LayerNorm


class SubLayer(nn.Module):
    """
    The SubLayer is an abstract class which can allow for a (seq_len, d_model) input to be transformed
    into another output of the same shape. In the case of the traditional architecture, we consider these
    either to be the PositionalFeedForward, the MultiHeadAttention, or the MaskedMultiHeadAttention
    in the case of the decoder.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__()

        # Instantiate some parameters for downstream classes to utilize (if needed)
        self.d_model = d_model
        self.dropout = dropout

    def forward(self, x):
        pass


class ResidualDropoutSubLayer(nn.Module):
    """
    A simple ResidualDropoutSubLayer is a generic block in any Transformer network. The idea here is to create
    an abstraction that makes building more complex network structures, simpler. So what exactly
    does this SubLayer do? Well, it takes in a given Module (so think: MHA, Feed Forward, or really, anything
    which takes as input a (batch_size, seq_len, d_model) input and outputs a tensor of the same
    shape). The reason for this requirement is that ResidualDropoutSubLayer implements a function as follows:

    ResidualDropoutSubLayer(x) = LayerNorm(x + SubLayer(x))

    What this enables is (a) A residual connection (which enables Identity mapping learning), (b) Dropout which
    ensures that we are learning robust connections, and (c) the LayerNorm which normalizes the outputs to maximize
    gradient stability during the learning procedure.
    """

    def __init__(self, d_model: int, sub_layer: SubLayer):
        super().__init__()

        # Create a LayerNorm for this component
        self.layer_norm = LayerNorm(d_model)

        # Instantiate the SubLayer
        self.sub_layer = sub_layer

    def forward(self, x, **kwargs):
        return self.layer_norm(x + self.sub_layer(x, **kwargs))
