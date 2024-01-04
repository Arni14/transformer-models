import torch
from torch import nn

import numpy as np

# Input: Output of Embedding Layer (batch_size, seq_len, d_model) = (# Samples, n, d_model)
# Output: Output of Positional Encoding Layer (batch_size, seq_len, d_model) = (# Samples, n, d_model)


class PositionalEncoding(nn.Module):
    """
    The Positional Encoding embeds some temporal information about the current index in the sequence.
    The paper uses the following equations in order to map the indices to the weights that we'd need
    to add to any embedding at position pos in the sequence (this comes from dim = 1 in the input to this
    layer), and at dimension i (this comes from dim = 2):

    PE(pos, 2i) = sin(pos * (10000^2i) / d_model)
    PE(pos,2i+1) = cos(pos * (10000^2i) / d_model)

    We then take the positional encodings that we will build (in a vectorized fashion) and add them to the
    input to this layer. Finally, once we do this addition, we will apply dropout to the
    """

    def __init__(self, dropout_probability: float, d_model: int, max_sequence_length: int):
        super(PositionalEncoding, self).__init__()

        # A dropout layer to apply after doing embeddings + positional encoding
        self.dropout_layer = nn.Dropout(p=dropout_probability)

        # Create a mask to add to our embeddings
        pe_mask = PositionalEncoding.create_mask(d_model, max_sequence_length)
        self.register_buffer("positional_encoding_mask", pe_mask)

    @staticmethod
    def create_mask(d_model: int, max_sequence_length: int) -> torch.Tensor:

        # The size of this mask will be identical to the dimensions of the input to this module,
        # ignoring the batch size, since we can just broadcast this operation on each batch.

        mask = torch.zeros(max_sequence_length, d_model)  # We build a mask of n x d_model
        positions = torch.arange(max_sequence_length) \
            .reshape(-1, 1)  # Positions are going to be fixed per-row
        div_terms = torch.exp(
            torch.arange(0, d_model, 2) * (-1 / d_model) * np.log(10000))  # Get the divisors (that will repeat)
        sinusoidal_frequency = positions * div_terms  # Frequency for each even/odd pair is the same

        mask[:, 0::2] = torch.sin(sinusoidal_frequency)  # Set the even ones to the sine
        mask[:, 1::2] = torch.cos(sinusoidal_frequency)  # Set the odd ones to the cosine

        return mask

    def forward(self, x):
        return self.dropout_layer(x + self.positional_encoding_mask)
