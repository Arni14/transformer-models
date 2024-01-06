from torch import nn

from transformer_models.transformer_blocks.sub_layer import ResidualDropoutSubLayer
from transformer_models.transformer_blocks.attention import MultiHeadAttention
from transformer_models.transformer_blocks.position_feed_forward import PositionWiseFeedForward

from transformer_models.utils import create_layer_clones


# Input: Output of PE or Previous Encoder Layer (batch_size, seq_len, d_model) = (# Samples, n, d_model)
# Output: Output of an entire Encoder Stack


class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, hidden_dimension: int):
        super(EncoderBlock, self).__init__()

        # The first "Sub Layer" is a Multi Head Attention Block paired with a LayerNorm and residual connection
        attention_sub_layer = MultiHeadAttention(
            d_model, dropout=0.1, num_heads=num_heads, key_dimension=d_model, value_dimension=d_model, masked=False)
        self.attention_block = ResidualDropoutSubLayer(d_model, attention_sub_layer)

        # The second "Sub Layer" is a Position Wise Feed Forward Block paired with LayerNorm and residual connection
        feed_forward_sub_layer = PositionWiseFeedForward(d_model, dropout=0.1, hidden_dimension=hidden_dimension)
        self.feed_forward_block = ResidualDropoutSubLayer(d_model, feed_forward_sub_layer)

    def forward(self, x):
        return self.feed_forward_block(
            self.attention_block(x))


class Encoder(nn.Module):

    def __init__(self, d_model: int, num_heads: int, hidden_dimension: int, num_copies: int):
        super(Encoder, self).__init__()

        # Simply create N copies of the encoder block
        self.encoder_blocks = create_layer_clones(
            EncoderBlock(d_model, num_heads, hidden_dimension), num_copies)

    def forward(self, x):

        # Run the encoder block N times for N copies.
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x
