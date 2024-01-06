from torch import nn

from transformer_models.transformer_blocks.sub_layer import ResidualDropoutSubLayer
from transformer_models.transformer_blocks.attention import MultiHeadAttention
from transformer_models.transformer_blocks.position_feed_forward import PositionWiseFeedForward

from transformer_models.utils import create_layer_clones


# Input: Output of PE or Previous Decoder Layer (batch_size, seq_len, d_model) = (# Samples, n, d_model)
# Output: Output of an entire Decoder Stack


class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, hidden_dimension: int):
        super(DecoderBlock, self).__init__()

        # The first "Sub Layer" is a Multi Head Attention Block paired with a LayerNorm and residual connection
        attention_sub_layer = MultiHeadAttention(
            d_model, dropout=0.1, num_heads=num_heads, key_dimension=d_model, value_dimension=d_model, masked=True)
        self.masked_attention_block = ResidualDropoutSubLayer(d_model, attention_sub_layer)

        # The second "Sub Layer" is a Cross-Attention block paired with a LayerNorm and residual connection
        cross_attention_sub_layer = MultiHeadAttention(
            d_model, dropout=0.1, num_heads=num_heads, key_dimension=d_model, value_dimension=d_model, masked=False)
        self.cross_attention_block = ResidualDropoutSubLayer(d_model, cross_attention_sub_layer)

        # The second "Sub Layer" is a Position Wise Feed Forward Block paired with LayerNorm and residual connection
        feed_forward_sub_layer = PositionWiseFeedForward(d_model, dropout=0.1, hidden_dimension=hidden_dimension)
        self.feed_forward_block = ResidualDropoutSubLayer(d_model, feed_forward_sub_layer)

    def forward(self, x, encoding=None):
        return self.feed_forward_block(self.cross_attention_block(self.masked_attention_block(x), encoding=encoding))


class Decoder(nn.Module):

    def __init__(self, d_model: int, num_heads: int, hidden_dimension: int, num_copies: int):
        super(Decoder, self).__init__()

        # Simply create N copies of the encoder block
        self.decoder_blocks = create_layer_clones(
            DecoderBlock(d_model, num_heads, hidden_dimension), num_copies)

    def forward(self, x, encoding=None):

        # Run the decoder block N times for N copies, each time using the same encoding
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoding=encoding)
        return x
