import torch
import numpy as np
from torch import nn

from transformer_models.transformer_blocks.encoder import Encoder
from transformer_models.transformer_blocks.decoder import Decoder

from transformer_models.transformer_blocks.positional_encoding import PositionalEncoding


class TransformerModel(nn.Module):

    def __init__(
            self, input_vocab_size: int, output_vocab_size: int, d_model: int, num_attention_heads: int,
            feed_forward_hidden_dimension: int, num_encoder_stacks: int, num_decoder_stacks: int, dropout: float = 0.1,
            decode_policy: str = "Greedy"):
        super(TransformerModel, self).__init__()

        # Parameters that might be useful
        self.d_model = d_model
        self.decode_policy = decode_policy

        # Generic Positional Encoder
        self.positional_encoder = PositionalEncoding(dropout_probability=dropout, d_model=d_model)

        # Encoder components
        self.input_embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        self.encoder = Encoder(
            d_model=d_model, num_heads=num_attention_heads, hidden_dimension=feed_forward_hidden_dimension,
            num_copies=num_encoder_stacks)

        # Decoder components
        self.output_embedding = nn.Embedding(num_embeddings=output_vocab_size, embedding_dim=d_model)
        self.decoder = Decoder(
            d_model=d_model, num_heads=num_attention_heads, hidden_dimension=feed_forward_hidden_dimension,
            num_copies=num_decoder_stacks)
        self.vocab_projection = nn.Linear(d_model, output_vocab_size)

    def encode(self, x):
        input_embedding = self.input_embedding(x) * np.sqrt(self.d_model)
        positionally_encoded_embedding = self.positional_encoder(input_embedding)
        return self.encoder(positionally_encoded_embedding)

    def decode(self, y, encoding):
        output_embedding = self.positional_encoder(self.output_embedding(y) * np.sqrt(self.d_model))
        decoded_output = self.decoder(output_embedding, encoding=encoding)
        return self.vocab_projection(decoded_output)

    def token_selector(self, probabilities, decode_policy):
        if decode_policy == "Greedy":
            return torch.argmax(probabilities, dim=-1, keepdim=True)
        else:
            return probabilities

    def forward(self, input_sequence, output_sequence, output_probabilities=False):

        # First, we will take our input sequence (of the form (batch_size, seq_len)) and encode it
        encoding = self.encode(input_sequence)

        # Now, given the encoding, we will take our output sequence (also of the form (batch_size, seq_len)) and
        # use it, paired with our input sequence encoding, to generate the next token.
        decoding = self.decode(output_sequence, encoding)

        # Now, let's predict the most likely token using the softmax
        probabilities = torch.softmax(decoding, dim=-2)

        if output_probabilities:
            return probabilities

        # Eventually, we will also use this function to actually make a choice (so an argmax policy, beam search, etc.)
        return self.token_selector(probabilities, self.decode_policy)
