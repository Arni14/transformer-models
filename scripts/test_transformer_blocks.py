import torch
import sys
import argparse

from transformer_models.transformer_blocks.positional_encoding import PositionalEncoding
from transformer_models.transformer_blocks.layer_norm import LayerNorm

from transformer_models.transformer_blocks.attention import SingleHeadAttention
from transformer_models.transformer_blocks.attention import MultiHeadAttention

from transformer_models.transformer_blocks.attention import AttentionUtilities

from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


def parse_arguments(args):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--device', required=True, help='Device on which to run computation', default="cpu",
        choices=(
            'cpu',
            'cuda',
            'mps'))

    parser.add_argument(
        '--block', required=True, help="Which transformer block would you like to test?",
        choices=(
            'PositionalEncoding',
            'LayerNorm',
            'SingleHeadAttention',
            'MultiHeadAttention',
            'MaskedMultiHeadAttention',
            'Decoder',
            'Encoder'))

    return vars(parser.parse_args(args))


# A transformer model will be used for sequence to sequence modeling.
# Typically, the data dimensionality for any input is as follows:
# (batch_size, seq_len, d_model)

# In order to get a concrete understanding of the model runtime, we'll not deal
# with real data for now. As we update this model and build the pieces, we'll
# test and validate different components. Once that's complete, we'll actually
# train this on some GPUs.


def get_dataset(batch_size: int, seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:

    # Input data dimensions: (batch_size, seq_len, d_model)
    input_data = torch \
        .randn((batch_size, seq_len, d_model)) \
        .to(device)

    return input_data


def main(args):

    arguments = parse_arguments(args)

    device = torch.device(arguments['device'])
    block = arguments['block']

    batch_size = 1
    seq_len = 10
    d_model = 512

    dataset = get_dataset(batch_size, seq_len, d_model, device)

    print(f"Dataset Size: {dataset.shape}")
    print(f"Dataset Device: {dataset.device}")
    print(f"Initial Tensor: {dataset}\n")

    if block == "PositionalEncoding":

        # Testing out the Positional Encoding

        print("Testing out Positional Embeddings")
        positional_encoder = PositionalEncoding(0, d_model, seq_len).to(device)

        print(f"Position Embedding Mask: {positional_encoder.positional_encoding_mask}")

        positional_context_embedding = positional_encoder(dataset)
        print(f"After Positional Embedding: {positional_context_embedding}")

    # Testing out Layer Normalization

    elif block == "LayerNorm":

        # Testing out LayerNorm

        print("Testing out Layer Normalization")
        layer_norm = LayerNorm(d_model).to(device)

        print(f"Gamma Vector: {layer_norm.gamma_vector}")
        print(f"Beta Vector: {layer_norm.beta_vector}")

        layer_norm_output = layer_norm(dataset)
        print(f"After Layer Normalization: {layer_norm_output}")

    elif block == "SingleHeadAttention":

        # Testing out SingleHeadAttention

        print("Testing out Single Head Attention")
        single_head_attention = SingleHeadAttention(d_model, d_model, d_model, d_model).to(device)

        single_head_attention.eval()
        print(f"Query Matrix: {single_head_attention.query_mapper(dataset)}")
        print(f"Key Matrix: {single_head_attention.key_mapper(dataset)}")
        print(f"Value Matrix: {single_head_attention.value_mapper(dataset)}")

        attention_output = single_head_attention(dataset)
        print(f"After Single Head Attention: {attention_output}")

    elif block == "MultiHeadAttention":

        # Testing out MultiHeadAttention

        print("Testing out Multi Head Attention")
        multi_head_attention = MultiHeadAttention(d_model, 8, d_model, d_model, dropout=0).to(device)

        multi_head_attention.eval()
        print(f"Query Matrix: {multi_head_attention.query_mapper(dataset)}")
        print(f"Key Matrix: {multi_head_attention.key_mapper(dataset)}")
        print(f"Value Matrix: {multi_head_attention.value_mapper(dataset)}")

        attention_output = multi_head_attention(dataset)
        print(f"After Multi Head Attention: {attention_output}")
        print(f"Shape of Multi Head Attention Output: {attention_output.shape}")

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter("../torchlogs/")
        writer.add_graph(multi_head_attention, dataset)
        writer.close()

    elif block == "MaskedMultiHeadAttention":

        # Testing out MultiHeadAttention

        print("Testing out Multi Head Attention")
        multi_head_attention = MultiHeadAttention(d_model, 8, d_model, d_model, dropout=0).to(device)

        multi_head_attention.eval()
        attention_output = multi_head_attention(dataset, mask=True)

        print(f"After Multi Head Attention: {attention_output}")
        print(f"Shape of Multi Head Attention Output: {attention_output.shape}")

    else:

        print(f"Block {block} is not a block that has yet been implemented.")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
