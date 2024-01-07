import torch
import sys
import argparse

from transformer_models.transformer_blocks.positional_encoding import PositionalEncoding
from transformer_models.transformer_blocks.layer_norm import LayerNorm

from transformer_models.transformer_blocks.attention import SingleHeadAttention
from transformer_models.transformer_blocks.attention import MultiHeadAttention

from transformer_models.transformer_blocks.encoder import EncoderBlock
from transformer_models.transformer_blocks.encoder import Encoder

from transformer_models.transformer_blocks.decoder import Decoder

from transformer_models.transformer_blocks.transformer_model import TransformerModel


def parse_arguments(args):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--device', help='Device on which to run computation', default="cpu",
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
            'Encoder',
            'EncoderBlock',
            'Transformer'))

    parser.add_argument(
        '--tensorboard', help="Write Tensor Event to torchlogs", action='store_true')

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


def get_encoding(batch_size: int, seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:

    # Input data dimensions: (batch_size, seq_len, d_model)
    encoding = torch \
        .ones((batch_size, seq_len, d_model)) \
        .to(device)

    return encoding


def main(args):

    arguments = parse_arguments(args)

    device = torch.device(arguments['device'])
    block = arguments['block']
    write_tensorboard = arguments['tensorboard']

    batch_size = 1
    seq_len = 2
    d_model = 4

    dataset = get_dataset(batch_size, seq_len, d_model, device)

    print(f"Dataset Size: {dataset.shape}")
    print(f"Dataset Device: {dataset.device}")
    print(f"Initial Tensor: {dataset}\n")

    if block == "PositionalEncoding":

        # Testing out the Positional Encoding

        print("Testing out Positional Embeddings")
        model = PositionalEncoding(0, d_model, 100).to(device)

        print(f"Position Embedding Mask: {model.positional_encoding_mask}")

        positional_context_embedding = model(dataset)
        print(f"After Positional Embedding: {positional_context_embedding}")

    # Testing out Layer Normalization

    elif block == "LayerNorm":

        # Testing out LayerNorm

        print("Testing out Layer Normalization")
        model = LayerNorm(d_model).to(device)

        print(f"Gamma Vector: {model.gamma_vector}")
        print(f"Beta Vector: {model.beta_vector}")

        layer_norm_output = model(dataset)
        print(f"After Layer Normalization: {layer_norm_output}")

    elif block == "SingleHeadAttention":

        # Testing out SingleHeadAttention

        print("Testing out Single Head Attention")
        model = SingleHeadAttention(d_model, d_model, d_model, d_model).to(device)

        model.eval()
        print(f"Query Matrix: {model.query_mapper(dataset)}")
        print(f"Key Matrix: {model.key_mapper(dataset)}")
        print(f"Value Matrix: {model.value_mapper(dataset)}")

        attention_output = model(dataset)
        print(f"After Single Head Attention: {attention_output}")

    elif block == "MultiHeadAttention":

        # Testing out MultiHeadAttention

        print("Testing out Multi Head Attention")
        model = MultiHeadAttention(
            d_model, 0, num_heads=8, key_dimension=d_model, value_dimension=d_model, masked=False).to(device)

        model.eval()
        print(f"Query Matrix: {model.query_mapper(dataset)}")
        print(f"Key Matrix: {model.key_mapper(dataset)}")
        print(f"Value Matrix: {model.value_mapper(dataset)}")

        attention_output = model(dataset)
        print(f"After Multi Head Attention: {attention_output}")
        print(f"Shape of Multi Head Attention Output: {attention_output.shape}")

    elif block == "MaskedMultiHeadAttention":

        # Testing out MultiHeadAttention

        print("Testing out Multi Head Attention")
        model = MultiHeadAttention(
            d_model, 0, num_heads=8, key_dimension=d_model, value_dimension=d_model, masked=True).to(device)

        model.eval()
        attention_output = model(dataset)

        print(f"After Multi Head Attention: {attention_output}")
        print(f"Shape of Multi Head Attention Output: {attention_output.shape}")

    elif block == "EncoderBlock":

        # Testing out EncoderBlock

        print("Testing out Encoder Block")
        model = EncoderBlock(d_model, 4, 64).to(device)

        model.eval()
        encoder_output = model(dataset)

        print(f"After Encoder Block: {encoder_output}")
        print(f"Shape of Encoder Output: {encoder_output.shape}")

    elif block == "Encoder":

        # Testing out Encoder

        print("Testing out Encoder")
        model = Encoder(d_model, 4, 64, 6).to(device)

        model.eval()
        encoder_output = model(dataset)

        print(f"After Encoder Block: {encoder_output}")
        print(f"Shape of Encoder Output: {encoder_output.shape}")

    elif block == "Decoder":

        # Testing out Decoder

        print("Testing out Decoder")
        model = Decoder(d_model, 2, 8, 2).to(device)

        # Let's create a dummy encoding

        encoding = get_encoding(batch_size, seq_len, d_model, device)

        model.eval()
        decoder_output = model(dataset, encoding=encoding)

        print(f"After Encoder Block: {decoder_output}")
        print(f"Shape of Encoder Output: {decoder_output.shape}")

    elif block == "Transformer":

        # Testing out Transformer

        transformer_input = torch.tensor([[1, 5]]).to(device)
        transformer_output = torch.tensor([[3, 8, 7]]).to(device)

        print("Testing out End-to-End Transformer")
        model = TransformerModel(
            input_vocab_size=10, output_vocab_size=10, d_model=4, num_attention_heads=2,
            feed_forward_hidden_dimension=8, num_encoder_stacks=2, num_decoder_stacks=2).to(device)

        # Let's create a dummy encoding

        model.eval()
        model_output = model(transformer_input, transformer_output)

        print(f"After Transformer: {model_output}")
        print(f"After Transformer Shape: {model_output.shape}")

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter("../torchlogs/")
        writer.add_graph(model, [transformer_input, transformer_output])
        writer.close()

    else:

        model = None
        print(f"Block {block} is not a block that has yet been implemented.")

    # if model is not None and write_tensorboard:
    #
    #     from torch.utils.tensorboard import SummaryWriter
    #
    #     writer = SummaryWriter("../torchlogs/")
    #     writer.add_graph(model, dataset,)
    #     writer.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
