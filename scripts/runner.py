import torch
import sys
import argparse


def parse_arguments(args):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--device', required=True, help='Device on which to run computation', default="cpu",
        choices=('cpu', 'cuda', 'mps'))

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

    batch_size = 32
    seq_len = 50
    d_model = 512

    dataset = get_dataset(batch_size, seq_len, d_model, device)

    print(f"Dataset Size: {dataset.shape}")
    print(f"Dataset Device: {dataset.device}")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]) or 0)
