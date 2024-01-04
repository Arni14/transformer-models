import torch
from torch import nn

# Input: Output of Attention / FFN Layer (batch_size, seq_len, d_model) = (# Samples, n, d_model)
# Output: Output of Layer Norm (batch_size, seq_len, d_model) = (# Samples, n, d_model)


class LayerNorm(nn.Module):
    """
    The LayerNorm is much like BatchNorm, but instead will normalize things by the model dimension.
    What this means is for a given dimension of the latent space, we'll normalize across both the set of sequences
    and all batches.
    """

    def __init__(self, num_features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()

        # Create our learnable parameters gamma and beta
        self.gamma_vector = nn.Parameter(torch.ones(num_features))
        self.beta_vector = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):

        mean_vector = torch.mean(x, dim=-1, keepdim=True)
        std_vector = torch.std(x, dim=-1, keepdim=True)
        normalized_input = (x - mean_vector) / (std_vector + self.eps)

        return self.gamma_vector * normalized_input + self.beta_vector
