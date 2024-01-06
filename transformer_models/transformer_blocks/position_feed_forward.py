import torch
from torch import nn


# Input: Output of Previous Layer (batch_size, seq_len, d_model) = (# Samples, n, d_model)
# Output: Output of Single Head Attention (batch_size, seq_len, d_model) = (# Samples, n, d_model)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model: int, hidden_dimension: int, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()

        # Mapping from the model dimension to the hidden dimension
        self.first_proj = nn.Linear(d_model, hidden_dimension)

        # Mapping from the hidden dimension back to model dimension
        self.second_proj = nn.Linear(hidden_dimension, d_model)

        # Dropout at the output
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        # With an activation at the hidden layer, map two linear transformations
        projections = self.second_proj(torch.relu(self.first_proj(x)))
        return self.dropout(projections)
