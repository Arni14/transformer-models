import torch
from torch import nn
import numpy as np

# Input: Output of Previous Layer (batch_size, seq_len, d_model) = (# Samples, n, d_model)
# Output: Output of Single Head Attention (batch_size, seq_len, d_model) = (# Samples, n, d_model)


class SingleHeadAttention(nn.Module):
    """
    As an exercise to better understand Attention mechanisms, we first implement the
    single-head attention mechanism. Here, we have three simple matrices which we will
    create based on a linear mapping.

    Q: Input Tensor
    K: Input Tensor with a Linear Mapping Applied
    V: Input Tensor with a Linear Mapping Applied

    Attention = (QK^T) / sqrt(d_k)
    Output = MatMul(Softmax(Attention), Values)
    """

    def __init__(self, d_model: int, query_dimension: int, key_dimension: int, value_dimension: int):
        super(SingleHeadAttention, self).__init__()

        # Store the key dimension for the attention computation
        self.key_dimension = key_dimension

        # A linear mapping from query to Q
        self.query_mapper = nn.Linear(d_model, query_dimension)

        # A linear mapping from query to K
        self.key_mapper = nn.Linear(d_model, key_dimension)

        # A linear mapping from query to V
        self.value_mapper = nn.Linear(d_model, value_dimension)

    def forward(self, x):

        # The keys and values are nothing but a linear mapping of the input embeddings

        query = self.query_mapper(x)
        key = self.key_mapper(x)
        value = self.value_mapper(x)

        # We compute scores, softmaxed probabilities, and then value-weighted scores (attention)
        # for a single head.

        scores = (1 / np.sqrt(self.key_dimension)) * torch.matmul(
            query, torch.transpose(key, -1, -2))
        softmax_scores = torch.softmax(scores, dim=-1)
        attention = torch.matmul(softmax_scores, value)

        return attention
