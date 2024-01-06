import torch
from torch import nn
import numpy as np

from transformer_models.transformer_blocks.sub_layer import SubLayer

# Input: Output of Previous Layer (batch_size, seq_len, d_model) = (# Samples, n, d_model)
# Output: Output of Single Head Attention (batch_size, seq_len, d_model) = (# Samples, n, d_model)


class AttentionUtilities:

    @staticmethod
    def attention_operation(
            query: torch.Tensor, value: torch.Tensor, key: torch.Tensor, mask: bool) -> torch.Tensor:

        # We'll need this for our attention computation
        key_dimension = key.shape[-1]

        # Compute the attention scoring based on the queries and keys
        scores = (1 / np.sqrt(key_dimension)) * torch.matmul(
            query, torch.transpose(key, -1, -2))

        # If we are masking, then we can simply create a mask of the upper triangular matrix, and set it to -BIG
        if mask:

            upper_triangular_mask: torch.Tensor = torch.triu(torch.ones_like(scores), diagonal=1) > 0
            scores = torch.masked_fill(scores, upper_triangular_mask, -1e9)

        # Compute the softmax scores for the attention scores
        softmax_scores = torch.softmax(scores, dim=-2)

        # Attention scores are the value vectors weighted by the softmaxed scores
        attention = torch.matmul(softmax_scores, value)

        return attention


class SingleHeadAttention(nn.Module):
    """
    As an exercise to better understand Attention mechanisms, we first implement the
    single-head attention mechanism. Here, we have three simple matrices which we will
    create based on a linear mapping.

    Q: Input Tensor with a Linear Mapping Applied
    K: Input Tensor with a Linear Mapping Applied
    V: Input Tensor with a Linear Mapping Applied

    Attention = (QK^T) / sqrt(d_k)
    Output = MatMul(Softmax(Attention), Values)
    """

    def __init__(
            self, d_model: int, query_dimension: int, key_dimension: int, value_dimension: int,
            masked_attention: bool = False, dropout: float = 0.1):
        super(SingleHeadAttention, self).__init__()

        # Is it masked attention?
        self.masked_attention = masked_attention

        # Store the key dimension for the attention computation
        self.key_dimension = key_dimension

        # A linear mapping from query to Q
        self.query_mapper = nn.Linear(d_model, query_dimension)

        # A linear mapping from query to K
        self.key_mapper = nn.Linear(d_model, key_dimension)

        # A linear mapping from query to V
        self.value_mapper = nn.Linear(d_model, value_dimension)

        # Dropout right at the end
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        # The keys and values are nothing but a linear mapping of the input embeddings

        query = self.query_mapper(x)
        key = self.key_mapper(x)
        value = self.value_mapper(x)

        # We compute scores, softmaxed probabilities, and then value-weighted scores (attention)
        # for a single head.

        attention = AttentionUtilities.attention_operation(query, value, key, self.masked_attention)

        # Run dropout on the Attention

        return self.dropout(attention)


class MultiHeadAttention(SubLayer):
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

    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs):
        super(MultiHeadAttention, self).__init__(d_model, dropout, **kwargs)

        # Grab the required configs from kwargs
        num_heads = kwargs['num_heads']
        key_dimension = kwargs['key_dimension']
        value_dimension = kwargs['value_dimension']
        masked = kwargs['masked']

        # Is it masked attention?
        self.masked = masked

        # Store the projected key dimension for the attention computation, which is reduced for more than 1 head
        self.projected_key_dimension = d_model // num_heads  # d_k,proj
        self.projected_value_dimension = d_model // num_heads  # d_v,proj
        self.key_dimension = key_dimension  # d_k = d_model (in the paper)
        self.value_dimension = value_dimension  # d_v = d_model (in the paper)

        # Store the number of heads
        self.num_heads = num_heads

        # A linear mapping from query to Q
        self.query_mapper = nn.Linear(d_model, self.key_dimension)  # (d_model -> d_k)

        # All linear mappings for Q to Q_{i} for i between 0 and num_heads - 1 (d_k -> d_k,proj)
        self.projected_query_mappers = nn.ModuleList(
            [nn.Linear(self.key_dimension, self.projected_key_dimension) for _ in range(num_heads)])

        # A linear mapping from query to K
        self.key_mapper = nn.Linear(d_model, self.key_dimension)  # (d_model -> d_k)

        # All linear mappings for K to K_{i} for i between 0 and num_heads - 1 (d_k->d_k,proj)
        self.projected_key_mappers = nn.ModuleList(
            [nn.Linear(self.key_dimension, self.projected_key_dimension) for _ in range(num_heads)])

        # A linear mapping from query to V
        self.value_mapper = nn.Linear(d_model, self.value_dimension)

        # All linear mappings for V to V_{i} for i between 0 and num_heads - 1 (d_v->d_v,proj)
        self.projected_value_mappers = nn.ModuleList(
            [nn.Linear(self.value_dimension, self.projected_value_dimension) for _ in range(num_heads)]
        )

        # Single projection layer from multihead concat to d_model output
        self.concat_projection_layer = nn.Linear(num_heads * self.projected_value_dimension, d_model)

        # After the projection, we'll optionally run dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, **kwargs):

        encoding = kwargs.get('encoding')

        # In any case, we will map the input to a query matrix

        Q = self.query_mapper(x)

        # For K and V, if it is cross attention, we will use an inputted key and an inputted value

        K = self.key_mapper(x) if encoding is None else self.key_mapper(encoding)
        V = self.value_mapper(x) if encoding is None else self.value_mapper(encoding)

        # Once we have computed these, we need to project each one of them down

        multi_head_concat = torch.concat([
            AttentionUtilities.attention_operation(
                self.projected_query_mappers[i](Q),
                self.projected_key_mappers[i](K),
                self.projected_value_mappers[i](V), self.masked)
            for i in range(self.num_heads)
        ], dim=2)

        # Linear projection of concatenated attention vectors into the output

        return self.dropout(self.concat_projection_layer(multi_head_concat))
