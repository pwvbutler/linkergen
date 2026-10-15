"""Multi-head attention wrapping scaled dot-product attention, used for both self- and cross-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_k = d_model // n_heads

        self.query_weights = nn.Linear(d_model, d_model)
        self.key_weights = nn.Linear(d_model, d_model)
        self.value_weights = nn.Linear(d_model, d_model)
        self.output_weights = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, query, key, value, is_causal=False, key_padding_mask=None):
        batch_size = query.size(0)

        # Compute Q, K, and V matrices (seq_len x d_model)
        Q = self.query_weights(query)
        K = self.key_weights(key)
        V = self.value_weights(value)

        # Split into heads: (batch_size, seq_len, n_heads, d_k). Not more
        # efficient than one large matrix multiply, but lets each head learn
        # over a different subspace of d_model (the token features).
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k)

        # matrix multiplication works on the last two dimensions, hence swap n_heads and seq_len
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Convert the (B, S) bool mask to a (B, 1, 1, S) additive mask for sdpa
        additive_mask = None
        if key_padding_mask is not None:
            # real token positions get 0.0, padded positions get -inf
            additive_mask = torch.zeros_like(key_padding_mask, dtype=query.dtype)
            additive_mask = additive_mask.masked_fill(~key_padding_mask, float("-inf"))
            additive_mask = additive_mask[:, None, None, :]  # broadcast over heads/queries

        # scaled dot-product attention
        # (batch_size, n_heads, seq_len, d_k) @ (batch_size, n_heads, d_k, seq_len)
        # gives (batch_size, n_heads, seq_len, seq_len)
        context = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            is_causal=is_causal,
            attn_mask=additive_mask,  # padding mask
            dropout_p=self.dropout if self.training else 0.0,
        )

        # merge heads: (batch_size, seq_len, n_heads, d_k)
        context = context.transpose(1, 2).contiguous()

        context = context.view(
            batch_size,
            -1,
            self.d_model,
        )

        # output: (batch_size, seq_len, d_model)
        return self.output_weights(context)
