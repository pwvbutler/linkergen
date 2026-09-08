"""GPT-style autoregressive Transformer decoder for SMILES generation.

Defines the core decoder stack (TransformerDecoderLayer, TransformerDecoder),
the two conditioning-memory encoders (PropertyEncoder, TextMemoryEncoder), the
AttentionPoolingLayer used to pool a variable-length text encoding down to a
fixed number of memory slots, and LinkerGen, which wraps a conditioner and the
decoder behind a single condition_type flag ("none" | "property" | "text"). In
text mode, the (hidden_states, mask) pair from an external nach0 encoder is
first pooled by AttentionPoolingLayer to n_pool_queries vectors before
TextMemoryEncoder projects them into cross-attention memory, so the decoder
itself never sees a text padding mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention


class PropertyEncoder(nn.Module):
    """Maps a numeric property vector to a fixed number of memory slots for cross-attention."""

    def __init__(self, property_dim, d_model, n_memory, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(property_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.slots = nn.Parameter(torch.randn(n_memory, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, prop_vec):
        h = self.mlp(prop_vec)                       # (B, d_model)
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)  # (B, n_memory, d_model)
        return self.norm(h)


class TextMemoryEncoder(nn.Module):
    """
    Projects nach0 encoder hidden states of shape (B, S, nach0_dim) to
    (B, S, d_model). In LinkerGen's text mode, the input is not the raw
    per-token nach0 sequence but the fixed-length output of AttentionPoolingLayer,
    so S here is n_pool_queries rather than the description length.
    """

    def __init__(self, nach0_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(nach0_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, S, nach0_dim), projected to (B, S, d_model)
        return self.dropout(self.norm(self.proj(hidden_states)))


class AttentionPoolingLayer(nn.Module):
    """
    Pools a variable-length sequence down to a fixed number of vectors via
    learned-query cross-attention: (B, S, d_model) -> (B, n_queries, d_model).
    """

    def __init__(
        self,
        d_model=128,
        n_heads=8,
        dropout=0.1,
        n_queries=1,
        learn_mean_correction=False,
    ):
        super().__init__()
        self.query = nn.Parameter(
            torch.randn(n_queries, d_model) * d_model**-0.5
        )
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.learn_mean_correction = learn_mean_correction
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def masked_mean(x, mask=None):
        if mask is None:
            return x.mean(dim=1, keepdim=True)

        weights = mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1, keepdim=True) / weights.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)

    def forward(self, x, memory_mask=None):
        batch_size = x.size(0)
        queries = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        kv = self.norm_kv(x)

        if memory_mask is not None:
            memory_mask = memory_mask.bool()
            empty = ~memory_mask.any(dim=1)
            if empty.any():
                memory_mask = memory_mask.clone()
                memory_mask[empty, 0] = True

        pooled = self.attention(
            query=queries,
            key=kv,
            value=kv,
            is_causal=False,
            key_padding_mask=memory_mask
        )
        pooled = self.dropout(pooled)

        if self.learn_mean_correction:
            pooled = pooled + self.masked_mean(kv, memory_mask)

        return self.norm_out(pooled)  # (B, n_queries, d_model)


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=128,
        d_hidden=512,
        n_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.cross_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory=None, memory_mask=None):

        # self-attention on input
        attn_out = self.self_attention(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            is_causal=True,
        )
        x = x + self.dropout1(attn_out)

        # cross-attention on memory (skipped for unconditional generation)
        if memory is not None:
            cross_attn_out = self.cross_attention(
                query=self.norm2(x),
                key=memory,
                value=memory,
                is_causal=False,
                key_padding_mask=memory_mask,
            )
            x = x + self.dropout2(cross_attn_out)

        # feed forward neural network
        ffn_out = self.ffn(self.norm3(x))
        x = x + self.dropout3(ffn_out)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        d_hidden,
        n_heads,
        n_layers,
        max_seq_len=500,
        dropout=0.1,
    ):
        super().__init__()

        # vector for each token
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.emb_dropout = nn.Dropout(dropout)

        # fixed sinusoidal positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # compute the denominator term using log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # assign sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # add batch dimension: (1, max_seq_len, d_model)

        # register_buffer ensures pe is saved with the model state_dict but not trained
        self.register_buffer('pe', pe)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, d_hidden, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, memory=None, memory_mask=None):

        _, seq_len = x.size()

        # embed tokens and add position encoding
        x = self.token_embedding(x) + self.pe[:, :seq_len]
        x = self.emb_dropout(x)

        # pass through all decoder layers (causal masking happens inside self-attention)
        for layer in self.layers:
            x = layer(x, memory, memory_mask)

        logits = self.fc_out(self.norm_out(x))

        return logits


class LinkerGen(nn.Module):
    """
    GPT-style autoregressive SMILES decoder with three interchangeable conditioning
    modes, selected by `condition_type`:
      - "none":     unconditional generation, no cross-attention memory.
      - "property": `conditioner` is a PropertyEncoder; `condition` is a (B, property_dim)
                    tensor of (already-standardized) numeric properties.
      - "text":     `conditioner` is a TextMemoryEncoder; `condition` is a precomputed
                    (hidden_states, mask) pair produced upstream by an external
                    Nach0Encoder. This class never imports transformers or runs
                    nach0 itself. The mask is consumed by an AttentionPoolingLayer
                    that pools hidden_states down to n_pool_queries vectors before
                    they reach TextMemoryEncoder, so no mask is passed to the decoder.
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        d_hidden=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=512,
        condition_type="none",
        property_dim=None,
        n_memory=8,
        text_hidden_size=None,
        n_pool_queries=1,
        dropout=0.05,
    ):
        super().__init__()

        if condition_type not in ("none", "property", "text"):
            raise ValueError(f"condition_type must be one of 'none'/'property'/'text', got {condition_type!r}")
        self.condition_type = condition_type

        if condition_type == "property":
            if property_dim is None:
                raise ValueError("condition_type='property' requires property_dim")
            self.conditioner = PropertyEncoder(property_dim=property_dim, d_model=d_model, n_memory=n_memory)
        elif condition_type == "text":
            if text_hidden_size is None:
                raise ValueError("condition_type='text' requires text_hidden_size")
            # Pools the variable-length nach0 sequence down to n_pool_queries vectors
            # before TextMemoryEncoder projects them into cross-attention memory.
            self.pooling = AttentionPoolingLayer(
                d_model=text_hidden_size,
                n_heads=n_heads,
                dropout=dropout,
                n_queries=n_pool_queries,
                learn_mean_correction=True,
            )
            self.conditioner = TextMemoryEncoder(nach0_dim=text_hidden_size, d_model=d_model, dropout=dropout)
        else:
            self.conditioner = None

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_hidden=d_hidden,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward(self, x, condition=None):
        memory, memory_mask = None, None

        if self.condition_type == "property" and condition is not None:
            memory = self.conditioner(condition)  # condition: (B, property_dim)
        elif self.condition_type == "text" and condition is not None:
            hidden_states, memory_mask = condition  # precomputed upstream by Nach0Encoder
            memory_mask = memory_mask.bool()  # Hugging Face gives an int mask; pooling needs bool for ~mask
            pooled = self.pooling(hidden_states, memory_mask=memory_mask)  # (B, n_pool_queries, text_hidden_size)
            memory = self.conditioner(pooled)  # (B, n_pool_queries, d_model)
            memory_mask = None  # pooled memory has no padding: every query slot is real

        logits = self.decoder(x, memory, memory_mask)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        condition=None,
        max_new_tokens=200,
        temperature=0.8,
        eos_token_id=None,
        pad_token_id=None,
    ):
        # `condition` is computed once by the caller and reused unchanged on every
        # step below, so text mode never re-runs nach0 during decoding.
        generated = input_ids.clone()
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)

        for _ in range(max_new_tokens):
            logits = self.forward(generated, condition)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if temperature > 0:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            if eos_token_id is not None:
                # once finished, keep emitting EOS or PAD, not random extra tokens
                freeze_id = eos_token_id if pad_token_id is None else pad_token_id
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, freeze_id),
                    next_token,
                )

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None:
                finished |= next_token.squeeze(1) == eos_token_id
                if finished.all():
                    break

        return generated
