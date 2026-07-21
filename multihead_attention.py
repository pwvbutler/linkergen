import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F




class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.query_weights = nn.Linear(d_model, d_model)
        self.key_weights = nn.Linear(d_model, d_model)
        self.value_weights = nn.Linear(d_model, d_model)
        self.output_weights = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, query, key, value, is_causal=True):
        # batch_size, seq_len, d_model = x.size()
        batch_size = query.size(0)

        # calc Q, K, and V matrices (seq_len x d_model)
        Q = self.query_weights(query)
        K = self.key_weights(key)
        V = self.value_weights(value)


        # split into heads (batch_size, seq_len, n_heads, d_k)
        # while not more efficient than using the large matrices
        # it is better for learning multiple contexts of each token
        # we are splitting d_model (i.e. the token features) across
        # the heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k)

        # mat mul works on last two dimensions, hence swap n_heads and seq_len
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)


        # Scaled Dot-Product Attention
        # (batch_size, n_heads, seq_len, d_k) @ (batch_size, n_heads, d_k, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        context = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            is_causal=is_causal,
            attn_mask=None, #padding mask
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Merge Heads
        # (seq_len, n_heads, d_k)
        context = context.transpose(1, 2).contiguous()

        context = context.view(
            batch_size,
            -1,
            self.d_model,
        )

        # output (seq_len, d_model)
        return self.output_weights(context)

        # # Scaled Dot-Product Attention
        # # (batch_size, n_heads, seq_len, d_k) @ (batch_size, n_heads, d_k, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)

        # if mask is not None:
        #     # Mask shape: (seq_len, seq_len)
        #     scores = scores.masked_fill(mask == 0, float("-inf"))

        # # Softmax
        # # (batch_size, n_heads, seq_len, seq_len)
        # attn = torch.softmax(scores, dim=-1)

        # # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_k) -> (batch_size, n_heads, seq_len, d_k)
        # context = torch.matmul(attn, V)

        # # Merge Heads
        # # (seq_len, n_heads, d_k)
        # context = context.transpose(1, 2).contiguous()

        # # concat
        # # (seq_len, d_model)
        # context = context.view(batch_size, -1, self.d_model)

        # # output (seq_len, d_model)
        # return self.output_weights(context)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model=128,
        d_hidden=512,
        n_heads=8,
    ):
        super().__init__()
        self.d_model = d_model

        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        # bi-directional self-attention
        attn_out = self.self_attention(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            mask=None
        )
        x = x + attn_out

        # feed forward neural network
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x
    

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=128,
        d_hidden=512,
        n_heads=8,
    ):
        super().__init__()
        self.d_model = d_model

        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.cross_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, mask):
        
        # self-attention on input
        attn_out = self.self_attention(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            mask=mask
        )
        x = x + attn_out

        # cross-attention on memory
        cross_attn_out = self.cross_attention(
            query=self.norm2(x),
            key=memory,
            value=memory,
        )
        x = x + cross_attn_out

        # feed forward neural network
        ffn_out = self.ffn(self.norm3(x))
        x = x + ffn_out

        return x


    
class TransformerEncoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model,
            d_hidden,
            n_heads,
            n_layers,
            max_seq_len=500,
    ):
        super().__init__()

        # vector for each token
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )

        # basic position embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, d_hidden, n_heads)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        """
        x torch.tensor:
            (batch_size, seq_len)
        """
        _, seq_len = x.size()

        # embed tokens and add position embedding
        positions = torch.arange(0, seq_len).to(x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)

        # pass through all encoder layers
        for layer in self.layers:
            x = layer(x)

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
    ):
        super().__init__()
        
        # vector for each token
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )

        # basic position embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, d_hidden, n_heads)
                for _ in range(n_layers)
            ]
        )

        # fully connect layer to vocab
        self.fc_out = nn.Linear(d_model, vocab_size)


    def forward(self, x, memory):

        _, seq_len = x.size()

        # embed tokens and add position embedding
        positions = torch.arange(0, seq_len).to(x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(positions)

        # create Causal Mask
        # square matrix of size (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        mask = mask[None, None, :, :]

        # pass through all encoder layers
        for layer in self.layers:
            x = layer(x, memory, mask)
        
        # fully connected layer to logits
        logits = self.fc_out(x)

        return logits


class Transformer(nn.Module):
    """
    basic encoder-decoder transformer
    """
    def __init__(self, vocab_size, d_model, d_hidden, n_heads, n_layers):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_hidden=d_hidden,
            n_heads=n_heads,
            n_layers=n_layers
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_hidden=d_hidden,
            n_heads=n_heads,
            n_layers=n_layers
        )

    def forward(self, x1, x2):
        x1 = self.encoder(x1)

        x2 = self.decoder(x2, x1)

        return x2

