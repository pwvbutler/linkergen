import torch
import torch.nn as nn
import torch.nn.functional as F

from multihead_attention import MultiHeadAttention


class PropertyEncoder(nn.Module):
    def __init__(self,
            property_dim,
            d_model,
            n_memory,
            dropout=0.1,
        ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(property_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.slots = nn.Parameter(torch.randn(n_memory, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, prop_vec):
        h = self.mlp(prop_vec)                       # (B, d_model)
        h = h.unsqueeze(1) + self.slots.unsqueeze(0)  # (B, n_memory, d_model)
        return self.norm(h)
        #return self.dropout(self.norm(h))


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

    def forward(self, x, memory=None):

        # self-attention on input
        attn_out = self.self_attention(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            is_causal=True
        )
        #x = x + attn_out
        x = x + self.dropout1(attn_out)

        # cross-attention on memory
        if memory is not None:
            cross_attn_out = self.cross_attention(
                query=self.norm2(x),
                key=memory,
                value=memory,
                is_causal=False
            )
            #x = x + cross_attn_out
            x = x + self.dropout2(cross_attn_out)

        # feed forward neural network
        ffn_out = self.ffn(self.norm3(x))
        #x = x + ffn_out
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

        # basic position embedding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the denominator term using log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Assign sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer ensures pe is saved with the model state_dict but not trained
        self.register_buffer('pe', pe)

        # self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, d_hidden, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)


    def forward(self, x, memory=None):

        batch_size, seq_len = x.size()

        # embed tokens and add position embedding
        #positions = torch.arange(0, seq_len).to(x.device).unsqueeze(0)
        #x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.token_embedding(x) + self.pe[:, :seq_len]
        x = self.emb_dropout(x)

        # create Causal Mask
        # a square matrix of size (seq_len, seq_len)
        # mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        # mask = mask[None, None, :, :]

        # pass through all decoder layers
        for layer in self.layers:
            x = layer(x, memory)
        
        logits = self.fc_out(self.norm_out(x))

        return logits


class LinkerGen(nn.Module):
    def __init__(self,
        vocab_size,
        d_model=128,
        d_hidden=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=512,
        property_dim=4,
        n_memory=8,
        dropout=0.05,
    ):
        super().__init__()

        self.property_encoder = PropertyEncoder(
            property_dim=property_dim,
            d_model=d_model,
            n_memory=n_memory
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            d_hidden=d_hidden,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward(self, x, prop_vec=None):

        memory = None
        if prop_vec is not None:
            memory = self.property_encoder(prop_vec)
        logits = self.decoder(x, memory)

        return logits
    


    @torch.no_grad()
    def generate(
        self,
        input_ids,
        prop_vec=None,
        max_new_tokens=200,
        temperature=0.8,
        eos_token_id=None,
        pad_token_id=None,
    ):
        generated = input_ids.clone()
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)

        for _ in range(max_new_tokens):
            logits = self.forward(generated, prop_vec)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if temperature > 0:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            if eos_token_id is not None:
                # Once finished, keep emitting EOS or PAD, not random extra tokens.
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
