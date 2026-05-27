import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SmilesGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h=None):
        """
        Args:
            x : (batch, seq_len) — token indices
            h : (num_layers, batch, hidden_dim), optional
        Returns:
            logits  : (batch, seq_len, vocab_size)
            h_final : (num_layers, batch, hidden_dim)
        """
        embedded = self.dropout(self.embedding(x))
        out, h_final = self.rnn(embedded, h)
        logits = self.fc_out(out)
        return logits, h_final

    def init_hidden(self, batch_size, device=None):
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device,
        )

    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        n: int = 1,
        max_len: int = 100,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> list[str]:
        """
        Autoregressively sample SMILES strings.

        Args:
            tokenizer   : SMILESTokenizer — needed for bos/eos/decode
            n           : number of SMILES to generate
            max_len     : maximum sequence length
            temperature : >1 more random, <1 more greedy
            device      : device to run on
        Returns:
            list of SMILES strings (special tokens stripped)
        """
        self.eval()
        x = torch.full((n, 1), tokenizer.bos_idx, dtype=torch.long, device=device)
        h = None
        sequences = [[] for _ in range(n)]
        done = [False] * n

        for _ in range(max_len):
            logits, h = self(x, h)  # logits: (n, 1, vocab_size)
            probs = torch.softmax(
                logits[:, -1] / temperature, dim=-1
            )  # (n, vocab_size)
            next_token = torch.multinomial(probs, 1)  # (n, 1)

            for i in range(n):
                if not done[i]:
                    tok = next_token[i].item()
                    if tok == tokenizer.eos_idx:
                        done[i] = True
                    else:
                        sequences[i].append(tok)

            if all(done):
                break

            x = next_token

        return [tokenizer.decode(seq) for seq in sequences]


