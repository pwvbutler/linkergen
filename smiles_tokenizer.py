import os
import re
from typing import Dict, List, Optional, Sequence

import json

PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
# PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

class SMILESTokenizer:
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.regex = re.compile(PATTERN)
        self.special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ]
        self.token2idx = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.idx2token = {i: tok for i, tok in enumerate(self.special_tokens)}

    def build_vocab(self, smiles_list: List[str]):
        for tok in sorted({t for smi in smiles_list for t in self.regex.findall(smi)}):
            if tok not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[tok] = idx
                self.idx2token[idx] = tok
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.token2idx)

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.token2idx[self.BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    def tokenize(self, smiles: str) -> List[str]:
        return self.regex.findall(smiles)

    def encode(self, smiles: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(smiles)
        ids = [self.token2idx.get(t, self.unk_idx) for t in tokens]
        if add_special_tokens:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        special = set(range(len(self.special_tokens)))
        return "".join(
            self.idx2token[i]
            for i in ids
            if not (remove_special_tokens and i in special)
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.token2idx, f)

    @classmethod
    def load(cls, path: str):
        tok = cls()
        with open(path) as f:
            tok.token2idx = json.load(f)
        tok.idx2token = {int(i): t for t, i in tok.token2idx.items()}
        return tok