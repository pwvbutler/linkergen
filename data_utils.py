"""
Data Loading and Preprocessing Utilities for SMILES Sequences.

Transformer-ready utilities for loading, tokenizing, dynamically padding,
and batching SMILES molecular sequences for autoregressive language modeling.
"""

from typing import List, Tuple, Optional, Dict
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from smiles_tokenizer import SMILESTokenizer


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


class SMILESDataset(Dataset):
    """
    Dataset for SMILES autoregressive language modeling.

    Each sample is returned as a variable-length token tensor.
    Padding is handled dynamically in the collate function.
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: SMILESTokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            smiles_list:
                List of SMILES strings.

            tokenizer:
                SMILES tokenizer instance.

            max_length:
                Maximum allowed sequence length.
                Longer sequences are truncated.
        """
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.pad_idx = tokenizer.pad_idx

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            Variable-length token tensor.
        """
        smiles = self.smiles_list[idx]

        token_indices = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
        )

        # truncate
        token_indices = token_indices[: self.max_length]

        return torch.tensor(token_indices, dtype=torch.long)


# ---------------------------------------------------------------------
# Collate Function
# ---------------------------------------------------------------------


def collate_fn(
    batch: List[torch.Tensor],
    pad_idx: int,
) -> Dict[str, torch.Tensor]:
    """
    Transformer-ready collate function.

    Creates:
        input_ids:
            Input token ids shifted right.
            Shape: (B, T)

        labels:
            Target token ids shifted left.
            Shape: (B, T)

        attention_mask:
            Boolean mask where True = real token,
            False = padding token.
            Shape: (B, T)

    Example:
        Original:
            [BOS, C, C, O, EOS]

        input_ids:
            [BOS, C, C, O]

        labels:
            [C, C, O, EOS]
    """

    # autoregressive LM shifting
    inputs = [seq[:-1] for seq in batch]
    labels = [seq[1:] for seq in batch]

    # dynamic padding
    input_ids = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=pad_idx,
    )

    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=pad_idx,
    )

    # attention mask
    attention_mask = input_ids != pad_idx

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------


def create_dataloader(
    smiles_list: List[str],
    tokenizer: SMILESTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create Transformer-ready DataLoader.

    Args:
        smiles_list:
            List of SMILES strings.

        tokenizer:
            SMILES tokenizer instance.

        batch_size:
            Batch size.

        max_length:
            Maximum sequence length.

        shuffle:
            Whether to shuffle dataset.

        num_workers:
            Number of worker processes.

    Returns:
        PyTorch DataLoader.
    """

    dataset = SMILESDataset(
        smiles_list=smiles_list,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(
            batch,
            dataset.pad_idx,
        ),
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def load_smiles_from_file(filepath: str) -> List[str]:
    """
    Load SMILES strings from text file.

    Assumes one SMILES string per line.

    Args:
        filepath:
            Path to SMILES file.

    Returns:
        List of SMILES strings.
    """

    with open(filepath, "r") as f:
        smiles_list = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return smiles_list


def train_val_split(
    smiles_list: List[str],
    val_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Split SMILES list into train/validation sets.

    Args:
        smiles_list:
            List of SMILES strings.

        val_ratio:
            Fraction used for validation.

        seed:
            Random seed.

    Returns:
        (train_list, val_list)
    """

    if seed is not None:
        random.seed(seed)

    shuffled = smiles_list.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - val_ratio))

    train_list = shuffled[:split_idx]
    val_list = shuffled[split_idx:]

    return train_list, val_list