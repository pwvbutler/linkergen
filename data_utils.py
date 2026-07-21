from typing import List, Tuple, Optional, Dict

import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from smiles_tokenizer import SMILESTokenizer


# Numeric property columns in the TSV, in the order they are fed to the model.
# NOTE: there are SIX of these -> build LinkerGPT with property_dim=6
# (or trim this list to the subset you actually want to condition on).
PROPERTY_COLUMNS = [
    "num_bonds",
    "path_rotatable_bonds",
    "Ring_Count",
    "MW",
    "Fsp3",
    "cLogP",
    "TPSA",
    "heteroatom_fraction",
]


class SMILESDataset(Dataset):
    """
    Dataset for SMILES autoregressive language modeling, with optional
    per-molecule property conditioning.

    Each sample is returned as (token_tensor, prop_tensor), where prop_tensor
    is None when the dataset has no properties (pure LM pretraining mode).
    Padding is handled dynamically in the collate function.
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: SMILESTokenizer,
        properties: Optional[torch.Tensor] = None,
        prop_mean: Optional[torch.Tensor] = None,
        prop_std: Optional[torch.Tensor] = None,
        max_length: int = 512,
    ):
        """
        Args:
            smiles_list:
                List of SMILES strings.

            tokenizer:
                SMILES tokenizer instance.

            properties:
                Optional (N, property_dim) float tensor of raw (un-standardized)
                properties, aligned with smiles_list. If None, the dataset runs
                in unconditional LM mode and __getitem__ returns None for props.

            prop_mean, prop_std:
                Optional (property_dim,) standardization stats. If properties
                are given but stats are not, they are computed from `properties`
                here -- only do that for the TRAINING set; pass the train stats
                explicitly to the val/test datasets.

            max_length:
                Maximum allowed sequence length. Longer sequences are truncated.
        """
        if len(properties) != len(smiles_list) if properties is not None else False:
            raise ValueError("properties and smiles_list must be the same length")

        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_idx = tokenizer.pad_idx

        if properties is None:
            self.properties = None
            self.prop_mean = None
            self.prop_std = None
        else:
            properties = properties.float()
            if prop_mean is None or prop_std is None:
                prop_mean, prop_std = compute_property_stats(properties)
            self.prop_mean = prop_mean
            self.prop_std = prop_std
            # standardize once, up front
            self.properties = (properties - prop_mean) / prop_std
            #self.properties = properties

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            (token_tensor, prop_tensor)
            token_tensor: variable-length (L,) long tensor
            prop_tensor:  (property_dim,) float tensor, or None
        """
        smiles = self.smiles_list[idx]

        token_indices = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
        )

        # truncate
        token_indices = token_indices[: self.max_length]
        tokens = torch.tensor(token_indices, dtype=torch.long)

        if self.properties is None:
            return tokens, None
        return tokens, self.properties[idx]


def compute_property_stats(
    properties: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-column mean/std for standardization.

    Columns with (near-)zero variance -- e.g. a property that is constant or
    binary-and-all-equal in this split -- get std=1 so they map to 0 instead
    of exploding via divide-by-zero.

    Args:
        properties: (N, property_dim) raw property tensor.

    Returns:
        (mean, std), each (property_dim,).
    """
    properties = properties.float()
    mean = properties.mean(dim=0)
    std = properties.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def collate_fn(
    batch: List[Tuple[torch.Tensor, Optional[torch.Tensor]]],
    pad_idx: int,
) -> Dict[str, torch.Tensor]:
    """
    Transformer-ready collate function.

    Creates:
        input_ids:      (B, T) input token ids   (shifted right)
        labels:         (B, T) target token ids  (shifted left)
        attention_mask: (B, T) bool, True = real token, False = padding
        prop_vec:       (B, property_dim) float -- ONLY present when the
                        batch carries properties.

    Example:
        Original: [BOS, C, C, O, EOS]
        input_ids: [BOS, C, C, O]
        labels:    [C, C, O, EOS]
    """
    token_seqs = [tokens for tokens, _ in batch]
    props = [p for _, p in batch]

    # autoregressive LM shifting
    inputs = [seq[:-1] for seq in token_seqs]
    labels = [seq[1:] for seq in token_seqs]

    # dynamic padding
    input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    labels = pad_sequence(labels, batch_first=True, padding_value=pad_idx)

    attention_mask = input_ids != pad_idx

    out = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

    # only attach prop_vec if this dataset actually has properties
    if props[0] is not None:
        out["prop_vec"] = torch.stack(props, dim=0)

    return out


def create_dataloader(
    smiles_list: List[str],
    tokenizer: SMILESTokenizer,
    properties: Optional[torch.Tensor] = None,
    prop_mean: Optional[torch.Tensor] = None,
    prop_std: Optional[torch.Tensor] = None,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a Transformer-ready DataLoader.

    Pass properties=None for unconditional pretraining; pass a property tensor
    (plus train stats for val/test) for conditional fine-tuning. When present,
    each batch gains a "prop_vec" key you forward straight into the model:
        model(batch["input_ids"], batch.get("prop_vec"))

    Returns:
        PyTorch DataLoader.
    """
    dataset = SMILESDataset(
        smiles_list=smiles_list,
        tokenizer=tokenizer,
        properties=properties,
        prop_mean=prop_mean,
        prop_std=prop_std,
        max_length=max_length,
    )

    pad_idx = dataset.pad_idx

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_idx),
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader


def load_smiles_from_file(filepath: str) -> List[str]:
    """
    Load SMILES strings from a plain text file (one per line).

    Use this for the unlabeled pretraining corpus.
    """
    with open(filepath, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    return smiles_list


def load_dataset_from_file(
    filepath: str,
    property_columns: Optional[List[str]] = None,
    smiles_column: str = "core_smiles",
    sep: str = ",",
) -> Tuple[List[str], Optional[torch.Tensor]]:
    """
    Load SMILES + properties from a tab-separated file with a header.

    Use this for the conditional fine-tuning set.

    Args:
        filepath:         path to the TSV.
        property_columns: numeric columns to use as the condition vector.
                          Defaults to PROPERTY_COLUMNS. Pass [] or None-of-them
                          to load SMILES only.
        smiles_column:    name of the SMILES column.
        sep:              column separator (tab by default).

    Returns:
        (smiles_list, properties) where properties is an (N, P) float tensor,
        or None if no property_columns were requested.
    """
    import pandas as pd

    if property_columns is None:
        property_columns = PROPERTY_COLUMNS

    df = pd.read_csv(filepath, sep=sep)

    missing = [c for c in [smiles_column, *property_columns] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {filepath}: {missing}")

    smiles_list = df[smiles_column].astype(str).tolist()

    if property_columns:
        properties = torch.tensor(
            df[property_columns].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )
    else:
        properties = None

    return smiles_list, properties


def train_val_split(
    smiles_list: List[str],
    properties: Optional[torch.Tensor] = None,
    val_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Split into train/validation sets, keeping SMILES and properties aligned.

    Returns:
        (train_smiles, val_smiles, train_props, val_props)
        train_props / val_props are None when properties is None.
    """
    n = len(smiles_list)
    indices = list(range(n))

    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)

    split_idx = int(n * (1 - val_ratio))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_smiles = [smiles_list[i] for i in train_idx]
    val_smiles = [smiles_list[i] for i in val_idx]

    if properties is None:
        return train_smiles, val_smiles, None, None

    train_props = properties[torch.tensor(train_idx)]
    val_props = properties[torch.tensor(val_idx)]
    return train_smiles, val_smiles, train_props, val_props


def load_core_smiles_embeddings_from_json(
    filepath: str,
    smiles_key: str = "core_smiles",
    embedding_key: str = "embedding",
) -> Tuple[List[str], torch.Tensor]:
    """
    Load core SMILES + precomputed embeddings from a JSON file.

    Expected JSON format:
    [
      {
        "original_smiles": "...",
        "core_smiles": "...",
        "text_description": "...",
        "embedding": [0.1, -0.2, ...]
      }
    ]

    Returns:
        smiles_list:
            List of core SMILES strings.

        properties:
            Float tensor of shape (N, embedding_dim). This is used as prop_vec
            by the existing Dataset/collate_fn/model pipeline.
    """
    import json

    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("JSON file must contain a list of records.")

    smiles_list = []
    embeddings = []

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {i} is not a JSON object.")

        if smiles_key not in record:
            raise ValueError(f"Record {i} is missing required key: {smiles_key}")

        if embedding_key not in record:
            raise ValueError(f"Record {i} is missing required key: {embedding_key}")

        smiles = record[smiles_key]
        embedding = record[embedding_key]

        if not isinstance(smiles, str):
            raise ValueError(f"Record {i} field {smiles_key} must be a string.")

        if not isinstance(embedding, list):
            raise ValueError(f"Record {i} field {embedding_key} must be a list.")

        smiles_list.append(smiles)
        embeddings.append(embedding)

    properties = torch.tensor(embeddings, dtype=torch.float32)

    if properties.ndim != 2:
        raise ValueError(
            f"Embeddings must form a 2D tensor of shape (N, embedding_dim), "
            f"but got shape {tuple(properties.shape)}"
        )

    return smiles_list, properties
