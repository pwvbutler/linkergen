"""SMILES dataset, collation, and data-loading helpers, shared by all three
conditioning modes. A sample's condition depends on `condition_type`:
  - "none":     no condition.
  - "property": an (property_dim,) standardized float tensor.
  - "text":     a raw text description string. Nach0 encoding happens later,
                per-batch, in train.py/sample.py, not here.
"""

from typing import Dict, List, Optional, Tuple, Union

import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .tokenizer import SMILESTokenizer


# Numeric property columns in the TSV, in the order they are fed to the model.
# NOTE: there are SIX of these, so build LinkerGen with property_dim=6
# (or trim this list to the desired subset of conditioning columns).
PROPERTY_COLUMNS = [
    "molecule_type_2-linear",
    "molecule_type_3-planar",
    "molecule_type_acyclic_core_2-bent",
    "molecule_type_cyclic_core_2-bent",
    "num_bonds",
    "heteroatom_fraction",
]


def compute_property_stats(
    properties: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-column mean/std for standardization.

    Continuous columns are standardized. One-hot molecule_type_* columns are
    left unchanged by setting mean=0, std=1.

    Args:
        properties: (N, property_dim) raw property tensor.

    Returns:
        (mean, std), each (property_dim,).
    """
    properties = properties.float()

    mean = properties.mean(dim=0)
    std = properties.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    # Do not standardize one-hot molecule type columns.
    # This assumes PROPERTY_COLUMNS is in the same order as the tensor columns.
    for i, col in enumerate(PROPERTY_COLUMNS):
        if col.startswith("molecule_type_"):
            mean[i] = 0.0
            std[i] = 1.0

    return mean, std


class SMILESDataset(Dataset):
    """
    Dataset for SMILES autoregressive language modeling, with an optional
    per-molecule condition whose shape is fixed by `condition_type`:
      - "none":     conditions must be None.
      - "property": conditions is an (N, property_dim) float tensor, standardized
                    with prop_mean/prop_std (computed here if not given; do that
                    only for the TRAINING set, and pass the train stats
                    explicitly to the val/test datasets).
      - "text":     conditions is a list[str] of raw descriptions, stored as-is.

    Each sample is returned as (token_tensor, condition), where condition is a
    tensor row, a string, or None, matching condition_type. Padding is handled
    dynamically in the collate function.
    """

    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: SMILESTokenizer,
        condition_type: str = "none",
        conditions: Optional[Union[torch.Tensor, List[str]]] = None,
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

            condition_type:
                One of "none" / "property" / "text"; fixes the shape conditions
                must take.

            conditions:
                Raw (un-standardized) conditions aligned with smiles_list, or
                None to run in unconditional LM mode.

            prop_mean, prop_std:
                Property mode only: optional (property_dim,) standardization
                stats. Computed from `conditions` here if not given.

            max_length:
                Maximum allowed sequence length. Longer sequences are truncated.
        """
        if condition_type not in ("none", "property", "text"):
            raise ValueError(f"condition_type must be one of 'none'/'property'/'text', got {condition_type!r}")
        if condition_type == "none" and conditions is not None:
            raise ValueError("condition_type='none' requires conditions=None")
        if conditions is not None and len(conditions) != len(smiles_list):
            raise ValueError("conditions and smiles_list must be the same length")

        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.condition_type = condition_type
        self.max_length = max_length
        self.pad_idx = tokenizer.pad_idx
        self.prop_mean = None
        self.prop_std = None

        if condition_type == "property" and conditions is not None:
            conditions = conditions.float()
            if prop_mean is None or prop_std is None:
                prop_mean, prop_std = compute_property_stats(conditions)
            self.prop_mean = prop_mean
            self.prop_std = prop_std
            # standardize once, up front
            conditions = (conditions - prop_mean) / prop_std

        self.conditions = conditions

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[Union[torch.Tensor, str]]]:
        """
        Returns:
            (token_tensor, condition)
            token_tensor: variable-length (L,) long tensor
            condition:    (property_dim,) float tensor, a description string, or None
        """
        token_indices = self.tokenizer.encode(
            self.smiles_list[idx],
            add_special_tokens=True,
        )
        tokens = torch.tensor(token_indices[: self.max_length], dtype=torch.long)

        if self.conditions is None:
            return tokens, None
        return tokens, self.conditions[idx]


def collate_fn(
    batch: List[Tuple[torch.Tensor, Optional[Union[torch.Tensor, str]]]],
    pad_idx: int,
    condition_type: str = "none",
) -> Dict:
    """
    Transformer-ready collate function.

    Creates:
        input_ids:      (B, T) input token ids   (shifted right)
        labels:         (B, T) target token ids  (shifted left)
        attention_mask: (B, T) bool, True = real token, False = padding
        condition:      only present when condition_type != "none":
                          "property": (B, property_dim) float tensor
                          "text":     list[str] of length B

    Example:
        Original: [BOS, C, C, O, EOS]
        input_ids: [BOS, C, C, O]
        labels:    [C, C, O, EOS]
    """
    token_seqs, conditions = zip(*batch)

    # autoregressive LM shifting
    inputs = [seq[:-1] for seq in token_seqs]
    labels = [seq[1:] for seq in token_seqs]

    # dynamic padding
    input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    labels = pad_sequence(labels, batch_first=True, padding_value=pad_idx)

    out = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids != pad_idx,
    }

    if condition_type == "property":
        out["condition"] = torch.stack(conditions, dim=0)
    elif condition_type == "text":
        out["condition"] = list(conditions)

    return out


def create_dataloader(
    smiles_list: List[str],
    tokenizer: SMILESTokenizer,
    condition_type: str = "none",
    conditions: Optional[Union[torch.Tensor, List[str]]] = None,
    prop_mean: Optional[torch.Tensor] = None,
    prop_std: Optional[torch.Tensor] = None,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a Transformer-ready DataLoader.

    Pass condition_type="none" for unconditional (pre)training; "property" with
    a property tensor (plus train stats for val/test); or "text" with a list of
    description strings. When present, each batch gains a "condition" key to
    forward straight into the model:
        model(batch["input_ids"], batch.get("condition"))

    Returns:
        PyTorch DataLoader.
    """
    dataset = SMILESDataset(
        smiles_list=smiles_list,
        tokenizer=tokenizer,
        condition_type=condition_type,
        conditions=conditions,
        prop_mean=prop_mean,
        prop_std=prop_std,
        max_length=max_length,
    )

    pad_idx = dataset.pad_idx

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_idx, condition_type),
        pin_memory=torch.cuda.is_available(),
    )


def load_smiles_from_file(filepath: str) -> List[str]:
    """
    Load SMILES strings from a plain text file (one per line).

    Use this for the unlabeled pretraining corpus (condition_type="none").
    """
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_dataset_from_file(
    filepath: str,
    property_columns: Optional[List[str]] = None,
    smiles_column: str = "original_smiles",
    sep: str = ",",
) -> Tuple[List[str], Optional[torch.Tensor]]:
    """
    Load SMILES + properties from a delimited file with a header.

    Use this for the conditional fine-tuning set (condition_type="property").

    Args:
        filepath:         path to the file.
        property_columns: numeric columns to use as the condition vector.
                          Defaults to PROPERTY_COLUMNS. Pass [] to load SMILES only.
        smiles_column:    name of the SMILES column.
        sep:              column separator.

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


def load_core_smiles_embeddings_from_json(
    filepath: str,
    smiles_key: str = "core_smiles",
    embedding_key: str = "embedding",
) -> Tuple[List[str], torch.Tensor]:
    """
    Load core SMILES + precomputed embeddings from a JSON file
    (condition_type="property", --data_format json_embeddings).

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
        smiles_list: List of core SMILES strings.
        properties:  Float tensor of shape (N, embedding_dim), used directly as
                     the "property" condition by the dataset/collate/model pipeline.
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


def load_smiles_and_descriptions_from_json(
    filepath: str,
    smiles_key: str = "original_smiles",
    description_key: str = "text_description",
) -> Tuple[List[str], List[str]]:
    """
    Load SMILES strings and text descriptions from a JSON file (condition_type="text").

    Expected format:
    [
      {
        "core_smiles": "...",
        "text_description": "..."
      }
    ]

    Args:
        filepath:        Path to the JSON file.
        smiles_key:      Key for the SMILES string in each record.
        description_key: Key for the text description in each record.

    Returns:
        (smiles_list, descriptions)
        Both are lists of strings of equal length.
    """
    import json

    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    smiles_list = []
    descriptions = []
    for i, record in enumerate(records):
        for key in (smiles_key, description_key):
            if key not in record:
                raise ValueError(f"Record {i} missing key: {key}")
        smiles_list.append(record[smiles_key])
        descriptions.append(record[description_key])

    return smiles_list, descriptions


def train_val_split(
    smiles_list: List[str],
    conditions: Optional[Union[torch.Tensor, List[str]]] = None,
    val_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str], Optional[Union[torch.Tensor, List[str]]], Optional[Union[torch.Tensor, List[str]]]]:
    """
    Split into train/validation sets, keeping SMILES and conditions aligned.

    `conditions` may be a (N, property_dim) tensor (property mode), a list of
    strings (text mode), or None (unconditional mode).

    Returns:
        (train_smiles, val_smiles, train_conditions, val_conditions)
        train_conditions / val_conditions are None when conditions is None.
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

    if conditions is None:
        return train_smiles, val_smiles, None, None

    if isinstance(conditions, torch.Tensor):
        train_conditions = conditions[torch.tensor(train_idx)]
        val_conditions = conditions[torch.tensor(val_idx)]
    else:
        train_conditions = [conditions[i] for i in train_idx]
        val_conditions = [conditions[i] for i in val_idx]

    return train_smiles, val_smiles, train_conditions, val_conditions
