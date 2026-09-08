"""Unit tests for the data module.

Mirrors the spec in plan.md Phase 6: one SMILESDataset/collate_fn/
train_val_split shared across all three condition_type values, and the
loaders ported unchanged from the two legacy data_utils.py files.
"""

import json

import pytest
import torch

from linkergen.data import (
    SMILESDataset,
    collate_fn,
    compute_property_stats,
    create_dataloader,
    load_core_smiles_embeddings_from_json,
    load_dataset_from_file,
    load_smiles_and_descriptions_from_json,
    load_smiles_from_file,
    train_val_split,
)
from linkergen.tokenizer import SMILESTokenizer

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "ClCCBr"]


@pytest.fixture
def tokenizer() -> SMILESTokenizer:
    return SMILESTokenizer().build_vocab(SMILES)


@pytest.fixture
def two_plain_columns(monkeypatch):
    # compute_property_stats looks up PROPERTY_COLUMNS by index to find
    # one-hot columns, so tests using a differently-sized property tensor
    # must monkeypatch it to match (or nothing to do with one-hot columns).
    import linkergen.data as data_module

    monkeypatch.setattr(data_module, "PROPERTY_COLUMNS", ["colA", "colB"])


@pytest.fixture
def one_plain_column(monkeypatch):
    import linkergen.data as data_module

    monkeypatch.setattr(data_module, "PROPERTY_COLUMNS", ["colA"])


# --- SMILESDataset construction ------------------------------------------

def test_rejects_unknown_condition_type(tokenizer):
    with pytest.raises(ValueError):
        SMILESDataset(SMILES, tokenizer, condition_type="bogus")


def test_none_mode_rejects_conditions(tokenizer):
    with pytest.raises(ValueError):
        SMILESDataset(SMILES, tokenizer, condition_type="none", conditions=torch.randn(len(SMILES), 3))


def test_rejects_length_mismatch(tokenizer):
    with pytest.raises(ValueError):
        SMILESDataset(SMILES, tokenizer, condition_type="text", conditions=["a description"])


# --- "none" mode ----------------------------------------------------------

def test_none_mode_getitem_has_no_condition(tokenizer):
    dataset = SMILESDataset(SMILES, tokenizer, condition_type="none")
    tokens, condition = dataset[0]
    assert condition is None
    assert tokens[0].item() == tokenizer.bos_idx


# --- "property" mode -------------------------------------------------------

def test_property_mode_standardizes_using_computed_stats(tokenizer, two_plain_columns):
    properties = torch.tensor([[10.0, 1.0], [20.0, 0.0], [30.0, 1.0], [40.0, 0.0]])
    dataset = SMILESDataset(SMILES, tokenizer, condition_type="property", conditions=properties)

    mean, std = compute_property_stats(properties)
    expected = (properties - mean) / std
    assert torch.allclose(dataset.conditions, expected)
    assert torch.equal(dataset.prop_mean, mean)
    assert torch.equal(dataset.prop_std, std)


def test_property_mode_reuses_given_stats_instead_of_recomputing(tokenizer):
    properties = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    given_mean, given_std = torch.tensor([100.0]), torch.tensor([5.0])

    dataset = SMILESDataset(
        SMILES, tokenizer, condition_type="property", conditions=properties,
        prop_mean=given_mean, prop_std=given_std,
    )

    assert torch.equal(dataset.prop_mean, given_mean)
    assert torch.equal(dataset.prop_std, given_std)
    assert torch.allclose(dataset.conditions, (properties - given_mean) / given_std)


def test_property_mode_getitem_returns_standardized_row(tokenizer, one_plain_column):
    properties = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    dataset = SMILESDataset(SMILES, tokenizer, condition_type="property", conditions=properties)
    _, condition = dataset[1]
    assert torch.equal(condition, dataset.conditions[1])


# --- "text" mode ------------------------------------------------------------

def test_text_mode_getitem_returns_raw_string_unstandardized(tokenizer):
    descriptions = [f"description {i}" for i in range(len(SMILES))]
    dataset = SMILESDataset(SMILES, tokenizer, condition_type="text", conditions=descriptions)
    _, condition = dataset[2]
    assert condition == "description 2"


# --- compute_property_stats -------------------------------------------------

def test_compute_property_stats_leaves_one_hot_columns_untouched(monkeypatch):
    import linkergen.data as data_module

    monkeypatch.setattr(data_module, "PROPERTY_COLUMNS", ["molecule_type_linear", "num_bonds"])
    properties = torch.tensor([[1.0, 10.0], [0.0, 20.0], [1.0, 30.0], [0.0, 40.0]])

    mean, std = compute_property_stats(properties)

    assert mean[0].item() == 0.0
    assert std[0].item() == 1.0
    assert mean[1].item() == pytest.approx(properties[:, 1].mean().item())
    assert std[1].item() == pytest.approx(properties[:, 1].std().item())


def test_compute_property_stats_guards_against_zero_variance(one_plain_column):
    properties = torch.tensor([[5.0], [5.0], [5.0]])
    mean, std = compute_property_stats(properties)
    assert std[0].item() == 1.0  # not 0, which would divide-by-zero downstream


# --- collate_fn --------------------------------------------------------------

def test_collate_shifts_tokens_by_one(tokenizer):
    dataset = SMILESDataset(SMILES, tokenizer, condition_type="none")
    batch = [dataset[0], dataset[1]]
    out = collate_fn(batch, pad_idx=tokenizer.pad_idx, condition_type="none")

    tokens0, _ = batch[0]
    assert torch.equal(out["input_ids"][0, : len(tokens0) - 1], tokens0[:-1])
    assert torch.equal(out["labels"][0, : len(tokens0) - 1], tokens0[1:])
    assert "condition" not in out


def test_collate_stacks_property_conditions(tokenizer, one_plain_column):
    properties = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    dataset = SMILESDataset(SMILES, tokenizer, condition_type="property", conditions=properties)
    batch = [dataset[0], dataset[1]]
    out = collate_fn(batch, pad_idx=tokenizer.pad_idx, condition_type="property")
    assert out["condition"].shape == (2, 1)


def test_collate_lists_text_conditions(tokenizer):
    descriptions = [f"description {i}" for i in range(len(SMILES))]
    dataset = SMILESDataset(SMILES, tokenizer, condition_type="text", conditions=descriptions)
    batch = [dataset[0], dataset[1]]
    out = collate_fn(batch, pad_idx=tokenizer.pad_idx, condition_type="text")
    assert out["condition"] == ["description 0", "description 1"]


def test_collate_attention_mask_marks_padding(tokenizer):
    dataset = SMILESDataset(SMILES, tokenizer, condition_type="none")
    batch = [dataset[i] for i in range(len(SMILES))]
    out = collate_fn(batch, pad_idx=tokenizer.pad_idx, condition_type="none")
    assert torch.equal(out["attention_mask"], out["input_ids"] != tokenizer.pad_idx)


# --- create_dataloader --------------------------------------------------------

def test_create_dataloader_property_batch_has_condition_key(tokenizer, one_plain_column):
    properties = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    loader = create_dataloader(
        SMILES, tokenizer, condition_type="property", conditions=properties,
        batch_size=2, shuffle=False,
    )
    batch = next(iter(loader))
    assert batch["condition"].shape == (2, 1)


def test_create_dataloader_none_batch_has_no_condition_key(tokenizer):
    loader = create_dataloader(SMILES, tokenizer, condition_type="none", batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert "condition" not in batch


# --- loaders -------------------------------------------------------------

def test_load_smiles_from_file_strips_blank_lines(tmp_path):
    path = tmp_path / "smiles.smi"
    path.write_text("CCO\n\nc1ccccc1\n  \nCCN\n")
    assert load_smiles_from_file(str(path)) == ["CCO", "c1ccccc1", "CCN"]


def test_load_dataset_from_file_reads_smiles_and_properties(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("original_smiles,num_bonds,heteroatom_fraction\nCCO,1,0.5\nCCN,2,0.25\n")

    smiles_list, properties = load_dataset_from_file(
        str(path), property_columns=["num_bonds", "heteroatom_fraction"]
    )

    assert smiles_list == ["CCO", "CCN"]
    assert torch.equal(properties, torch.tensor([[1.0, 0.5], [2.0, 0.25]]))


def test_load_dataset_from_file_raises_on_missing_column(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("original_smiles,num_bonds\nCCO,1\n")
    with pytest.raises(ValueError):
        load_dataset_from_file(str(path), property_columns=["num_bonds", "heteroatom_fraction"])


def test_load_core_smiles_embeddings_from_json(tmp_path):
    records = [
        {"core_smiles": "CCO", "embedding": [0.1, 0.2]},
        {"core_smiles": "CCN", "embedding": [0.3, 0.4]},
    ]
    path = tmp_path / "embeddings.json"
    path.write_text(json.dumps(records))

    smiles_list, properties = load_core_smiles_embeddings_from_json(str(path))
    assert smiles_list == ["CCO", "CCN"]
    assert properties.shape == (2, 2)


def test_load_core_smiles_embeddings_from_json_rejects_missing_key(tmp_path):
    path = tmp_path / "embeddings.json"
    path.write_text(json.dumps([{"core_smiles": "CCO"}]))
    with pytest.raises(ValueError):
        load_core_smiles_embeddings_from_json(str(path))


def test_load_smiles_and_descriptions_from_json(tmp_path):
    records = [
        {"original_smiles": "CCO", "text_description": "a small alcohol linker"},
        {"original_smiles": "CCN", "text_description": "a small amine linker"},
    ]
    path = tmp_path / "descriptions.json"
    path.write_text(json.dumps(records))

    smiles_list, descriptions = load_smiles_and_descriptions_from_json(str(path))
    assert smiles_list == ["CCO", "CCN"]
    assert descriptions == ["a small alcohol linker", "a small amine linker"]


def test_load_smiles_and_descriptions_from_json_rejects_missing_key(tmp_path):
    path = tmp_path / "descriptions.json"
    path.write_text(json.dumps([{"original_smiles": "CCO"}]))
    with pytest.raises(ValueError):
        load_smiles_and_descriptions_from_json(str(path))


# --- train_val_split -----------------------------------------------------

def test_train_val_split_none_conditions():
    smiles_list = [f"smi{i}" for i in range(10)]
    train_smiles, val_smiles, train_conditions, val_conditions = train_val_split(
        smiles_list, conditions=None, val_ratio=0.2, seed=0
    )
    assert len(train_smiles) == 8
    assert len(val_smiles) == 2
    assert train_conditions is None
    assert val_conditions is None
    assert set(train_smiles) | set(val_smiles) == set(smiles_list)


def test_train_val_split_keeps_tensor_conditions_aligned():
    smiles_list = [f"smi{i}" for i in range(10)]
    conditions = torch.arange(10).float().unsqueeze(1)  # conditions[i] == i

    train_smiles, val_smiles, train_conditions, val_conditions = train_val_split(
        smiles_list, conditions=conditions, val_ratio=0.3, seed=1
    )

    for smi, cond in zip(train_smiles + val_smiles, torch.cat([train_conditions, val_conditions])):
        assert int(cond.item()) == int(smi[len("smi"):])


def test_train_val_split_keeps_string_conditions_aligned():
    smiles_list = [f"smi{i}" for i in range(10)]
    conditions = [f"description {i}" for i in range(10)]

    train_smiles, val_smiles, train_conditions, val_conditions = train_val_split(
        smiles_list, conditions=conditions, val_ratio=0.3, seed=2
    )

    for smi, cond in zip(train_smiles + val_smiles, train_conditions + val_conditions):
        assert cond == f"description {smi[len('smi'):]}"


def test_train_val_split_is_reproducible_with_same_seed():
    smiles_list = [f"smi{i}" for i in range(20)]
    split_a = train_val_split(smiles_list, seed=42)
    split_b = train_val_split(smiles_list, seed=42)
    assert split_a[0] == split_b[0]
    assert split_a[1] == split_b[1]
