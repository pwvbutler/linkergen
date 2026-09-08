"""Unit tests for the train module.

Mirrors the spec in plan.md Phase 7: covers the three testable units factored
out of the CLI entrypoint -- prepare_condition, train_epoch, validate -- for
all three condition_type values. main() itself (argparse + the full data /
model / checkpoint / sampling pipeline) is a CLI entrypoint rather than a
unit and is left to Phase 9's manual smoke tests, per plan.

Text-mode tests use a FakeNach0Encoder (a stand-in with the same
descriptions/device -> (hidden_states, mask) call signature as the real
Nach0Encoder) so this file stays fast and network-free, the same principle
test_model.py already applies to text-mode wiring. One additional test
exercises the real Nach0Encoder end to end and is marked `network` (skipped
by default, see pyproject.toml), matching test_nach0.py's convention.
"""

import math
import random

import pytest
import torch
import torch.nn as nn

from linkergen.data import create_dataloader
from linkergen.model import LinkerGen
from linkergen.tokenizer import SMILESTokenizer
from linkergen.train import prepare_condition, train_epoch, validate

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "ClCCBr"]

D_MODEL = 16
N_HEADS = 2
N_LAYERS = 2
D_HIDDEN = 32


class FakeNach0Encoder:
    """Stands in for the real Nach0Encoder in tests that exercise train.py's
    wiring without needing a network download of the actual nach0 model.
    Records every call so tests can check call count/arguments."""

    def __init__(self, hidden_size=10, seq_len=5):
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.calls = []

    def __call__(self, descriptions, device):
        self.calls.append((descriptions, device))
        batch_size = len(descriptions)
        hidden_states = torch.randn(batch_size, self.seq_len, self.hidden_size, device=device)
        mask = torch.ones(batch_size, self.seq_len, dtype=torch.long, device=device)
        return hidden_states, mask


@pytest.fixture
def tokenizer() -> SMILESTokenizer:
    return SMILESTokenizer().build_vocab(SMILES)


@pytest.fixture
def one_plain_column(monkeypatch):
    # compute_property_stats (called by SMILESDataset) looks up PROPERTY_COLUMNS
    # by index to find one-hot columns, so tests using a 1-column property
    # tensor must monkeypatch it to match -- same fixture as test_data.py.
    import linkergen.data as data_module

    monkeypatch.setattr(data_module, "PROPERTY_COLUMNS", ["colA"])


def make_model(vocab_size, condition_type="none", **kwargs):
    torch.manual_seed(0)
    return LinkerGen(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        d_hidden=D_HIDDEN,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=32,
        condition_type=condition_type,
        dropout=0.0,
        **kwargs,
    )


def make_loader(tokenizer, condition_type, conditions=None, batch_size=2):
    return create_dataloader(
        SMILES,
        tokenizer,
        condition_type=condition_type,
        conditions=conditions,
        batch_size=batch_size,
        shuffle=False,
    )


# --- prepare_condition -----------------------------------------------------

def test_prepare_condition_returns_none_without_condition_key():
    assert prepare_condition({}, "none", torch.device("cpu")) is None
    assert prepare_condition({}, "property", torch.device("cpu")) is None


def test_prepare_condition_property_mode_moves_tensor_to_device():
    condition = torch.tensor([[1.0, 2.0]])
    out = prepare_condition({"condition": condition}, "property", torch.device("cpu"))
    assert torch.equal(out, condition)
    assert out.device == torch.device("cpu")


def test_prepare_condition_text_mode_calls_nach0_encoder():
    fake = FakeNach0Encoder(hidden_size=10, seq_len=5)
    device = torch.device("cpu")
    descriptions = ["a linker", "another linker"]

    hidden_states, mask = prepare_condition({"condition": descriptions}, "text", device, fake)

    # one nach0 call, with the raw descriptions and device passed straight through
    assert fake.calls == [(descriptions, device)]
    assert hidden_states.shape == (2, fake.seq_len, fake.hidden_size)
    assert mask.shape == (2, fake.seq_len)


# --- train_epoch -------------------------------------------------------------

def test_train_epoch_none_mode_returns_finite_loss(tokenizer):
    model = make_model(tokenizer.vocab_size, "none")
    loader = make_loader(tokenizer, "none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    loss = train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    assert math.isfinite(loss)


def test_train_epoch_updates_model_weights(tokenizer, one_plain_column):
    model = make_model(tokenizer.vocab_size, "property", property_dim=1, n_memory=2)
    conditions = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    loader = make_loader(tokenizer, "property", conditions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    before = [p.clone() for p in model.parameters()]
    train_epoch(model, loader, optimizer, criterion, torch.device("cpu"), condition_type="property")
    after = list(model.parameters())

    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_train_epoch_text_mode_uses_one_nach0_call_per_batch(tokenizer):
    fake = FakeNach0Encoder(hidden_size=10, seq_len=5)
    descriptions = [f"description {i}" for i in range(len(SMILES))]
    model = make_model(tokenizer.vocab_size, "text", text_hidden_size=fake.hidden_size)
    loader = make_loader(tokenizer, "text", descriptions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    loss = train_epoch(
        model, loader, optimizer, criterion, torch.device("cpu"),
        condition_type="text", nach0_encoder=fake,
    )

    assert math.isfinite(loss)
    # same frequency as the property path's condition tensor: once per batch,
    # not once per sample or per decoding step.
    assert len(fake.calls) == len(loader)


def test_train_epoch_drops_condition_when_rng_triggers(tokenizer, one_plain_column, monkeypatch):
    model = make_model(tokenizer.vocab_size, "property", property_dim=1, n_memory=2)
    conditions = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    loader = make_loader(tokenizer, "property", conditions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    seen_conditions = []
    original_forward = LinkerGen.forward

    def spy_forward(self, x, condition=None):
        seen_conditions.append(condition)
        return original_forward(self, x, condition)

    monkeypatch.setattr(LinkerGen, "forward", spy_forward)
    monkeypatch.setattr(random, "random", lambda: 0.0)  # always below any positive cond_drop_prob

    train_epoch(
        model, loader, optimizer, criterion, torch.device("cpu"),
        condition_type="property", cond_drop_prob=0.5,
    )

    assert seen_conditions and all(c is None for c in seen_conditions)


def test_train_epoch_keeps_condition_when_cond_drop_prob_is_zero(tokenizer, one_plain_column, monkeypatch):
    model = make_model(tokenizer.vocab_size, "property", property_dim=1, n_memory=2)
    conditions = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    loader = make_loader(tokenizer, "property", conditions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    seen_conditions = []
    original_forward = LinkerGen.forward

    def spy_forward(self, x, condition=None):
        seen_conditions.append(condition)
        return original_forward(self, x, condition)

    monkeypatch.setattr(LinkerGen, "forward", spy_forward)
    monkeypatch.setattr(random, "random", lambda: 0.0)  # would trigger a drop if cond_drop_prob > 0

    train_epoch(
        model, loader, optimizer, criterion, torch.device("cpu"),
        condition_type="property", cond_drop_prob=0.0,
    )

    assert seen_conditions and all(c is not None for c in seen_conditions)


# --- validate -----------------------------------------------------------------

def test_validate_returns_finite_loss_and_does_not_update_weights(tokenizer):
    model = make_model(tokenizer.vocab_size, "none")
    loader = make_loader(tokenizer, "none")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    before = [p.clone() for p in model.parameters()]
    loss = validate(model, loader, criterion, torch.device("cpu"))
    after = list(model.parameters())

    assert math.isfinite(loss)
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_validate_never_touches_the_dropout_rng(tokenizer, one_plain_column, monkeypatch):
    # validate() has no cond_drop_prob parameter at all -- CFG dropout is a
    # training-only concern (validation is always fully conditioned). Prove
    # it by making random.random() explode if validate() ever calls it.
    import linkergen.train as train_module

    def explode():
        raise AssertionError("validate() must never sample the CFG dropout RNG")

    monkeypatch.setattr(train_module.random, "random", explode)

    model = make_model(tokenizer.vocab_size, "property", property_dim=1, n_memory=2)
    conditions = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    loader = make_loader(tokenizer, "property", conditions)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    validate(model, loader, criterion, torch.device("cpu"), condition_type="property")


# --- real Nach0Encoder integration (network) ----------------------------------

@pytest.mark.network
def test_train_epoch_text_mode_with_real_nach0_encoder(tokenizer):
    from linkergen.nach0 import Nach0Encoder

    nach0_encoder = Nach0Encoder()
    descriptions = [f"a linker described in words {i}" for i in range(len(SMILES))]
    model = make_model(tokenizer.vocab_size, "text", text_hidden_size=nach0_encoder.hidden_size)
    loader = make_loader(tokenizer, "text", descriptions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    loss = train_epoch(
        model, loader, optimizer, criterion, torch.device("cpu"),
        condition_type="text", nach0_encoder=nach0_encoder,
    )

    assert math.isfinite(loss)
