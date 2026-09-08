"""Unit tests for the sample module.

Mirrors the spec in plan.md Phase 8: covers build_condition, the one piece of
sample.py that is a testable unit in isolation from the CLI entrypoint.
main() itself (argparse + checkpoint reconstruction + generation + RDKit
reporting) is a CLI entrypoint rather than a unit and is left to Phase 9's
manual smoke tests, per plan -- same scope decision made for train.py's
main() in test_train.py.

Text-mode tests use a FakeNach0Encoder (a stand-in with the same
model_name -> descriptions/device -> (hidden_states, mask) call signature as
the real Nach0Encoder) so this file stays fast and network-free. One
additional test exercises the real Nach0Encoder end to end and is marked
`network` (skipped by default, see pyproject.toml), matching test_nach0.py's
and test_train.py's convention.
"""

from types import SimpleNamespace

import pytest
import torch

import linkergen.sample as sample_module
from linkergen.sample import build_condition


class FakeNach0Encoder:
    """Stands in for the real Nach0Encoder in tests that exercise
    build_condition's wiring without needing a network download of the
    actual nach0 model. Records every constructed instance and every call
    on it so tests can check both."""

    hidden_size = 10
    seq_len = 5

    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = []
        FakeNach0Encoder.instances.append(self)

    def to(self, device):
        return self

    def __call__(self, descriptions, device):
        self.calls.append((descriptions, device))
        batch_size = len(descriptions)
        hidden_states = torch.randn(batch_size, self.seq_len, self.hidden_size, device=device)
        mask = torch.ones(batch_size, self.seq_len, dtype=torch.long, device=device)
        return hidden_states, mask


@pytest.fixture
def fake_nach0(monkeypatch):
    FakeNach0Encoder.instances = []
    monkeypatch.setattr(sample_module, "Nach0Encoder", FakeNach0Encoder)
    return FakeNach0Encoder


# --- "none" mode -------------------------------------------------------------

def test_none_mode_returns_none_when_no_flags_set():
    args = SimpleNamespace(properties=None, text_description=None, num_samples=1)
    condition = build_condition(args, {"condition_type": "none"}, {}, torch.device("cpu"))
    assert condition is None


def test_none_mode_rejects_properties():
    args = SimpleNamespace(properties=[1.0], text_description=None, num_samples=1)
    with pytest.raises(ValueError):
        build_condition(args, {"condition_type": "none"}, {}, torch.device("cpu"))


def test_none_mode_rejects_text_description():
    args = SimpleNamespace(properties=None, text_description="a linker", num_samples=1)
    with pytest.raises(ValueError):
        build_condition(args, {"condition_type": "none"}, {}, torch.device("cpu"))


# --- "property" mode -----------------------------------------------------------

def test_property_mode_rejects_text_description():
    args = SimpleNamespace(properties=[1.0], text_description="a linker", num_samples=1)
    ckpt = {"condition_type": "property", "property_columns": ["colA"]}
    with pytest.raises(ValueError):
        build_condition(args, ckpt, {"property_dim": 1}, torch.device("cpu"))


def test_property_mode_requires_properties():
    args = SimpleNamespace(properties=None, text_description=None, num_samples=1)
    ckpt = {"condition_type": "property", "property_columns": ["colA"]}
    with pytest.raises(ValueError):
        build_condition(args, ckpt, {"property_dim": 1}, torch.device("cpu"))


def test_property_mode_rejects_wrong_length():
    args = SimpleNamespace(properties=[1.0, 2.0], text_description=None, num_samples=1)
    ckpt = {"condition_type": "property", "property_columns": ["colA"]}
    with pytest.raises(ValueError):
        build_condition(args, ckpt, {"property_dim": 1}, torch.device("cpu"))


def test_property_mode_standardizes_and_repeats_across_batch():
    args = SimpleNamespace(properties=[10.0, 20.0], text_description=None, num_samples=3)
    ckpt = {
        "condition_type": "property",
        "property_columns": ["colA", "colB"],
        "prop_mean": torch.tensor([5.0, 15.0]),
        "prop_std": torch.tensor([2.0, 5.0]),
    }
    config = {"property_dim": 2}

    condition = build_condition(args, ckpt, config, torch.device("cpu"))

    expected_row = (torch.tensor([10.0, 20.0]) - ckpt["prop_mean"]) / ckpt["prop_std"]
    assert condition.shape == (3, 2)
    assert torch.allclose(condition, expected_row.repeat(3, 1))


def test_property_mode_guards_against_zero_std():
    args = SimpleNamespace(properties=[10.0], text_description=None, num_samples=2)
    ckpt = {
        "condition_type": "property",
        "property_columns": ["colA"],
        "prop_mean": torch.tensor([10.0]),
        "prop_std": torch.tensor([0.0]),
    }
    config = {"property_dim": 1}

    condition = build_condition(args, ckpt, config, torch.device("cpu"))

    # std of 0 is treated as 1, so (10 - 10) / 1 == 0 -- not a divide-by-zero NaN/inf
    assert torch.equal(condition, torch.zeros(2, 1))


# --- "text" mode ---------------------------------------------------------------

def test_text_mode_rejects_properties():
    args = SimpleNamespace(properties=[1.0], text_description=None, num_samples=1)
    ckpt = {"condition_type": "text", "nach0_model_name": "fake-nach0"}
    with pytest.raises(ValueError):
        build_condition(args, ckpt, {}, torch.device("cpu"))


def test_text_mode_requires_text_description():
    args = SimpleNamespace(properties=None, text_description=None, num_samples=1)
    ckpt = {"condition_type": "text", "nach0_model_name": "fake-nach0"}
    with pytest.raises(ValueError):
        build_condition(args, ckpt, {}, torch.device("cpu"))


def test_text_mode_encodes_once_and_expands_across_batch(fake_nach0):
    args = SimpleNamespace(properties=None, text_description="a rigid linear linker", num_samples=4)
    ckpt = {"condition_type": "text", "nach0_model_name": "fake-nach0"}
    device = torch.device("cpu")

    hidden_states, mask = build_condition(args, ckpt, {}, device)

    # constructed from the checkpoint's nach0_model_name, encoded exactly once,
    # with the description wrapped in a single-element list (batch of one)
    assert len(fake_nach0.instances) == 1
    encoder = fake_nach0.instances[0]
    assert encoder.model_name == "fake-nach0"
    assert encoder.calls == [(["a rigid linear linker"], device)]

    assert hidden_states.shape == (4, fake_nach0.seq_len, fake_nach0.hidden_size)
    assert mask.shape == (4, fake_nach0.seq_len)
    # expand() broadcasts the single encoded description across the whole batch
    assert torch.equal(hidden_states[0], hidden_states[1])
    assert torch.equal(mask[0], mask[1])


# --- real Nach0Encoder integration (network) ------------------------------------

@pytest.mark.network
def test_text_mode_with_real_nach0_encoder():
    args = SimpleNamespace(
        properties=None,
        text_description="a rigid linear linker with two aromatic rings",
        num_samples=3,
    )
    ckpt = {"condition_type": "text", "nach0_model_name": "insilicomedicine/nach0_base"}
    device = torch.device("cpu")

    hidden_states, mask = build_condition(args, ckpt, {}, device)

    assert hidden_states.shape[0] == 3
    assert mask.shape[0] == 3
    assert torch.equal(hidden_states[0], hidden_states[1])
