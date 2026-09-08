"""Tests for Nach0Encoder.

These need the real `insilicomedicine/nach0_base` weights (downloaded from
Hugging Face on first use), so the whole module is marked `network` and
skipped by default (see pyproject.toml's `addopts`). Run explicitly with:

    pytest -m network tests/test_nach0.py

Text-mode wiring elsewhere (model.py, and later train.py/sample.py) is
covered separately with a mocked (hidden_states, mask) pair, so that
coverage doesn't depend on network access.
"""

import pytest
import torch

pytestmark = pytest.mark.network

nach0 = pytest.importorskip("linkergen.nach0")


@pytest.fixture(scope="module")
def encoder():
    return nach0.Nach0Encoder()


def test_output_shapes_match_batch_and_sequence_length(encoder):
    descriptions = ["a rigid linear linker", "a flexible linker with an ether bridge"]
    hidden_states, attention_mask = encoder(descriptions, device=torch.device("cpu"))

    batch_size, seq_len, hidden_size = hidden_states.shape
    assert batch_size == len(descriptions)
    assert hidden_size == encoder.hidden_size
    assert attention_mask.shape == (batch_size, seq_len)


def test_encoder_parameters_are_frozen(encoder):
    assert all(not p.requires_grad for p in encoder.encoder.parameters())


def test_output_carries_no_grad(encoder):
    hidden_states, _ = encoder(["a short linker"], device=torch.device("cpu"))
    assert not hidden_states.requires_grad


def test_attention_mask_reflects_padding(encoder):
    # A short and a much longer description, batched together, should
    # produce a padding mask that is all-real for the long one and
    # partially padded for the short one.
    short, long = "short", "a considerably longer molecular linker description"
    _, attention_mask = encoder([short, long], device=torch.device("cpu"))

    short_real = attention_mask[0].sum().item()
    long_real = attention_mask[1].sum().item()
    assert short_real < attention_mask.shape[1]
    assert long_real <= attention_mask.shape[1]
    assert short_real < long_real
