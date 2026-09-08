"""Unit tests for LinkerGen and its three conditioning modes.

Mirrors the spec in plan.md Phase 5: constructor validation per
condition_type, forward shapes, end-to-end causal masking, text-mode
padding-mask correctness, and the generate() batched early-stopping loop.
"""

import ast
import inspect

import pytest
import torch

from linkergen.model import LinkerGen

VOCAB_SIZE = 6
D_MODEL = 16
N_HEADS = 2
N_LAYERS = 2
D_HIDDEN = 32


def make_model(condition_type="none", **kwargs):
    torch.manual_seed(0)
    model = LinkerGen(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_hidden=D_HIDDEN,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=32,
        condition_type=condition_type,
        dropout=0.0,
        **kwargs,
    )
    return model.eval()


# --- construction -----------------------------------------------------

def test_rejects_unknown_condition_type():
    with pytest.raises(ValueError):
        LinkerGen(vocab_size=VOCAB_SIZE, condition_type="bogus")


def test_property_mode_requires_property_dim():
    with pytest.raises(ValueError):
        LinkerGen(vocab_size=VOCAB_SIZE, condition_type="property")


def test_text_mode_requires_text_hidden_size():
    with pytest.raises(ValueError):
        LinkerGen(vocab_size=VOCAB_SIZE, condition_type="text")


def test_model_module_does_not_import_transformers_or_nach0():
    # Guards the design decision that nach0 stays out of the model: LinkerGen
    # only ever consumes precomputed (hidden_states, mask) tuples.
    import linkergen.model as model_module

    tree = ast.parse(inspect.getsource(model_module))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert "transformers" not in imported_roots
    assert "nach0" not in imported_roots


# --- forward shapes -----------------------------------------------------

def test_none_mode_forward_shape():
    model = make_model("none")
    x = torch.randint(0, VOCAB_SIZE, (2, 7))
    logits = model(x)
    assert logits.shape == (2, 7, VOCAB_SIZE)


def test_property_mode_forward_shape():
    model = make_model("property", property_dim=4, n_memory=3)
    x = torch.randint(0, VOCAB_SIZE, (2, 7))
    condition = torch.randn(2, 4)
    logits = model(x, condition=condition)
    assert logits.shape == (2, 7, VOCAB_SIZE)


def test_property_mode_condition_none_does_not_crash():
    # Classifier-free-guidance dropout drops the condition batch-wise at
    # train time; forward must still produce a valid (unconditional) pass.
    model = make_model("property", property_dim=4, n_memory=3)
    x = torch.randint(0, VOCAB_SIZE, (2, 7))
    logits = model(x, condition=None)
    assert logits.shape == (2, 7, VOCAB_SIZE)


def test_text_mode_forward_shape():
    model = make_model("text", text_hidden_size=10)
    x = torch.randint(0, VOCAB_SIZE, (2, 7))
    hidden_states = torch.randn(2, 5, 10)
    mask = torch.ones(2, 5, dtype=torch.long)
    logits = model(x, condition=(hidden_states, mask))
    assert logits.shape == (2, 7, VOCAB_SIZE)


# --- masking correctness -------------------------------------------------

def test_causal_masking_end_to_end():
    model = make_model("none")
    torch.manual_seed(7)
    x = torch.randint(0, VOCAB_SIZE, (2, 6))
    x_perturbed = x.clone()
    x_perturbed[:, -1] = (x_perturbed[:, -1] + 1) % VOCAB_SIZE

    with torch.no_grad():
        out_base = model(x)
        out_perturbed = model(x_perturbed)

    # Perturbing the last token must not change logits at earlier positions,
    # through the full stack (multiple layers, cross-attention-free here).
    assert torch.equal(out_base[:, :-1, :], out_perturbed[:, :-1, :])


def test_text_mode_ignores_padded_description_tokens():
    model = make_model("text", text_hidden_size=10)
    torch.manual_seed(5)
    x = torch.randint(0, VOCAB_SIZE, (2, 7))
    hidden_states = torch.randn(2, 5, 10)
    mask = torch.ones(2, 5, dtype=torch.long)
    mask[:, -1] = 0  # last description token is padding

    hidden_states_perturbed = hidden_states.clone()
    hidden_states_perturbed[:, -1, :] += 10.0

    with torch.no_grad():
        out_base = model(x, condition=(hidden_states, mask))
        out_perturbed = model(x, condition=(hidden_states_perturbed, mask))

    assert torch.equal(out_base, out_perturbed)


# --- generate() -----------------------------------------------------------

def test_generate_respects_max_new_tokens_without_eos():
    model = make_model("none")
    input_ids = torch.ones(3, 1, dtype=torch.long)
    generated = model.generate(
        input_ids, max_new_tokens=5, temperature=0.8, eos_token_id=None, pad_token_id=None
    )
    assert generated.shape == (3, 1 + 5)


def test_generate_freezes_finished_sequences(monkeypatch):
    # Stub forward() so the eos/pad-freeze bookkeeping in generate() is
    # exercised deterministically, independent of the (randomly-initialized)
    # model's actual token predictions.
    eos_token_id, pad_token_id, other_token_id = 2, 0, 3
    prompt_len = 1

    def fake_forward(self, x, condition=None):
        batch_size, seq_len = x.shape
        logits = torch.zeros(batch_size, seq_len, VOCAB_SIZE)
        for b in range(batch_size):
            # Row 0 emits EOS on the first generated step, then (if the
            # freeze logic were broken) would keep emitting other_token_id;
            # row 1 never emits EOS within max_new_tokens.
            emits_eos = b == 0 and seq_len == prompt_len
            logits[b, -1, eos_token_id if emits_eos else other_token_id] = 100.0
        return logits

    monkeypatch.setattr(LinkerGen, "forward", fake_forward)
    model = make_model("none")

    input_ids = torch.ones(2, prompt_len, dtype=torch.long)
    generated = model.generate(
        input_ids,
        max_new_tokens=4,
        temperature=0.0,  # argmax decoding, so the stub above is deterministic
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    new_tokens = generated[:, prompt_len:]
    assert new_tokens[0].tolist() == [eos_token_id, pad_token_id, pad_token_id, pad_token_id]
    assert new_tokens[1].tolist() == [other_token_id] * 4
