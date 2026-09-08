"""Unit tests for MultiHeadAttention.

Beyond shape checks, the causal and key_padding_mask tests are black-box
correctness checks: they perturb an input position that should be masked
out and assert the output is unaffected, which would catch a masking bug
(e.g. an off-by-one, or the additive mask built with the wrong polarity)
that a shape-only test would miss.
"""

import torch

from linkergen.attention import MultiHeadAttention

D_MODEL = 16
N_HEADS = 4


def make_attn(dropout=0.0):
    torch.manual_seed(0)
    return MultiHeadAttention(D_MODEL, N_HEADS, dropout=dropout).eval()


def test_self_attention_output_shape():
    attn = make_attn()
    x = torch.randn(2, 5, D_MODEL)
    out = attn(x, x, x)
    assert out.shape == (2, 5, D_MODEL)


def test_cross_attention_output_shape_follows_query_length():
    attn = make_attn()
    query = torch.randn(2, 5, D_MODEL)  # decoder side, seq len 5
    memory = torch.randn(2, 7, D_MODEL)  # conditioning memory, seq len 7
    out = attn(query, memory, memory)
    # Output sequence length tracks the query, not the key/value.
    assert out.shape == (2, 5, D_MODEL)


def test_eval_mode_is_deterministic():
    attn = make_attn(dropout=0.5)  # high dropout, but disabled in eval()
    x = torch.randn(2, 5, D_MODEL)
    out1 = attn(x, x, x)
    out2 = attn(x, x, x)
    assert torch.equal(out1, out2)


def test_causal_mask_blocks_future_positions():
    attn = make_attn()
    torch.manual_seed(1)
    x = torch.randn(2, 6, D_MODEL)

    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 10.0  # perturb only the last (future-most) token

    with torch.no_grad():
        out_base = attn(x, x, x, is_causal=True)
        out_perturbed = attn(x_perturbed, x_perturbed, x_perturbed, is_causal=True)

    # Every position except the perturbed one must be unaffected, since a
    # causal mask forbids attending to a later position.
    assert torch.equal(out_base[:, :-1, :], out_perturbed[:, :-1, :])
    # Sanity check the test itself has teeth: the perturbed position *can*
    # attend to itself, so it should actually change.
    assert not torch.equal(out_base[:, -1, :], out_perturbed[:, -1, :])


def test_key_padding_mask_ignores_padded_memory_positions():
    attn = make_attn()
    torch.manual_seed(2)
    query = torch.randn(2, 4, D_MODEL)
    memory = torch.randn(2, 7, D_MODEL)

    # Mark the last memory position as padding; the rest are real tokens.
    key_padding_mask = torch.ones(2, 7, dtype=torch.bool)
    key_padding_mask[:, -1] = False

    memory_perturbed = memory.clone()
    memory_perturbed[:, -1, :] += 10.0  # perturb only the padded position

    with torch.no_grad():
        out_base = attn(query, memory, memory, key_padding_mask=key_padding_mask)
        out_perturbed = attn(
            query, memory_perturbed, memory_perturbed, key_padding_mask=key_padding_mask
        )

    # A fully-masked key position must contribute nothing, at any query
    # position, regardless of what its (unused) value vector holds.
    assert torch.equal(out_base, out_perturbed)


def test_key_padding_mask_none_is_equivalent_to_all_real():
    attn = make_attn()
    torch.manual_seed(3)
    query = torch.randn(2, 4, D_MODEL)
    memory = torch.randn(2, 7, D_MODEL)
    all_real_mask = torch.ones(2, 7, dtype=torch.bool)

    with torch.no_grad():
        out_no_mask = attn(query, memory, memory, key_padding_mask=None)
        out_all_real = attn(query, memory, memory, key_padding_mask=all_real_mask)

    assert torch.allclose(out_no_mask, out_all_real, atol=1e-6)
