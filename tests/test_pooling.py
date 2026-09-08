"""Unit tests for AttentionPoolingLayer.

Mirrors test_attention.py's approach: beyond shape checks, the masking test
is a black-box correctness check that perturbs a masked-out key/value
position and asserts the pooled output is unaffected, which would catch a
masking bug that a shape-only test would miss.
"""

import torch

from linkergen.model import AttentionPoolingLayer

D_MODEL = 16
N_HEADS = 4


def make_pool(n_queries=1, dropout=0.0, learn_mean_correction=True):
    torch.manual_seed(0)
    return AttentionPoolingLayer(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        dropout=dropout,
        n_queries=n_queries,
        learn_mean_correction=learn_mean_correction,
    ).eval()


def test_output_shape_pools_sequence_to_single_query():
    pool = make_pool(n_queries=1)
    x = torch.randn(2, 9, D_MODEL)
    out = pool(x)
    assert out.shape == (2, 1, D_MODEL)


def test_output_shape_follows_n_queries():
    pool = make_pool(n_queries=3)
    x = torch.randn(2, 9, D_MODEL)
    out = pool(x)
    assert out.shape == (2, 3, D_MODEL)


def test_eval_mode_is_deterministic():
    pool = make_pool(dropout=0.5)  # high dropout, but disabled in eval()
    x = torch.randn(2, 9, D_MODEL)
    out1 = pool(x)
    out2 = pool(x)
    assert torch.equal(out1, out2)


def test_memory_mask_ignores_padded_positions():
    pool = make_pool(n_queries=2)
    torch.manual_seed(2)
    x = torch.randn(2, 7, D_MODEL)

    # Mark the last position as padding; the rest are real tokens.
    memory_mask = torch.ones(2, 7, dtype=torch.bool)
    memory_mask[:, -1] = False

    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 10.0  # perturb only the padded position

    with torch.no_grad():
        out_base = pool(x, memory_mask=memory_mask)
        out_perturbed = pool(x_perturbed, memory_mask=memory_mask)

    # A fully-masked position must contribute nothing -- neither through the
    # cross-attention nor through the learn_mean_correction residual, both of
    # which respect memory_mask.
    assert torch.equal(out_base, out_perturbed)


def test_empty_mask_row_does_not_produce_nan():
    # A row whose mask is all-False (e.g. a batch element with zero real
    # tokens) must fall back to the empty-mask safeguard rather than divide
    # by zero in softmax.
    pool = make_pool(n_queries=2)
    x = torch.randn(2, 5, D_MODEL)
    memory_mask = torch.ones(2, 5, dtype=torch.bool)
    memory_mask[0] = False  # first batch row has no real positions at all

    out = pool(x, memory_mask=memory_mask)
    assert not torch.isnan(out).any()


def test_learn_mean_correction_changes_output():
    pool = make_pool(n_queries=1, learn_mean_correction=True)
    x = torch.randn(2, 9, D_MODEL)

    with torch.no_grad():
        out_with_correction = pool(x)
        pool.learn_mean_correction = False
        out_without_correction = pool(x)

    assert not torch.equal(out_with_correction, out_without_correction)
