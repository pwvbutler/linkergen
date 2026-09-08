"""Unit tests for SMILESTokenizer.

Covers the regex-based tokenization, vocab construction, encode/decode
round-tripping, and the on-disk save/load format that checkpoints rely on
(see CLAUDE.md: "tokenizer_vocab ... must be rebuilt from that on load").
"""

import json

import pytest

from linkergen.tokenizer import SMILESTokenizer

# A handful of SMILES exercising the tokenizer's distinct regex branches:
# bracket atoms, two-letter halogens, aromatic lowercase, ring-bond %nn,
# charges/stereo/bond symbols.
SAMPLE_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "ClCCBr",
    "[Na+].[Cl-]",
    "C1CC%10CCC%10C1",
    "C/C=C/C",
    "C#N",
]


@pytest.fixture
def tokenizer() -> SMILESTokenizer:
    return SMILESTokenizer().build_vocab(SAMPLE_SMILES)


def test_special_tokens_at_fixed_indices(tokenizer):
    # Special tokens must stay at indices 0-3 regardless of corpus content,
    # since sampling code and saved checkpoints assume this fixed layout.
    assert tokenizer.token2idx[SMILESTokenizer.PAD_TOKEN] == 0
    assert tokenizer.token2idx[SMILESTokenizer.BOS_TOKEN] == 1
    assert tokenizer.token2idx[SMILESTokenizer.EOS_TOKEN] == 2
    assert tokenizer.token2idx[SMILESTokenizer.UNK_TOKEN] == 3
    assert tokenizer.pad_idx == 0
    assert tokenizer.bos_idx == 1
    assert tokenizer.eos_idx == 2
    assert tokenizer.unk_idx == 3


def test_vocab_size_accounts_for_specials_and_corpus(tokenizer):
    distinct_chem_tokens = {t for smi in SAMPLE_SMILES for t in tokenizer.tokenize(smi)}
    assert tokenizer.vocab_size == len(distinct_chem_tokens) + 4


@pytest.mark.parametrize("smiles", SAMPLE_SMILES)
def test_encode_decode_round_trip(tokenizer, smiles):
    ids = tokenizer.encode(smiles)
    # BOS/EOS bracket the sequence by default.
    assert ids[0] == tokenizer.bos_idx
    assert ids[-1] == tokenizer.eos_idx
    assert tokenizer.decode(ids) == smiles


def test_encode_without_special_tokens(tokenizer):
    ids = tokenizer.encode("CCO", add_special_tokens=False)
    assert tokenizer.bos_idx not in ids
    assert tokenizer.eos_idx not in ids
    assert tokenizer.decode(ids, remove_special_tokens=False) == "CCO"


def test_decode_keeps_special_tokens_when_requested(tokenizer):
    ids = tokenizer.encode("CCO")
    decoded = tokenizer.decode(ids, remove_special_tokens=False)
    assert decoded.startswith(SMILESTokenizer.BOS_TOKEN)
    assert decoded.endswith(SMILESTokenizer.EOS_TOKEN)


def test_unknown_token_maps_to_unk(tokenizer):
    # "I" (iodine) is not in the training corpus above, so it must fall
    # back to <UNK> rather than raising a KeyError.
    ids = tokenizer.encode("I", add_special_tokens=False)
    assert ids == [tokenizer.unk_idx]


def test_build_vocab_is_deterministic_across_instances():
    tok_a = SMILESTokenizer().build_vocab(SAMPLE_SMILES)
    tok_b = SMILESTokenizer().build_vocab(list(reversed(SAMPLE_SMILES)))
    assert tok_a.token2idx == tok_b.token2idx


def test_save_load_round_trip(tokenizer, tmp_path):
    path = tmp_path / "vocab.json"
    tokenizer.save(str(path))

    # The saved file must be a plain token -> index JSON mapping, since
    # this is the format persisted into a checkpoint's "tokenizer_vocab".
    with open(path) as f:
        raw = json.load(f)
    assert raw == tokenizer.token2idx

    reloaded = SMILESTokenizer.load(str(path))
    assert reloaded.token2idx == tokenizer.token2idx
    assert reloaded.idx2token == tokenizer.idx2token
    assert reloaded.vocab_size == tokenizer.vocab_size

    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    assert reloaded.decode(reloaded.encode(smiles)) == smiles
