"""
Unified sampling entrypoint for LinkerGen.

Reconstructs a LinkerGen model, and for text-conditioned checkpoints a
Nach0Encoder, entirely from a checkpoint saved by train.py, then runs
batched autoregressive generation and reports RDKit validity.

  1) Unconditional checkpoint:
        linkergen-sample --checkpoint checkpoints/pretrain/best_model.pt \\
            --num_samples 100

  2) Property-conditioned checkpoint (raw, un-standardized values, in the
     checkpoint's property_columns order; the script standardizes them
     internally using the prop_mean/prop_std saved at train time):
        linkergen-sample --checkpoint checkpoints/finetune/best_model.pt \\
            --properties 350.0 2.5 1 2 --num_samples 100

  3) Text-conditioned checkpoint:
        linkergen-sample --checkpoint checkpoints/finetune_text/best_model.pt \\
            --text_description "a rigid linear linker with two aromatic rings" \\
            --num_samples 100

--properties/--text_description must match the checkpoint's condition_type:
passing the wrong one, or omitting the one it requires, is a ValueError rather
than a silently ignored flag.
"""

import argparse
from pathlib import Path

import torch
from rdkit import Chem

from .model import LinkerGen
from .nach0 import Nach0Encoder
from .tokenizer import SMILESTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling script for LinkerGen")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of SMILES to generate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max length of generated SMILES")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--output_file", type=str, default="generated_smiles.txt", help="File to save outputs")
    parser.add_argument(
        "--properties",
        type=float,
        nargs="+",
        default=None,
        help="--condition property checkpoints only: raw target values, in the "
             "checkpoint's property_columns order (e.g. 350.0 2.5 1 2).",
    )
    parser.add_argument(
        "--text_description",
        type=str,
        default=None,
        help="--condition text checkpoints only: natural-language description, "
             "encoded once via the checkpoint's nach0 model and reused unchanged "
             "across the whole sampling batch.",
    )
    return parser.parse_args()


def build_condition(args, ckpt, config, device):
    """
    Build the `condition` argument for LinkerGen.generate(), matching the
    checkpoint's condition_type. Raises ValueError if the conditioning flags
    given on the command line don't match what this checkpoint expects.
    """
    condition_type = ckpt["condition_type"]

    if condition_type == "none":
        if args.properties is not None or args.text_description is not None:
            raise ValueError(
                "This checkpoint is unconditional (--condition none); "
                "--properties/--text_description are not accepted."
            )
        return None

    if condition_type == "property":
        if args.text_description is not None:
            raise ValueError("This checkpoint is property-conditioned; --text_description is not accepted.")
        if args.properties is None:
            raise ValueError(
                f"This checkpoint requires --properties matching "
                f"{ckpt['property_columns']} ({config['property_dim']} values)."
            )
        if len(args.properties) != config["property_dim"]:
            raise ValueError(
                f"--properties has {len(args.properties)} values, expected "
                f"{config['property_dim']} ({ckpt['property_columns']})."
            )

        # Standardize using the training statistics preserved in the checkpoint:
        # sampling must apply the same transform applied at train time.
        raw = torch.tensor(args.properties, dtype=torch.float).unsqueeze(0)  # (1, property_dim)
        prop_mean, prop_std = ckpt["prop_mean"], ckpt["prop_std"]
        prop_std = torch.where(prop_std == 0, torch.ones_like(prop_std), prop_std)
        standardized = (raw - prop_mean) / prop_std
        return standardized.repeat(args.num_samples, 1).to(device)

    # condition_type == "text"
    if args.properties is not None:
        raise ValueError("This checkpoint is text-conditioned; --properties is not accepted.")
    if args.text_description is None:
        raise ValueError("This checkpoint requires --text_description.")

    # Encode the description once, then reuse it unchanged across the whole
    # sampling batch and every decoding step (see LinkerGen.generate).
    nach0_encoder = Nach0Encoder(ckpt["nach0_model_name"]).to(device)
    hidden_states, mask = nach0_encoder([args.text_description], device)
    hidden_states = hidden_states.expand(args.num_samples, -1, -1)
    mask = mask.expand(args.num_samples, -1)
    return hidden_states, mask


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the checkpoint and reconstruct the tokenizer. Vocab is
    # data-dependent (see tokenizer.py), so it is rebuilt from the
    # checkpoint's saved vocab, never from a fresh corpus scan.
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    condition_type = ckpt["condition_type"]
    print(f"Checkpoint condition_type: {condition_type}")

    tokenizer = SMILESTokenizer()
    tokenizer.token2idx = ckpt["tokenizer_vocab"]
    tokenizer.idx2token = {v: k for k, v in tokenizer.token2idx.items()}

    # Reconstruct LinkerGen, mirroring train.py's construction for this mode.
    model = LinkerGen(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        d_hidden=config["d_hidden"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_seq_len=config["max_length"],
        condition_type=condition_type,
        property_dim=config["property_dim"] if condition_type == "property" else None,
        n_memory=config["n_memory"],
        text_hidden_size=ckpt["text_hidden_size"] if condition_type == "text" else None,
        n_pool_queries=ckpt["n_pool_queries"] if condition_type == "text" else 1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    condition = build_condition(args, ckpt, config, device)

    print(f"\nGenerating {args.num_samples} samples (temp={args.temperature})...")

    input_ids = torch.full(
        (args.num_samples, 1),
        tokenizer.bos_idx,
        dtype=torch.long,
        device=device,
    )

    generated_tokens = model.generate(
        input_ids=input_ids,
        condition=condition,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        eos_token_id=tokenizer.eos_idx,
        pad_token_id=tokenizer.pad_idx,
    )

    # Decode, check RDKit validity, report, and save
    valid_count = 0
    results = []

    print("\nEvaluating molecular structures:")
    for i, tokens in enumerate(generated_tokens):
        smiles = tokenizer.decode(tokens.tolist(), remove_special_tokens=True)
        # strip trailing content occasionally left by loose generations
        smiles = smiles.split()[0] if smiles.strip() else ""

        mol = Chem.MolFromSmiles(smiles) if smiles else None
        is_valid = mol is not None
        if is_valid:
            valid_count += 1

        status = "VALID" if is_valid else "INVALID"
        print(f"[{i + 1:03d}] ({status}, len={len(smiles)}): {smiles}")
        results.append(smiles)

    validity_rate = valid_count / args.num_samples
    print("\n" + "=" * 50)
    print("Generation Statistics:")
    print(f"  Total Extracted:     {args.num_samples}")
    print(f"  Chemically Valid:    {valid_count}")
    print(f"  Overall Validity:    {validity_rate * 100:.2f}%")
    print("=" * 50)

    output_path = Path(args.output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        for smiles in results:
            f.write(f"{smiles}\n")
    print(f"Saved generated strings to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
