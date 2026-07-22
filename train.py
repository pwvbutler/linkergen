"""
Training script for LinkerGen.

Two phases share this one script, switched by --conditional:

  1) Unconditional pretraining (plain SMILES file, one per line):
        python train.py \
            --data_path data/pretrain.smi \
            --epochs 20 --batch_size 64 \
            --save_dir checkpoints/pretrain

  2) Conditional fine-tuning (TSV with property columns + SMILES):
        python train.py \
            --data_path data/finetune.tsv \
            --conditional \
            --init_from checkpoints/pretrain/best_model.pt \
            --cond_drop_prob 0.1 \
            --epochs 20 --batch_size 64 \
            --save_dir checkpoints/finetune
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from rdkit import Chem

from data_utils import (
    create_dataloader,
    load_smiles_from_file,
    load_dataset_from_file,
    load_core_smiles_embeddings_from_json,
    train_val_split,
    PROPERTY_COLUMNS,
)

from model import LinkerGen
from smiles_tokenizer import SMILESTokenizer


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    cond_drop_prob: float = 0.0,
):
    model.train()

    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # property vector is only present in conditional batches
        prop_vec = batch.get("prop_vec")
        if prop_vec is not None:
            prop_vec = prop_vec.to(device)
            # classifier-free guidance dropout (batch-level): occasionally
            # train an unconditional step so the prop_vec=None path stays
            # calibrated for CFG at sampling time. See note below on why this
            # is batch-level rather than per-sample for this architecture.
            if cond_drop_prob > 0.0 and random.random() < cond_drop_prob:
                prop_vec = None

        # -------------------------------------------------------------
        # Forward
        # -------------------------------------------------------------

        logits = model(input_ids, prop_vec)

        # -------------------------------------------------------------
        # Loss
        # -------------------------------------------------------------

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        # -------------------------------------------------------------
        # Backward
        # -------------------------------------------------------------

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # -------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # validation is always fully conditioned (no dropout)
        prop_vec = batch.get("prop_vec")
        if prop_vec is not None:
            prop_vec = prop_vec.to(device)

        logits = model(input_ids, prop_vec)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    # conditioning
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Load properties from a TSV and condition on them. "
             "Omit for unconditional pretraining on a plain SMILES file.",
    )
    parser.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.0,
        help="Prob. of dropping the condition during training (for CFG). "
             "Only used when --conditional.",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint to warm-start from "
             "(e.g. the unconditional pretrain checkpoint).",
    )
    
    parser.add_argument(
        "--json_embeddings",
        action="store_true",
        help="Load core_smiles and embedding from a JSON file. "
            "The embedding is used as prop_vec.",
    )

    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)#default=3e-4)
    parser.add_argument("--max_length", type=int, default=512)

    # model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_hidden", type=int, default=1024)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument(
        "--property_dim",
        type=int,
        default=len(PROPERTY_COLUMNS),
        help="Number of conditioning properties. Must match the number of "
             "property columns loaded from the TSV.",
    )
    parser.add_argument("--n_memory", type=int, default=8)

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # import pdb; pdb.set_trace()
    # -------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------

    if args.json_embeddings:
        smiles_list, properties = load_core_smiles_embeddings_from_json(
            args.data_path,
            smiles_key="core_smiles",
            embedding_key="embedding",
        )

        print(
            f"Loaded {len(smiles_list)} core SMILES with "
            f"{properties.size(1)}-dimensional embeddings"
        )

        if properties.size(1) != args.property_dim:
            raise ValueError(
                f"--property_dim={args.property_dim} but JSON embeddings have "
                f"{properties.size(1)} dimensions. Set --property_dim {properties.size(1)}."
            )

    elif args.conditional:
        smiles_list, properties = load_dataset_from_file(
            args.data_path,
            property_columns=PROPERTY_COLUMNS,
        )
        print(
            f"Loaded {len(smiles_list)} SMILES with "
            f"{properties.size(1)} properties: {PROPERTY_COLUMNS}"
        )
        if properties.size(1) != args.property_dim:
            raise ValueError(
                f"--property_dim={args.property_dim} but the TSV has "
                f"{properties.size(1)} property columns. They must match."
            )

    else:
        smiles_list = load_smiles_from_file(args.data_path)
        properties = None
        print(f"Loaded {len(smiles_list)} SMILES, unconditional")

    # -------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------

    tokenizer = SMILESTokenizer()
    tokenizer.build_vocab(smiles_list)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # -------------------------------------------------------------
    # Train/Val Split  (keeps SMILES and properties aligned)
    # -------------------------------------------------------------

    train_smiles, val_smiles, train_props, val_props = train_val_split(
        smiles_list,
        properties=properties,
        val_ratio=0.1,
        seed=42,
    )

    print(f"Train: {len(train_smiles)}")
    print(f"Val:   {len(val_smiles)}")


    # -------------------------------------------------------------
    # DataLoaders
    #   Stats are computed from the TRAIN dataset only, then reused for
    #   val so there is no leakage. These same stats must be used at
    #   inference -- they are saved into the checkpoint below.
    # -------------------------------------------------------------

    train_loader = create_dataloader(
        smiles_list=train_smiles,
        tokenizer=tokenizer,
        properties=train_props,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
    )

    prop_mean = train_loader.dataset.prop_mean
    prop_std = train_loader.dataset.prop_std

    val_loader = create_dataloader(
        smiles_list=val_smiles,
        tokenizer=tokenizer,
        properties=val_props,
        prop_mean=prop_mean,   # reuse train stats (None in unconditional mode)
        prop_std=prop_std,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = LinkerGen(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_length,
        property_dim=args.property_dim,
        n_memory=args.n_memory,
    )

    # warm-start from a pretrained checkpoint if given
    if args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        print(f"Warm-started from {args.init_from}")
        if missing:
            print(f"  (randomly initialized, not in checkpoint): {missing}")
        if unexpected:
            print(f"  (in checkpoint but unused): {unexpected}")

    model = model.to(device)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # -------------------------------------------------------------
    # Optimizer / Loss
    # -------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    # -------------------------------------------------------------
    # Checkpoint Directory
    # -------------------------------------------------------------

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(args.epochs):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 60)

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cond_drop_prob=args.cond_drop_prob if args.conditional else 0.0,
        )

        val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # ---------------------------------------------------------
        # Save Best Model
        #   The stats are part of the model: without them the weights
        #   are unusable, because nothing else records what input scale
        #   the conditioning was trained on.
        # ---------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            checkpoint_path = save_dir / "best_model.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "tokenizer_vocab": tokenizer.token2idx,
                    "config": vars(args),
                    # conditioning metadata
                    "conditional": args.conditional,
                    "property_columns": PROPERTY_COLUMNS if args.conditional else None,
                    "prop_mean": prop_mean.cpu() if prop_mean is not None else None,
                    "prop_std": prop_std.cpu() if prop_std is not None else None,
                },
                checkpoint_path,
            )

            print(f"Saved best model to: {checkpoint_path}")

    print("\nTraining complete.")

    # ---------------------------------------------------------
    # Sample
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("Sampling Test")
    print("=" * 60)

    model.eval()
    num_samples = 100

    input_ids = torch.full(
        (num_samples, 1),
        tokenizer.bos_idx,
        dtype=torch.long,
        device=device,
    )
    total_valid = 0

    if args.conditional:
        # condition on real property vectors drawn from the val set.
        # val_props are RAW; standardize with the train stats before use,
        # exactly as you must at real inference time.
        idx = torch.randint(0, val_props.size(0), (num_samples,))
        raw = val_props[idx]
        prop_vec = ((raw - prop_mean) / prop_std).to(device)

        generated = model.generate(
            input_ids=input_ids,
            prop_vec=prop_vec,            # requires the updated generate()
            max_new_tokens=512,
            temperature=0.8,
            eos_token_id=tokenizer.eos_idx,
        )

        for i, tokens in enumerate(generated):
            smiles = tokenizer.decode(
                tokens.tolist(), remove_special_tokens=True
            )
            target = ", ".join(
                f"{c}={v:.3g}" for c, v in zip(PROPERTY_COLUMNS, raw[i].tolist())
            )
            try:
                m = Chem.MolFromSmiles(smiles)
                valid = True
            except:
                valid = False

            if m is None:
                valid = False
            if valid:
                total_valid += 1

            print(f"{i+1:02d} ({valid=}, len={len(smiles)}): {smiles}")
            #print(f"{i+1:02d} ({valid=}, len={len(smiles)}, props={prop_vec.shape[1]}): {smiles}")
            #print(f"{i+1:02d}: {smiles}")
            #print(f"    target: {target}")
    else:
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            temperature=0.8,
            eos_token_id=tokenizer.eos_idx,
        )

        for i, tokens in enumerate(generated):
            smiles = tokenizer.decode(
                tokens.tolist(), remove_special_tokens=True
            )
            try:
                m = Chem.MolFromSmiles(smiles)
                valid = True
            except:
                valid = False

            if m is None:
                valid = False

            if valid:
                total_valid += 1

            print(f"{i+1:02d} ({valid=}, len={len(smiles)}): {smiles}")

    print(f"validity = {total_valid / num_samples}")

if __name__ == "__main__":
    main()
