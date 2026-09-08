"""
Unified training entrypoint for LinkerGen.

One script covers all three conditioning modes, switched by --condition:

  1) Unconditional pretraining (plain SMILES file, one per line):
        linkergen-train --data_path data/pretrain.smi --condition none \\
            --epochs 20 --batch_size 64 --save_dir checkpoints/pretrain

  2) Property-conditioned fine-tuning (TSV with property columns + SMILES),
     warm-started from a pretrained checkpoint:
        linkergen-train --data_path data/finetune.tsv --condition property \\
            --init_from checkpoints/pretrain/best_model.pt --cond_drop_prob 0.1 \\
            --epochs 20 --batch_size 64 --save_dir checkpoints/finetune

  3) Text-conditioned fine-tuning (JSON of {original_smiles, text_description}):
        linkergen-train --data_path data/finetune.json --condition text \\
            --init_from checkpoints/pretrain/best_model.pt --cond_drop_prob 0.1 \\
            --epochs 20 --batch_size 64 --save_dir checkpoints/finetune_text

--cond_drop_prob randomly drops the condition batch-wise (classifier-free guidance);
it is a no-op when --condition none. --data_format only matters for --condition
property: "tsv" reads SMILES + property columns from a delimited file, while
"json_embeddings" reads core_smiles + a precomputed embedding used directly as the
property vector. In text mode, descriptions are encoded by a frozen Nach0Encoder
(see nach0.py) once per batch, never inside LinkerGen itself.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from rdkit import Chem
from tqdm import tqdm

from .data import (
    PROPERTY_COLUMNS,
    create_dataloader,
    load_core_smiles_embeddings_from_json,
    load_dataset_from_file,
    load_smiles_and_descriptions_from_json,
    load_smiles_from_file,
    train_val_split,
)
from .model import LinkerGen
from .nach0 import Nach0Encoder
from .tokenizer import SMILESTokenizer


def prepare_condition(batch, condition_type, device, nach0_encoder=None):
    """
    Turn a collated batch's raw "condition" entry into what LinkerGen.forward
    expects: a standardized property tensor, an encoded (hidden_states, mask)
    pair, or None.
    """
    condition = batch.get("condition")
    if condition is None:
        return None
    if condition_type == "property":
        return condition.to(device)
    # "text": one nach0 forward pass per batch, at the same frequency as the
    # property path's condition tensor, now made explicit since nach0
    # lives outside the model (see nach0.py, and LinkerGen's docstring in
    # model.py).
    return nach0_encoder(condition, device)


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    condition_type="none",
    cond_drop_prob=0.0,
    nach0_encoder=None,
):
    model.train()

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        condition = prepare_condition(batch, condition_type, device, nach0_encoder)
        # classifier-free guidance dropout (batch-level): occasionally train an
        # unconditional step so the condition=None path stays calibrated for
        # CFG at sampling time.
        if condition is not None and cond_drop_prob > 0.0 and random.random() < cond_drop_prob:
            condition = None

        logits = model(input_ids, condition)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device, condition_type="none", nach0_encoder=None):
    model.eval()

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # validation is always fully conditioned (no CFG dropout)
        condition = prepare_condition(batch, condition_type, device, nach0_encoder)

        logits = model(input_ids, condition)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    # conditioning
    parser.add_argument(
        "--condition",
        type=str,
        choices=["none", "property", "text"],
        default="none",
        help="Conditioning mode: unconditional pretraining, numeric property "
             "vectors, or natural-language text descriptions.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        choices=["tsv", "json_embeddings"],
        default="tsv",
        help="--condition property only: 'tsv' loads SMILES + property columns "
             "from a delimited file; 'json_embeddings' loads core_smiles + a "
             "precomputed embedding used directly as the property vector.",
    )
    parser.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.0,
        help="Probability of dropping the condition during training (for CFG). "
             "No-op when --condition none.",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Path to a checkpoint to warm-start from. May have a different "
             "--condition than this run (e.g. none to text).",
    )

    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
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
        help="--condition property only: number of conditioning properties. "
             "Must match the number of columns loaded from the data.",
    )
    parser.add_argument("--n_memory", type=int, default=8, help="--condition property only.")
    parser.add_argument(
        "--nach0_model_name",
        type=str,
        default="insilicomedicine/nach0_base",
        help="--condition text only: pretrained nach0 T5 checkpoint used to "
             "encode descriptions. Always frozen, never fine-tuned.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    if args.condition == "property":
        if args.data_format == "json_embeddings":
            smiles_list, conditions = load_core_smiles_embeddings_from_json(
                args.data_path,
                smiles_key="core_smiles",
                embedding_key="embedding",
            )
            print(
                f"Loaded {len(smiles_list)} core SMILES with "
                f"{conditions.size(1)}-dimensional embeddings"
            )
        else:
            smiles_list, conditions = load_dataset_from_file(
                args.data_path,
                property_columns=PROPERTY_COLUMNS,
            )
            print(
                f"Loaded {len(smiles_list)} SMILES with "
                f"{conditions.size(1)} properties: {PROPERTY_COLUMNS}"
            )
        if conditions.size(1) != args.property_dim:
            raise ValueError(
                f"--property_dim={args.property_dim} but the data has "
                f"{conditions.size(1)} columns. They must match."
            )
    elif args.condition == "text":
        smiles_list, conditions = load_smiles_and_descriptions_from_json(args.data_path)
        print(f"Loaded {len(smiles_list)} SMILES with text descriptions")
    else:
        smiles_list = load_smiles_from_file(args.data_path)
        conditions = None
        print(f"Loaded {len(smiles_list)} SMILES, unconditional")

    tokenizer = SMILESTokenizer()
    tokenizer.build_vocab(smiles_list)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Split into train/validation sets, keeping SMILES and conditions aligned.
    train_smiles, val_smiles, train_conditions, val_conditions = train_val_split(
        smiles_list,
        conditions=conditions,
        val_ratio=0.1,
        seed=42,
    )

    print(f"Train: {len(train_smiles)}")
    print(f"Val:   {len(val_smiles)}")

    # DataLoaders. Property stats are computed from the training split only,
    # then reused for validation so there is no leakage. The same stats must
    # be used at inference, so they are saved into the checkpoint below.
    train_loader = create_dataloader(
        smiles_list=train_smiles,
        tokenizer=tokenizer,
        condition_type=args.condition,
        conditions=train_conditions,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
    )

    prop_mean = train_loader.dataset.prop_mean
    prop_std = train_loader.dataset.prop_std

    val_loader = create_dataloader(
        smiles_list=val_smiles,
        tokenizer=tokenizer,
        condition_type=args.condition,
        conditions=val_conditions,
        prop_mean=prop_mean,  # reuse train stats (None outside property mode)
        prop_std=prop_std,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )

    # Conditioning encoder (text mode only) and model
    nach0_encoder = None
    text_hidden_size = None
    if args.condition == "text":
        nach0_encoder = Nach0Encoder(args.nach0_model_name).to(device)
        text_hidden_size = nach0_encoder.hidden_size

    model = LinkerGen(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_length,
        condition_type=args.condition,
        property_dim=args.property_dim if args.condition == "property" else None,
        n_memory=args.n_memory,
        text_hidden_size=text_hidden_size,
    )

    if args.init_from is not None:
        ckpt = torch.load(args.init_from, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Warm-started from {args.init_from}")
        if missing:
            print(f"  (randomly initialized, not in checkpoint): {missing}")
        if unexpected:
            print(f"  (in checkpoint but unused): {unexpected}")

    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss. nach0_encoder's parameters are always frozen (see
    # nach0.py) and never chained into the optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
            condition_type=args.condition,
            cond_drop_prob=args.cond_drop_prob,
            nach0_encoder=nach0_encoder,
        )

        val_loss = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            condition_type=args.condition,
            nach0_encoder=nach0_encoder,
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        # Save the best model. Conditioning metadata is saved alongside the
        # weights, since the checkpoint is the only record of which mode,
        # input scale, or text encoder the conditioning used.
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            checkpoint_path = save_dir / "best_model.pt"

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "tokenizer_vocab": tokenizer.token2idx,
                "config": vars(args),
                "condition_type": args.condition,
            }
            if args.condition == "property":
                checkpoint.update(
                    property_columns=PROPERTY_COLUMNS,
                    prop_mean=prop_mean.cpu() if prop_mean is not None else None,
                    prop_std=prop_std.cpu() if prop_std is not None else None,
                )
            elif args.condition == "text":
                checkpoint.update(
                    nach0_model_name=args.nach0_model_name,
                    text_hidden_size=text_hidden_size,
                )

            torch.save(checkpoint, checkpoint_path)

            print(f"Saved best model to: {checkpoint_path}")

    print("\nTraining complete.")

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

    if args.condition == "property":
        # condition on real property vectors drawn from the val set. val_conditions
        # holds raw (un-standardized) values; standardize with the train stats
        # before use, matching what real inference requires.
        idx = torch.randint(0, val_conditions.size(0), (num_samples,))
        raw = val_conditions[idx]
        condition = ((raw - prop_mean) / prop_std).to(device)
    elif args.condition == "text":
        # condition on real descriptions drawn from the val set, encoded once and
        # reused unchanged across the whole generate() loop.
        idx = torch.randint(0, len(val_conditions), (num_samples,))
        descriptions = [val_conditions[i] for i in idx.tolist()]
        condition = nach0_encoder(descriptions, device)
    else:
        condition = None

    generated = model.generate(
        input_ids=input_ids,
        condition=condition,
        max_new_tokens=512,
        temperature=0.8,
        eos_token_id=tokenizer.eos_idx,
    )

    for i, tokens in enumerate(generated):
        smiles = tokenizer.decode(tokens.tolist(), remove_special_tokens=True)
        mol = Chem.MolFromSmiles(smiles)
        valid = mol is not None
        if valid:
            total_valid += 1
        print(f"{i + 1:02d} ({valid=}, len={len(smiles)}): {smiles}")

    print(f"validity = {total_valid / num_samples}")


if __name__ == "__main__":
    main()
