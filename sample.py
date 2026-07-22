import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from rdkit import Chem
from tqdm import tqdm

from model import LinkerGen
from smiles_tokenizer import SMILESTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Sampling Script for LinkerGen")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of SMILES to generate")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max length of generated SMILES")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--output_file", type=str, default="generated_smiles.txt", help="File to save outputs")
    
    # Conditional target properties (Only used if the checkpoint was trained conditionally)
    parser.add_argument(
        "--properties", 
        type=float, 
        nargs="+", 
        default=None, 
        help="Target values for property columns sequentially (e.g. 350.0 2.5 1 2)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------
    # 1. Load Checkpoint Metadata
    # -------------------------------------------------------------
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    is_conditional = ckpt.get("conditional", False)
    
    # -------------------------------------------------------------
    # 2. Reconstruct Tokenizer
    # -------------------------------------------------------------
    tokenizer = SMILESTokenizer()
    tokenizer.token2idx = ckpt["tokenizer_vocab"]
    tokenizer.idx2token = {v: k for k, v in tokenizer.token2idx.items()}
    #tokenizer.vocab_size = len(tokenizer.token2idx)
    
    # Fallback setups for typical index configurations
    #tokenizer.pad_idx = tokenizer.token2idx.get("<pad>", 0)
    #tokenizer.bos_idx = tokenizer.token2idx.get("<bos>", 1)
    #tokenizer.eos_idx = tokenizer.token2idx.get("<eos>", 2)

    # -------------------------------------------------------------
    # 3. Instantiate Model matching Checkpoint Configuration
    # -------------------------------------------------------------
    model = LinkerGen(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        d_hidden=config["d_hidden"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_seq_len=config["max_length"],
        property_dim=config["property_dim"],
        n_memory=config["n_memory"],
    )
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # -------------------------------------------------------------
    # 4. Handle Conditioning Vectors (with Z-score scaling)
    # -------------------------------------------------------------
    prop_vec = None
    if is_conditional:
        print(f"\nModel is conditional. Expected targets for: {ckpt['property_columns']}")
        if args.properties is None:
            raise ValueError(
                f"This checkpoint requires conditional parameters. "
                f"Please provide values via --properties matching dimensions ({config['property_dim']})."
            )
        if len(args.properties) != config["property_dim"]:
            raise ValueError(
                f"Dimension mismatch. Expected {config['property_dim']} values, got {len(args.properties)}."
            )
            
        # Convert raw entries to tensors
        raw_props = torch.tensor(args.properties, dtype=torch.float).unsqueeze(0) # (1, property_dim)
        
        # Standardize using the training statistics preserved in the checkpoint
        prop_mean = ckpt["prop_mean"]
        prop_std = ckpt["prop_std"]
        
        # Guard against zero-division errors if standard deviations are static
        prop_std = torch.where(prop_std == 0, torch.ones_like(prop_std), prop_std)
        
        normalized_props = (raw_props - prop_mean) / prop_std
        # Expand across batch generation axis
        prop_vec = normalized_props.repeat(args.num_samples, 1).to(device)
    else:
        if args.properties is not None:
            print("[Warning] Checkpoint is unconditional! --properties input will be ignored.")

    # -------------------------------------------------------------
    # 5. Execute Autoregressive Batch Generation
    # -------------------------------------------------------------
    print(f"\nGenerating {args.num_samples} samples (temp={args.temperature})...")
    
    input_ids = torch.full(
        (args.num_samples, 1),
        tokenizer.bos_idx,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=input_ids,
            prop_vec=prop_vec,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_idx,
        )

    # -------------------------------------------------------------
    # 6. Parse Outputs and Calculate Validity via RDKit
    # -------------------------------------------------------------
    valid_smiles_count = 0
    results = []

    print("\nEvaluating molecular structures:")
    for i, tokens in enumerate(generated_tokens):
        smiles = tokenizer.decode(tokens.tolist(), remove_special_tokens=True)
        
        # Strip trailing syntax loops occasionally caused by loose generations
        smiles = smiles.split()[0] if smiles.strip() else ""
        
        # Check chemical sanity
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        is_valid = mol is not None
        
        if is_valid:
            valid_smiles_count += 1
            
        status_str = "VALID" if is_valid else "INVALID"
        print(f"[{i+1:03d}] ({status_str}) -> {smiles}")
        results.append(smiles)

    validity_rate = valid_smiles_count / args.num_samples
    print("\n" + "="*50)
    print(f"Generation Statistics:")
    print(f"  Total Extracted:     {args.num_samples}")
    print(f"  Chemically Valid:    {valid_smiles_count}")
    print(f"  Overall Validity:    {validity_rate * 100:.2f}%")
    print("="*50)

    # -------------------------------------------------------------
    # 7. Write results out to disk
    # -------------------------------------------------------------
    output_path = Path(args.output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        for smiles in results:
            f.write(f"{smiles}\n")
    print(f"Saved generated strings to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
