import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from rdkit import Chem
from tqdm import tqdm

# Added HuggingFace transformers imports for SciBERT
from transformers import AutoTokenizer, AutoModel

from model import LinkerGPT
from smiles_tokenizer import SMILESTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Sampling Script for LinkerGPT via SciBERT Prompts")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of SMILES to generate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max length of generated SMILES (bumped to prevent truncation)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--output_file", type=str, default="generated_smiles.txt", help="File to save outputs")
    
    # Replaced --properties with --text_description
    parser.add_argument(
        "--text_description", 
        type=str, 
        default=None, 
        help="Natural language description of target molecular properties to encode using SciBERT."
    )
    parser.add_argument(
        "--scibert_model",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="Hugging Face path for SciBERT weights."
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
    vocab_size = len(tokenizer.token2idx)
    
    # Extract structural token pointers safely
    bos_idx = tokenizer.token2idx.get("<bos>", 1)
    eos_idx = tokenizer.token2idx.get("<eos>", 2)

    # -------------------------------------------------------------
    # 3. Instantiate Model matching Checkpoint Configuration
    # -------------------------------------------------------------
    model = LinkerGPT(
        vocab_size=vocab_size,
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
    # 4. Handle Text Conditioning using SciBERT
    # -------------------------------------------------------------
    prop_vec = None
    if is_conditional:
        if args.text_description is None:
            raise ValueError(
                "This checkpoint was trained conditionally. Please provide a prompt via --text_description."
            )
            
        print(f"\nEncoding description using SciBERT: '{args.text_description}'")
        
        # Load SciBERT components
        bert_tokenizer = AutoTokenizer.from_pretrained(args.scibert_model)
        bert_model = AutoModel.from_pretrained(args.scibert_model).to(device)
        bert_model.eval()
        
        # Tokenize text description
        encoded_input = bert_tokenizer(
            [args.text_description],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Extract CLS sequence-level embedding
        with torch.no_grad():
            model_outputs = bert_model(**encoded_input)
            raw_embeddings = model_outputs.last_hidden_state[:, 0, :] # Shape: (1, 768)
            
        if raw_embeddings.size(1) != config["property_dim"]:
            raise ValueError(
                f"SciBERT output dimension ({raw_embeddings.size(1)}) does not match "
                f"the model's expected property_dim ({config['property_dim']})."
            )
        
        # Standardize using preserved training Z-score statistics
        prop_mean = ckpt["prop_mean"].to(device)
        prop_std = ckpt["prop_std"].to(device)
        prop_std = torch.where(prop_std == 0, torch.ones_like(prop_std), prop_std)
        
        # normalized_props = (raw_embeddings - prop_mean) / prop_std
        normalized_props = raw_embeddings
        
        # Expand across generation batch axis
        prop_vec = normalized_props.repeat(args.num_samples, 1).to(device)
    else:
        if args.text_description is not None:
            print("[Warning] Checkpoint is unconditional! --text_description input will be ignored.")

    # -------------------------------------------------------------
    # 5. Execute Autoregressive Batch Generation
    # -------------------------------------------------------------
    print(f"\nGenerating {args.num_samples} samples (temp={args.temperature})...")
    
    input_ids = torch.full(
        (args.num_samples, 1),
        bos_idx,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=input_ids,
            prop_vec=prop_vec,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eos_token_id=eos_idx,
        )

    # -------------------------------------------------------------
    # 6. Parse Outputs and Calculate Validity via RDKit
    # -------------------------------------------------------------
    valid_smiles_count = 0
    results = []

    print("\nEvaluating molecular structures:")
    for i, tokens in enumerate(generated_tokens):
        smiles = tokenizer.decode(tokens.tolist(), remove_special_tokens=True)
        smiles = smiles.split()[0] if smiles.strip() else ""
        
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        is_valid = mol is not None
        
        if is_valid:
            valid_smiles_count += 1
            
        status_str = "VALID" if is_valid else "INVALID"
        print(f"[{i+1:03d}] ({status_str}, len={len(smiles)}) -> {smiles}")
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