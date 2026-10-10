"""
prepare_vocab.py
Builds the vocabulary locally from public sources and computes BLIP-2 embeddings.
"""
import os
import pickle
import torch
from src import build_vocab, models, utils

def main():
    os.makedirs("data", exist_ok=True)
    vocab_path = "data/keywords.pkl"
    emb_path = "data/keywords_emb_blip2_mean.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Mine and filter vocabulary
    if not os.path.exists(vocab_path):
        print("\n--- Step 1: Building vocabulary from open sources ---")
        keywords = build_vocab.build_vocabulary(
            cache_dir="./cache_data",
            output_path=vocab_path,
            verbose=True
        )
    else:
        print(f"\nFound existing vocabulary at {vocab_path}. Loading...")
        with open(vocab_path, "rb") as f:
            keywords = pickle.load(f)

    # 2. Precompute embeddings
    if not os.path.exists(emb_path):
        print(f"\n--- Step 2: Precomputing BLIP-2 embeddings for {len(keywords)} terms ---")
        processor, text_model, _ = models.load_blip2_components()
        
        # Matches your strategy="mean" requirement
        emb_matrix = utils.blip_encode_text(
            text_list=keywords,
            processor=processor,
            text_model=text_model,
            device=device,
            batch_size=128,
            strategy="mean"
        )
        
        torch.save(emb_matrix, emb_path)
        print(f"Saved embedding matrix to {emb_path} (Shape: {emb_matrix.shape})")
    else:
        print(f"Embeddings already exist at {emb_path}.")

if __name__ == "__main__":
    main()