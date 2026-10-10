"""
run_experiment.py
Runs multi-seed batch generation experiments across prompt sets.
"""
import argparse
import os
import pickle
from functools import partial
import torch

from src import data, method, models, utils

def parse_args():
    parser = argparse.ArgumentParser(description="Run QR Concept Subtraction Generation")
    parser.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "flux"])
    parser.add_argument("--output_dir", type=str, default="./results/sdxl_custom_prompts")
    parser.add_argument("--vocab_path", type=str, default="data/keywords.pkl")
    parser.add_argument("--emb_path", type=str, default="data/keywords_emb_blip2_mean.pt")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--num_prompts", type=int, default=None, help="Limit number of prompts (for testing)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.vocab_path) or not os.path.exists(args.emb_path):
        raise FileNotFoundError(
            f"Vocabulary files missing. Please run `python prepare_vocab.py` first."
        )

    print("Loading vocabulary and embeddings...")
    with open(args.vocab_path, "rb") as f:
        keywords = pickle.load(f)
    kw_matrix = torch.load(args.emb_path, map_location=device)

    print("Loading BLIP-2 components...")
    processor, text_model, vision_model = models.load_blip2_components()

    print(f"Loading {args.model.upper()} pipeline...")
    if args.model == "sdxl":
        pipe = models.load_sdxl_pipeline(device=device)
    else:
        pipe = models.load_flux_pipeline(device=device)
    pipe.set_progress_bar_config(disable=True)

    # Partial matches your notebook:
    get_modified_embeddings = partial(
        method.get_clean_embeddings_qr, 
        alpha=args.alpha, 
        preserve_start=False
    )

    prompts_to_run = data.PROMPTS[:args.num_prompts] if args.num_prompts else data.PROMPTS

    print(f"\nStarting experiment: {len(prompts_to_run)} prompts across {len(utils.SEEDS)} seeds.")
    method.run_intervention_experiment(
        pipe=pipe,
        prompts=prompts_to_run,
        seeds=utils.SEEDS,
        save_dir=args.output_dir,
        get_modified_embeddings=get_modified_embeddings,
        encoding_processor=processor,
        encoding_model=(text_model, vision_model),
        emb_matrix=kw_matrix,
        keywords_list=keywords
    )
    print("\nGeneration finished successfully.")

if __name__ == "__main__":
    main()