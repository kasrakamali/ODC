"""
evaluate.py
Evaluates generated images using CLIP-Score and BLIP-ITM across seeds.
"""
import argparse
import glob
import json
from src import data, models, utils

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated paired images")
    parser.add_argument(
        "--output_base", 
        type=str, 
        default="./results/sdxl_custom_prompts", 
        help="Base path used in run_experiment (without _0, _1)"
    )
    parser.add_argument(
        "--metrics", 
        nargs="+", 
        default=["clip", "blip"], 
        choices=["clip", "blip"],
        help="Metrics to compute (options: clip, blip)"
    )
    parser.add_argument(
        "--save_json", 
        type=str, 
        default="evaluation_results.json",
        help="Path to save evaluation summary"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_dirs = sorted(glob.glob(f"{args.output_base}_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No directories found matching '{args.output_base}_*'")

    print(f"Found evaluation folders: {seed_dirs}")
    results = {}

    # 1. CLIP-Score
    if "clip" in args.metrics:
        print("\n[1/2] Computing CLIP-Scores...")
        clip_res = utils.calculate_clip_score_multi_seed(seed_dirs, data.PROMPTS)
        results["clip"] = clip_res
        
        orig_mean = clip_res['original']['mean']
        orig_std = clip_res['original']['std']
        impr_mean = clip_res['improved']['mean']
        impr_std = clip_res['improved']['std']
        delta_mean = clip_res['delta']['mean']
        p_val = clip_res['significance']['p_value']
        
        print(f"Original: {orig_mean:.2f} ± {orig_std:.2f}")
        print(f"Improved: {impr_mean:.2f} ± {impr_std:.2f}")
        print(f"Delta:    {delta_mean:.2f} (p-val: {p_val:.4e})")

    # 2. BLIP-ITM Score
    if "blip" in args.metrics:
        print("\n[2/2] Computing BLIP-ITM Alignment Scores...")
        processor, itm_model = models.load_blip2_retrieval_model()
        blip_res = utils.calculate_blip_itm_score_multi_seed(
            seed_dirs, data.PROMPTS, processor, itm_model
        )
        results["blip_itm"] = blip_res
        
        orig_mean = blip_res['original']['mean']
        impr_mean = blip_res['improved']['mean']
        delta_mean = blip_res['delta']['mean']
        p_val = blip_res['significance']['p_value']
        
        print(f"Original: {orig_mean:.4f}")
        print(f"Improved: {impr_mean:.4f}")
        print(f"Delta:    {delta_mean:.4f} (p-val: {p_val:.4e})")

    # Save outputs
    with open(args.save_json, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nFull results saved to {args.save_json}")


if __name__ == "__main__":
    main()