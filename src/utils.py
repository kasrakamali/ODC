import math
import os
import random
import shutil
import statistics
import tempfile

import clip
import numpy as np
import torch
import tqdm
from cleanfid import fid
from PIL import Image
from scipy import stats

SEEDS = [1608637542, 1273642419, 1935803228]


@torch.no_grad()
def blip_encode_image(pil_image, processor, vision_model, device="cuda"):
    """
    Encodes an image into a normalized embedding using BLIP-2.
    """
    with torch.autocast(device):
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        image_outputs = vision_model(**inputs, return_dict=True)

    image_emb = image_outputs.image_embeds.mean(dim=1)
    return torch.nn.functional.normalize(image_emb, dim=-1)


@torch.no_grad()
def blip_encode_text(text_list, processor, text_model, device="cuda", batch_size=64, strategy="cls"):
    """
    Encodes a list of texts into normalized embeddings using BLIP-2.
    
    Args:
        text_list (list): List of strings to encode.
        processor: BLIP processor.
        text_model: BLIP text model.
        device (str): "cuda" or "cpu".
        batch_size (int): Batch size for inference.
        strategy (str): Pooling strategy. Options: "cls" or "mean".
                        "cls" uses the [CLS] token (standard).
                        "mean" uses the average of all masked tokens.
    """
    if strategy not in ["cls", "mean"]:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'cls' or 'mean'.")

    all_vecs = []
    text_model.eval()

    for i in tqdm.trange(0, len(text_list), batch_size, desc=f"Encoding ({strategy})"):

        batch_texts = text_list[i: i + batch_size]

        with torch.autocast(device):
            inputs = processor(text=batch_texts, padding=True,
                               truncation=True, return_tensors="pt").to(device)
            text_outputs = text_model(**inputs, return_dict=True)

        token_embeddings = text_outputs.text_embeds

        if strategy == "cls":
            raw_emb = token_embeddings[:, 0, :]

        elif strategy == "mean":
            attention_mask = inputs['attention_mask']

            input_mask_expanded = attention_mask.unsqueeze(
                -1).expand(token_embeddings.size()).float()

            sum_embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1)

            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            raw_emb = sum_embeddings / sum_mask

        vec = torch.nn.functional.normalize(raw_emb, dim=-1)

        all_vecs.append(vec.cpu())

    return torch.cat(all_vecs)


@torch.no_grad()
def clip_encode_image(pil_image, processor, model, device="cuda"):
    """
    Generates a normalized embedding for an image using CLIP.
    """
    with torch.autocast(device):
        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        image_emb = model.get_image_features(**inputs)

    return torch.nn.functional.normalize(image_emb, dim=-1)


@torch.no_grad()
def clip_encode_text(text_list, processor, model, device="cuda", batch_size=64):
    """
    Generates normalized embeddings for a list of text keywords using CLIP.
    """
    all_vecs = []

    model.eval()

    for i in tqdm.trange(0, len(text_list), batch_size, desc="Encoding CLIP Text"):
        batch_texts = text_list[i: i + batch_size]

        with torch.autocast(device):
            inputs = processor(text=batch_texts, padding=True,
                               truncation=True, return_tensors="pt").to(device)

            text_emb = model.get_text_features(**inputs)

        vec = torch.nn.functional.normalize(text_emb, dim=-1)

        all_vecs.append(vec.cpu())

    return torch.cat(all_vecs)


@torch.no_grad()
def siglip2_encode_image(pil_image, processor, model, device="cuda", normalize=True):
    """
    Generates an embedding for an image using SigLIP2.
    """
    with torch.autocast(device):
        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        image_emb = model.get_image_features(**inputs)

    if normalize:
        return torch.nn.functional.normalize(image_emb, dim=-1)
    else:
        return image_emb


@torch.no_grad()
def siglip2_encode_text(text_list, processor, model, device="cuda", batch_size=64, normalize=True):
    """
    Generates embeddings for a list of text keywords using SigLIP2.
    """
    all_vecs = []

    model.eval()

    for i in tqdm.trange(0, len(text_list), batch_size, desc="Encoding SigLIP2 Text"):
        batch_texts = text_list[i: i + batch_size]

        with torch.autocast(device):
            inputs = processor(text=batch_texts, padding=True,
                               truncation=True, return_tensors="pt").to(device)

            text_emb = model.get_text_features(**inputs)

        if normalize:
            text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

        all_vecs.append(text_emb.cpu())

    return torch.cat(all_vecs)


@torch.no_grad()
def evaclip_encode_image(pil_image, preprocess_val, model, device="cuda", normalize=True):
    """
    Generates an embedding for an image using EVA-CLIP.
    """
    with torch.autocast(device):
        image_input = preprocess_val(pil_image).unsqueeze(0).to(device)
        
        image_emb = model.encode_image(image_input)

    if normalize:
        return torch.nn.functional.normalize(image_emb, dim=-1)
    else:
        return image_emb


@torch.no_grad()
def evaclip_encode_text(text_list, tokenizer, model, device="cuda", batch_size=64, normalize=True):
    """
    Generates embeddings for a list of text keywords using EVA-CLIP.
    """
    all_vecs = []

    model.eval()

    for i in tqdm.trange(0, len(text_list), batch_size, desc="Encoding EVA-CLIP Text"):
        batch_texts = text_list[i: i + batch_size]

        with torch.autocast(device):
            text_input = tokenizer(batch_texts).to(device)

            text_emb = model.encode_text(text_input)

        # Normalize
        if normalize:
            text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

        all_vecs.append(text_emb.cpu())

    return torch.cat(all_vecs)


def get_vector_rejection(v1, v2, normalize=True):
    """
    Computes the component of v2 that is orthogonal to v1.
    (Removes the direction of v1 from v2).
    
    Args:
        v1 (Tensor): The vector to project onto (must be float).
        v2 (Tensor): The vector to be decomposed (must be float).
        normalize (bool): If True, return a unit vector.
    """
    dot_product = torch.dot(v2.flatten(), v1.flatten())
    v1_norm_squared = torch.dot(v1.flatten(), v1.flatten())

    if v1_norm_squared == 0:
        orthogonal = v2
    else:
        projection = (dot_product / v1_norm_squared) * v1
        orthogonal = v2 - projection

    if normalize:
        ortho_norm = torch.norm(orthogonal)
        if ortho_norm > 0:
            orthogonal = orthogonal / ortho_norm

    return orthogonal


def get_safe_vocab(processor, model):
    """
    Creates a list of safe words for EITHER BLIP or CLIP models.
    Automatically detects model architecture and cleaning rules.
    """
    print("Building safe vocabulary...")

    if hasattr(model, "text_model") and hasattr(model.text_model, "embeddings"):
        if hasattr(model.text_model.embeddings, "word_embeddings"):
            emb_layer = model.text_model.embeddings.word_embeddings
            is_bpe = False
        elif hasattr(model.text_model.embeddings, "token_embedding"):
            emb_layer = model.text_model.embeddings.token_embedding
            is_bpe = True
        else:
            raise ValueError("Unknown embedding layer structure.")

    elif hasattr(model, "get_input_embeddings"):
        emb_layer = model.get_input_embeddings()
        is_bpe = False
    else:
        try:
            emb_layer = model.qformer.embeddings.word_embeddings
            is_bpe = False
        except AttributeError:
            raise ValueError("Could not find embedding layer.")

    max_valid_id = emb_layer.num_embeddings
    print(f"Detected embedding size: {max_valid_id}")

    safe_words = set()
    vocab_dict = processor.tokenizer.get_vocab()

    for word, token_id in vocab_dict.items():
        if token_id < max_valid_id:

            clean_word = word

            if is_bpe:
                clean_word = word.replace("</w>", "")
            else:
                if word.startswith("##"):
                    continue

            if clean_word.isalpha() and len(clean_word) > 2:
                safe_words.add(clean_word)

    safe_list = sorted(list(safe_words))
    print(f"Safe Vocabulary Size: {len(safe_list)} words")
    return safe_list


def genetic_text_search(target_vector, processor, model, safe_vocab,
                        model_type="blip",
                        seeds=[42, 100, 999],
                        num_outputs=5,
                        generations=30,
                        batch_size=500,
                        device="cuda"):
    """
    Evolutionary search to find text that matches a target embedding.
    Works for both BLIP and CLIP.
    
    Args:
        model_type (str): "blip" or "clip". Determines which encoder to use.
    """
    if model_type not in ["blip", "clip"]:
        raise ValueError("model_type must be 'blip' or 'clip'")

    all_seed_results = []
    target_vector = target_vector.to(device)

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        rng = random.Random(seed)

        # Initialize
        population = []
        for _ in range(num_outputs):
            sent = " ".join(rng.choices(safe_vocab, k=3))
            population.append(sent)
        population.sort()

        current_batch_size = batch_size

        for gen in range(generations):
            candidates = []
            if len(population) == 0:
                break

            mutations_per_parent = current_batch_size // len(population)

            # Mutation Logic
            for parent_text in population:
                candidates.append(parent_text)
                words = parent_text.split()

                for _ in range(mutations_per_parent):
                    r = rng.random()
                    if r < 0.5 and len(words) > 0:  # Replace
                        temp = words.copy()
                        temp[rng.randint(0, len(temp)-1)
                             ] = rng.choice(safe_vocab)
                        candidates.append(" ".join(temp))
                    elif r < 0.75 and len(words) < 15:  # Insert
                        temp = words.copy()
                        temp.insert(rng.randint(0, len(temp)),
                                    rng.choice(safe_vocab))
                        candidates.append(" ".join(temp))
                    elif len(words) > 1:  # Delete
                        temp = words.copy()
                        temp.pop(rng.randint(0, len(temp)-1))
                        candidates.append(" ".join(temp))

            # Filter & Sort
            unique_candidates = sorted(list(set(candidates)))

            try:
                if model_type == "blip":
                    candidate_vecs = blip_encode_text(
                        unique_candidates, processor, model, device=device,
                        batch_size=len(unique_candidates), strategy="cls"
                    )
                elif model_type == "clip":
                    candidate_vecs = clip_encode_text(
                        unique_candidates, processor, model, device=device,
                        batch_size=len(unique_candidates)
                    )

                # Scoring
                scores = torch.cosine_similarity(candidate_vecs, target_vector)
                sorted_indices = torch.argsort(scores, descending=True)

                new_population = []
                count = 0
                for idx in sorted_indices:
                    txt = unique_candidates[idx]
                    if txt not in new_population:
                        new_population.append(txt)
                        count += 1
                    if count >= num_outputs:
                        break
                population = sorted(new_population)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("OOM. Reducing batch size...")
                    current_batch_size = int(current_batch_size * 0.7)
                    torch.cuda.empty_cache()
                else:
                    raise e

        all_seed_results.extend(population)

    print("\n--- Final Ranking ---")
    unique_finalists = sorted(list(set(all_seed_results)))

    if not unique_finalists:
        return []

    if model_type == "blip":
        final_vecs = blip_encode_text(
            unique_finalists, processor, model, device=device, strategy="cls")
    else:
        final_vecs = clip_encode_text(
            unique_finalists, processor, model, device=device)

    final_scores = torch.cosine_similarity(final_vecs, target_vector)
    final_indices = torch.argsort(final_scores, descending=True)

    final_output = []
    for idx in final_indices:
        txt = unique_finalists[idx]
        score = final_scores[idx].item()
        print(f"Score {score:.4f}: {txt}")
        final_output.append(txt)

    return final_output


def calculate_clip_score_multi_seed(image_dirs, prompts, indices=None, model_name='ViT-L/14', device='cuda', aggregate_by_prompt=True):
    """
    Calculates CLIP-Score for paired images across multiple directories.
    Returns a dictionary containing means, standard deviations, 95% confidence intervals, 
    sample size, and paired t-test significance.
    Also computes a "best of two" score (max of base and improved for each pair).
    
    Args:
        image_dirs (list): List of directories containing images.
        prompts (list): List of text prompts.
        indices (list, optional): List of specific indexes to process. If None, processes all.
        model_name (str): CLIP model name.
        device (str): Device to run the model on ('cuda' or 'cpu').
        aggregate_by_prompt (bool): If True, averages scores across seeds for each prompt 
                                    before computing statistics (recommended for correct p-values).
    """
    prompt_scores = {}

    print(f"Loading CLIP model ({model_name})...")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    if indices is None:
        indices_to_process = range(len(prompts))
    else:
        indices_to_process = indices

    for directory in image_dirs:
        print(f"\n--- Processing directory: {directory} ---")
        
        for i in tqdm.tqdm(indices_to_process, desc=f"Scoring images"):
            if i >= len(prompts):
                continue
                
            prompt = prompts[i]

            path_original = os.path.join(directory, f"{i}_0.png")
            path_improved = os.path.join(directory, f"{i}_1.png")

            if not os.path.exists(path_original) or not os.path.exists(path_improved):
                continue

            try:
                pil_img_orig = Image.open(path_original).convert("RGB")
                pil_img_impr = Image.open(path_improved).convert("RGB")

                img_tensors = torch.stack([
                    preprocess(pil_img_orig), 
                    preprocess(pil_img_impr)
                ]).to(device)

                txt_tensor = clip.tokenize([prompt], truncate=True).to(device)

                with torch.no_grad():
                    img_embeds = model.encode_image(img_tensors)
                    txt_embed = model.encode_text(txt_tensor)

                    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
                    txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)

                    scores = (img_embeds * txt_embed).sum(dim=-1) * 100.0

                    score_orig = scores[0].item()
                    score_impr = scores[1].item()

                if i not in prompt_scores:
                    prompt_scores[i] = {'orig': [], 'impr': [], 'bot': []}
                
                prompt_scores[i]['orig'].append(score_orig)
                prompt_scores[i]['impr'].append(score_impr)
                prompt_scores[i]['bot'].append(max(score_orig, score_impr))

            except Exception as e:
                print(f"\nError on index {i} in {directory}: {e}")
                continue

    scores_original = []
    scores_improved = []
    scores_best_of_two = []
    deltas = []

    for i in sorted(prompt_scores.keys()):
        orig_list = prompt_scores[i]['orig']
        impr_list = prompt_scores[i]['impr']
        bot_list = prompt_scores[i]['bot']

        if aggregate_by_prompt and len(orig_list) > 0:
            mean_orig = sum(orig_list) / len(orig_list)
            mean_impr = sum(impr_list) / len(impr_list)
            mean_bot = sum(bot_list) / len(bot_list)
            
            scores_original.append(mean_orig)
            scores_improved.append(mean_impr)
            scores_best_of_two.append(mean_bot)
            deltas.append(mean_impr - mean_orig)
        else:
            scores_original.extend(orig_list)
            scores_improved.extend(impr_list)
            scores_best_of_two.extend(bot_list)
            deltas.extend([impr - orig for impr, orig in zip(impr_list, orig_list)])

    n = len(scores_original)
    results = {"n": n, "aggregated_by_prompt": aggregate_by_prompt}

    if n > 1:
        def get_stats(score_list):
            mean = statistics.mean(score_list)
            std = statistics.stdev(score_list)
            se = std / math.sqrt(n)
            ci_lower, ci_upper = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
            return {"mean": mean, "std": std, "ci_95": (ci_lower, ci_upper)}

        results["original"] = get_stats(scores_original)
        results["improved"] = get_stats(scores_improved)
        results["best_of_two"] = get_stats(scores_best_of_two)
        results["delta"] = get_stats(deltas)
        
        t_stat, p_value = stats.ttest_rel(scores_improved, scores_original)
        results["significance"] = {
            "test": "Paired t-test",
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant_0.05": p_value < 0.05
        }
        
    elif n == 1:
        results["original"] = {"mean": scores_original[0], "std": 0, "ci_95": (scores_original[0], scores_original[0])}
        results["improved"] = {"mean": scores_improved[0], "std": 0, "ci_95": (scores_improved[0], scores_improved[0])}
        results["best_of_two"] = {"mean": scores_best_of_two[0], "std": 0, "ci_95": (scores_best_of_two[0], scores_best_of_two[0])}
        results["delta"] = {"mean": deltas[0], "std": 0, "ci_95": (deltas[0], deltas[0])}
        results["significance"] = {"test": "Paired t-test", "t_statistic": None, "p_value": None, "is_significant_0.05": False}
    else:
        results["original"] = {"mean": 0, "std": 0, "ci_95": (0, 0)}
        results["improved"] = {"mean": 0, "std": 0, "ci_95": (0, 0)}
        results["best_of_two"] = {"mean": 0, "std": 0, "ci_95": (0, 0)}
        results["delta"] = {"mean": 0, "std": 0, "ci_95": (0, 0)}
        results["significance"] = {"test": "Paired t-test", "t_statistic": None, "p_value": None, "is_significant_0.05": False}

    return results


def calculate_blip_itm_score_multi_seed(image_dirs, prompts, processor, model, indices=None, device="cuda", aggregate_by_prompt=True):
    """
    Calculates BLIP-2 Image-Text Alignment scores for paired images across multiple directories.
    Returns a dictionary containing means, standard deviations, 95% confidence intervals, 
    sample size, and paired t-test significance.
    Also computes a "best of two" score (max of base and improved for each pair).
    
    Args:
        image_dirs (list): List of directories containing image pairs.
        prompts (list): List of text prompts corresponding to the image indices.
        processor: The associated BLIP-2 processor.
        model: The BLIP-2 model (Blip2ForImageTextRetrieval).
        indices (list, optional): List of specific indexes to process. If None, processes all.
        device (str): Device to run inference on (default: "cuda").
        aggregate_by_prompt (bool): If True, averages scores across seeds for each prompt 
                                    before computing statistics (recommended for correct p-values).
        
    Returns:
        dict: Dictionary of statistics including CIs and significance.
    """
    prompt_scores = {}

    model.eval()

    if indices is None:
        indices_to_process = range(len(prompts))
    else:
        indices_to_process = indices

    for directory in image_dirs:
        print(f"\n--- Processing directory: {directory} ---")
        
        for i in tqdm.tqdm(indices_to_process, desc=f"Scoring images"):
            if i >= len(prompts):
                continue
                
            prompt = prompts[i]

            path_original = os.path.join(directory, f"{i}_0.png")
            path_improved = os.path.join(directory, f"{i}_1.png")

            if not os.path.exists(path_original) or not os.path.exists(path_improved):
                continue

            try:
                pil_img_orig = Image.open(path_original).convert("RGB")
                pil_img_impr = Image.open(path_improved).convert("RGB")

                with torch.no_grad():
                    inputs = processor(
                        images=[pil_img_orig, pil_img_impr],
                        text=[prompt, prompt],
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device)

                    outputs = model(**inputs)

                    if hasattr(outputs, "logits_per_image"):
                        logits_diag = outputs.logits_per_image.diagonal()
                        match_probs = torch.sigmoid(logits_diag)
                    
                    elif hasattr(outputs, "logits"):
                        if outputs.logits.dim() == 2 and outputs.logits.shape[0] == outputs.logits.shape[1]:
                             match_probs = torch.sigmoid(outputs.logits.diagonal())
                        else:
                             probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                             match_probs = probs[:, 1]
                    else:
                        raise ValueError("Model output does not contain 'logits_per_image' or 'logits'.")

                    score_orig = match_probs[0].item()
                    score_impr = match_probs[1].item()

                if i not in prompt_scores:
                    prompt_scores[i] = {'orig': [], 'impr': [], 'bot': []}
                
                prompt_scores[i]['orig'].append(score_orig)
                prompt_scores[i]['impr'].append(score_impr)
                prompt_scores[i]['bot'].append(max(score_orig, score_impr))

            except Exception as e:
                print(f"\nError on index {i} in {directory}: {e}")
                continue

    scores_original = []
    scores_improved = []
    scores_best_of_two = []
    deltas = []

    for i in sorted(prompt_scores.keys()):
        orig_list = prompt_scores[i]['orig']
        impr_list = prompt_scores[i]['impr']
        bot_list = prompt_scores[i]['bot']

        if aggregate_by_prompt and len(orig_list) > 0:
            mean_orig = sum(orig_list) / len(orig_list)
            mean_impr = sum(impr_list) / len(impr_list)
            mean_bot = sum(bot_list) / len(bot_list)
            
            scores_original.append(mean_orig)
            scores_improved.append(mean_impr)
            scores_best_of_two.append(mean_bot)
            deltas.append(mean_impr - mean_orig)
        else:
            scores_original.extend(orig_list)
            scores_improved.extend(impr_list)
            scores_best_of_two.extend(bot_list)
            deltas.extend([impr - orig for impr, orig in zip(impr_list, orig_list)])

    n = len(scores_original)
    results = {"n": n, "aggregated_by_prompt": aggregate_by_prompt}

    if n > 1:
        def get_stats(score_list):
            mean = statistics.mean(score_list)
            std = statistics.stdev(score_list)
            se = std / math.sqrt(n)
            ci_lower, ci_upper = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
            return {"mean": mean, "std": std, "ci_95": (ci_lower, ci_upper)}

        results["original"] = get_stats(scores_original)
        results["improved"] = get_stats(scores_improved)
        results["best_of_two"] = get_stats(scores_best_of_two)
        results["delta"] = get_stats(deltas)
        
        t_stat, p_value = stats.ttest_rel(scores_improved, scores_original)
        results["significance"] = {
            "test": "Paired t-test",
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant_0.05": p_value < 0.05
        }
        
    elif n == 1:
        results["original"] = {"mean": scores_original[0], "std": 0, "ci_95": (scores_original[0], scores_original[0])}
        results["improved"] = {"mean": scores_improved[0], "std": 0, "ci_95": (scores_improved[0], scores_improved[0])}
        results["best_of_two"] = {"mean": scores_best_of_two[0], "std": 0, "ci_95": (scores_best_of_two[0], scores_best_of_two[0])}
        results["delta"] = {"mean": deltas[0], "std": 0, "ci_95": (deltas[0], deltas[0])}
        results["significance"] = {"test": "Paired t-test", "t_statistic": None, "p_value": None, "is_significant_0.05": False}
    else:
        results["original"] = {"mean": 0, "std": 0, "ci_95": (0, 0)}
        results["improved"] = {"mean": 0, "std": 0, "ci_95": (0, 0)}
        results["best_of_two"] = {"mean": 0, "std": 0, "ci_95": (0, 0)}
        results["delta"] = {"mean": 0, "std": 0, "ci_95": (0, 0)}
        results["significance"] = {"test": "Paired t-test", "t_statistic": None, "p_value": None, "is_significant_0.05": False}

    return results


def evaluate_kid_across_seeds(seed_directories):
    """
    Calculates the KID score between baseline (0) and method (1) images across multiple seeds.
    
    Args:
        seed_directories (list of str): List of paths to your seed directories.
    
    Returns:
        mean_kid (float): Average KID across all seeds.
        std_kid (float): Standard deviation of KID.
    """
    kid_scores = []
    
    for seed_dir in seed_directories:
        print(f"\nProcessing directory: {seed_dir}...")
        
        with tempfile.TemporaryDirectory() as temp_base, tempfile.TemporaryDirectory() as temp_method:
            
            file_count = 0
            for filename in os.listdir(seed_dir):
                if not filename.endswith('.png'):
                    continue
                
                name_parts = filename.replace('.png', '').split('_')
                if len(name_parts) != 2:
                    continue
                
                i, j = name_parts[0], name_parts[1]
                src_path = os.path.join(seed_dir, filename)
                
                if j == '0':
                    shutil.copy(src_path, os.path.join(temp_base, filename))
                    file_count += 1
                elif j == '1':
                    shutil.copy(src_path, os.path.join(temp_method, filename))
                    file_count += 1
                    
            print(f"Sorted {file_count} images into temporary comparison folders.")
            
            print("Calculating KID score...")
            score = fid.compute_kid(temp_base, temp_method)
            kid_scores.append(score)
            print(f"KID for {os.path.basename(seed_dir)}: {score:.5f}")
            
    mean_kid = np.mean(kid_scores)
    std_kid = np.std(kid_scores)
    
    print("\n" + "="*40)
    print("FINAL RESULTS (Variance Method)")
    print("="*40)
    print(f"Average KID Score: {mean_kid:.5f} ± {std_kid:.5f}")
    print("="*40)
    
    return mean_kid, std_kid, kid_scores
