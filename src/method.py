import os

import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from src import utils


def get_clean_embeddings_qr(pipe, prompt, keywords, weights, alpha=0.5, threshold=0.01, min_rank=3, max_rank=15, normalize=False, preserve_start=True):
    """
    Encodes the prompt and removes weighted negative concepts using 
    QR Decomposition to determine the subtraction rank.
    
    Args:
        preserve_start (bool): If True, restores the Start-of-Sentence token 
                               (index 0) of the sequence embedding after cleaning.
    """
    # 1. Encode Target Prompt
    with torch.no_grad():
        encoded = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=pipe.device)

    # Detect Model and Unpack
    target_meta = None  # For Flux

    if len(encoded) == 4:
        # SDXL
        target_seq = encoded[0]
        target_pool = encoded[2]
    elif len(encoded) == 3:
        # FLUX
        target_seq = encoded[0]
        target_pool = encoded[1]
        target_meta = encoded[2]
    else:
        raise ValueError("Unknown pipeline type.")

    target_seq = target_seq.detach()
    target_pool = target_pool.detach()

    # 2. Build Weighted Negative Accumulators
    # Create zero-filled tensors to sum up the negative keywords
    neg_seq_accum = torch.zeros_like(target_seq)
    neg_pool_accum = torch.zeros_like(target_pool)

    with torch.no_grad():
        for i, kw in enumerate(keywords):
            kw_enc = pipe.encode_prompt(
                prompt=kw, prompt_2=kw, device=pipe.device)

            # Unpack keyword
            if len(kw_enc) == 4:
                k_seq, k_pool = kw_enc[0], kw_enc[2]
            else:
                k_seq, k_pool = kw_enc[0], kw_enc[1]

            # Add to accumulator
            w = weights[i]
            neg_seq_accum += (k_seq * w)
            neg_pool_accum += (k_pool * w)

    # 3. Apply QR Subtraction
    # We only preserve start for Sequence embeddings. Pooled embeddings are 1D vectors per batch.
    cleaned_seq = apply_qr_subtraction(
        target_seq, neg_seq_accum, alpha, threshold=threshold, min_rank=min_rank, max_rank=max_rank, normalize=normalize, preserve_start=preserve_start)

    cleaned_pool = apply_qr_subtraction(
        target_pool, neg_pool_accum, alpha, threshold=threshold, min_rank=min_rank, max_rank=max_rank, normalize=normalize, preserve_start=False)

    # Clean up memory
    del neg_seq_accum
    del neg_pool_accum

    # Return
    if target_meta is not None:
        return cleaned_seq, cleaned_pool, target_meta
    else:
        return cleaned_seq, cleaned_pool


def apply_qr_subtraction(target, negative, alpha=1.0, threshold=0.01, min_rank=3, max_rank=15, normalize=True, preserve_start=False):
    """
    Subtracts negative concepts using QR decomposition to find dominant ranks.
    """
    # Use float32 for math precision
    t_f32 = target.float()
    n_f32 = negative.float()

    # Store original norm for final restoration
    if normalize:
        orig_norm = torch.norm(t_f32, dim=-1, keepdim=True)

    # Save Start Token
    # Only applicable if this is a sequence embedding (Batch, Tokens, Dim)
    saved_start = None
    if preserve_start and t_f32.dim() == 3:
        saved_start = t_f32[:, 0, :].clone()

    output = torch.zeros_like(t_f32)
    batch_size = t_f32.shape[0]

    for b in range(batch_size):
        t_item = t_f32[b]
        n_item = n_f32[b]

        # Handle 1D inputs (Pooled embeds)
        is_1d = (t_item.dim() == 1)
        if is_1d:
            t_item = t_item.unsqueeze(0)
            n_item = n_item.unsqueeze(0)

        # QR Decomposition
        # Transpose (.T) to analyze feature correlation
        Q, R = torch.linalg.qr(n_item.T, mode='reduced')
        diag_R = torch.abs(torch.diag(R))

        # Calculate Rank
        if diag_R[0] > 1e-8:
            ratios = diag_R / diag_R[0]
            rank = (ratios > threshold).sum().item()
            rank = max(min_rank, min(rank, max_rank))
        else:
            rank = min_rank

        # Reduce Negative Embedding
        Q_reduced = Q[:, :rank]
        n_reduced = (n_item @ Q_reduced) @ Q_reduced.T

        # Flatten & Subtract
        t_flat = t_item.reshape(-1)
        n_flat = n_reduced.reshape(-1)

        # Dynamic Scale
        norm_t = t_flat.norm().clamp_min(1e-8)
        t_normed = t_flat / norm_t
        n_scaled = n_flat / norm_t

        dot = torch.dot(t_normed, n_scaled)
        dynamic_scale = 1 + (alpha * dot)

        # Apply Subtraction
        result_flat = (t_flat * dynamic_scale) - (alpha * n_flat)

        # Restore Shape
        result = result_flat.reshape(t_item.shape)
        if is_1d:
            result = result.squeeze(0)

        output[b] = result

    # L2 Normalization (Intensity Match)
    if normalize:
        current_norm = torch.norm(output, dim=-1, keepdim=True)
        output = output * (orig_norm / (current_norm + 1e-8))

    # Restore Start Token
    if saved_start is not None:
        output[:, 0, :] = saved_start

    return output.to(target.dtype)


def find_nearest_keywords(target_vec, emb_matrix, keywords_list, k=5, temperature=0.5):
    """
    Finds top-k keywords matching the target vector and calculates softmax weights.
    """
    target_vec = target_vec.to(emb_matrix.device)

    # Cosine Similarity
    scores = torch.matmul(target_vec, emb_matrix.T).squeeze()

    # Top-K
    topk = torch.topk(scores, k=k)
    found_keywords = [keywords_list[i] for i in topk.indices.cpu().numpy()]

    # Weights with Temperature
    scores_with_temp = topk.values / temperature
    weights = F.softmax(scores_with_temp, dim=0)

    return found_keywords, weights


def get_drift_keywords(
    pipe,
    prompt,
    seed,
    encoding_processor,
    encoding_model,
    encoding_model_type="blip",
    blip_text_encoding_strategy="mean",
    num_inference_steps=30,
    guidance_scale=7.5,
    emb_matrix=None,
    keywords_list=None,
    topk=5,
    img_path=None
):
    # 1. Setup Encoding Functions
    if encoding_model_type == "blip":

        def encode_txt_fn(txt):
            return utils.blip_encode_text(
                [txt],
                encoding_processor,
                encoding_model[0],
                strategy=blip_text_encoding_strategy,
            )

        def encode_img_fn(img):
            return utils.blip_encode_image(img, encoding_processor, encoding_model[1])

    elif encoding_model_type == "clip":

        def encode_txt_fn(txt):
            return utils.clip_encode_text([txt], encoding_processor, encoding_model)

        def encode_img_fn(img):
            return utils.clip_encode_image(img, encoding_processor, encoding_model)

    elif encoding_model_type == "siglip2":

        def encode_txt_fn(txt):
            return utils.siglip2_encode_text([txt], encoding_processor, encoding_model)

        def encode_img_fn(img):
            return utils.siglip2_encode_image(img, encoding_processor, encoding_model)

    elif encoding_model_type == "evaclip":

        def encode_txt_fn(txt):
            return utils.evaclip_encode_text(
                [txt], encoding_processor[1], encoding_model
            )

        def encode_img_fn(img):
            return utils.evaclip_encode_image(
                img, encoding_processor[0], encoding_model
            )

    else:
        raise ValueError(
            "encoding_model_type must be 'blip', 'clip', 'siglip2', or 'evaclip'"
        )

    if img_path is None:
        # Generate Normal Image
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        img_normal = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    else:
        img_normal = Image.open(img_path).convert("RGB")

    # Analyze Error
    img_emb = encode_img_fn(img_normal)
    text_emb = encode_txt_fn(prompt)

    # We move text_emb to the device of img_emb (usually CUDA)
    text_emb = text_emb.to(img_emb.device)

    drift_vec = img_emb - text_emb

    drift_vec_orth = utils.get_vector_rejection(text_emb, drift_vec, normalize=False)

    # Find keywords using matrix search
    drift_keywords, drift_keywords_weights = find_nearest_keywords(
        drift_vec_orth, emb_matrix, keywords_list, k=topk
    )

    return (drift_keywords, drift_keywords_weights, img_normal)


def run_intervention_experiment(
    pipe,
    prompts,
    seeds,
    save_dir,
    get_modified_embeddings,
    encoding_processor,
    encoding_model,
    encoding_model_type="blip",
    blip_text_encoding_strategy="mean",
    num_inference_steps=30,
    guidance_scale=7.5,
    mode="keywords",           # Options: "keywords", "genetic"
    emb_matrix=None,           # Required for "keywords" mode
    keywords_list=None,        # Required for "keywords" mode
    keywords_topk=5,
    keywords_weights_temp=0.5,
    safe_vocab=None,           # Required for "genetic" mode
    genetic_generations=30,
    genetic_num_outputs=3,
    genetic_seeds=[42, 100, 999],
    genetic_batch_size=500,
    ablation_mode=None         # None or "FVC"
):

    # Setup Encoding Functions
    if encoding_model_type == "blip":
        def encode_txt_fn(txt): return utils.blip_encode_text(
            [txt], encoding_processor, encoding_model[0], strategy=blip_text_encoding_strategy)

        def encode_img_fn(img): return utils.blip_encode_image(
            img, encoding_processor, encoding_model[1])
    elif encoding_model_type == "clip":
        def encode_txt_fn(txt): return utils.clip_encode_text(
            [txt], encoding_processor, encoding_model)

        def encode_img_fn(img): return utils.clip_encode_image(
            img, encoding_processor, encoding_model)     
    elif encoding_model_type == "siglip2":
        def encode_txt_fn(txt): return utils.siglip2_encode_text(
            [txt], encoding_processor, encoding_model)

        def encode_img_fn(img): return utils.siglip2_encode_image(
            img, encoding_processor, encoding_model)
    elif encoding_model_type == "evaclip":
        def encode_txt_fn(txt): return utils.evaclip_encode_text(
            [txt], encoding_processor[1], encoding_model)

        def encode_img_fn(img): return utils.evaclip_encode_image(
            img, encoding_processor[0], encoding_model)
    else:
        raise ValueError("encoding_model_type must be 'blip', 'clip', 'siglip2', or 'evaclip'")

    # Loop over Seeds
    for run_idx, seed in enumerate(seeds):
        # Folder structure: save_dir_0, save_dir_1...
        current_save_path = f"{save_dir}_{run_idx}"
        os.makedirs(current_save_path, exist_ok=True)

        print(
            f"\n[Run {run_idx+1}/{len(seeds)}] Seed {seed} -> {current_save_path}")

        # Loop over Prompts
        for k, original_prompt in enumerate(tqdm.tqdm(prompts, desc="Prompts")):

            # Generate Normal Image
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            img_normal = pipe(
                prompt=original_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

            # Analyze Error
            img_emb = encode_img_fn(img_normal)
            text_emb = encode_txt_fn(original_prompt)

            text_emb = text_emb.to(img_emb.device)

            drift_vec = img_emb - text_emb

            if ablation_mode is None:
                drift_vec_orth = utils.get_vector_rejection(
                    text_emb, drift_vec, normalize=False)
            elif ablation_mode == "FVC":
                drift_vec_orth = drift_vec
            else:
                raise ValueError("ablation_mode must be None or 'FVC'")

            # Determine Intervention
            if mode == "keywords":
                # Find keywords using matrix search
                drift_keywords, drift_keywords_weights = find_nearest_keywords(
                    drift_vec_orth, emb_matrix, keywords_list, k=keywords_topk, temperature=keywords_weights_temp)

                # Get new embeddings
                prompt_embeds, pooled_embeds = get_modified_embeddings(
                    pipe, original_prompt, drift_keywords, drift_keywords_weights)[0:2]

            elif mode == "genetic":
                # Find text using genetic algorithm
                drift_texts = utils.genetic_text_search(
                    target_vector=drift_vec_orth,
                    processor=encoding_processor,
                    model=encoding_model,
                    safe_vocab=safe_vocab,
                    model_type=encoding_model_type,
                    seeds=genetic_seeds,
                    num_outputs=genetic_num_outputs,
                    generations=genetic_generations,
                    batch_size=genetic_batch_size
                )

                # Uniform weights
                drift_texts_weights = [1 / len(drift_texts)] * len(drift_texts)

                # Get new embeddings
                prompt_embeds, pooled_embeds = get_modified_embeddings(
                    pipe, original_prompt, drift_texts, drift_texts_weights)[0:2]

            else:
                raise ValueError("mode must be 'keywords' or 'genetic'")

            # Generate with Method
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            img_my_method = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

            # Save
            img_normal.save(os.path.join(current_save_path, f"{k}_0.png"))
            img_my_method.save(os.path.join(current_save_path, f"{k}_1.png"))
