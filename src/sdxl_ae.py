"""
Attend-and-Excite implementation adapted for Stable Diffusion XL.
Based on the original implementation by Chefer et al. (SIGGRAPH 2023):
https://github.com/yuval-alaluf/Attend-and-Excite
"""

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
import math
from tqdm import tqdm
import spacy


def get_auto_indices(prompt: str, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading Spacy model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(prompt)
    subjects = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    if not subjects:
        print("No subjects found.")
        return []

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    input_ids = tokenizer(prompt).input_ids
    
    indices = []
    print(f"\nAuto-Detected Subjects: {subjects}")
    for i, token_id in enumerate(input_ids):
        token_str = tokenizer.decode(token_id).strip()
        for subj in subjects:
            if token_str.lower() in subj.lower() and len(token_str) > 1:
                indices.append(i)
                print(f"  Found '{token_str}' at index {i} (Part of '{subj}')")
    
    return sorted(list(set(indices)))


class GaussianSmoothing(torch.nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super().__init__()
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel_2d)
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, input):
        return F.conv2d(input, weight=self.weight.to(input.device), padding=self.padding, groups=self.groups)


class AttendExciteSDXLAttnProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_probs = None 

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None: encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        if attention_mask is not None: scores = scores + attention_mask

        attn_probs = F.softmax(scores, dim=-1)

        if encoder_hidden_states is not None and hidden_states.shape[1] != encoder_hidden_states.shape[1]:
            self.attention_probs = attn_probs

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        return hidden_states


def get_loss_and_update(pipe, latents, t, prompt_embeds, add_text_embeds, add_time_ids, 
                        token_indices, smoother, attn_res, device, text_len):
    pipe.unet(latents.half(), t, encoder_hidden_states=prompt_embeds, added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids})

    attention_maps = []
    for name, processor in pipe.unet.attn_processors.items():
        if isinstance(processor, AttendExciteSDXLAttnProcessor) and processor.attention_probs is not None:
            probs = processor.attention_probs
            if probs.shape[2] == attn_res**2: attention_maps.append(probs)
    
    loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    max_attns = []

    if len(attention_maps) > 0:
        agg_map = torch.cat(attention_maps, dim=0).mean(dim=0).mean(dim=0).float()
        agg_map = agg_map.view(attn_res, attn_res, -1) 
        
        valid_map = agg_map[:, :, 1 : text_len - 1]
        
        valid_map = valid_map * 100.0
        valid_map = F.softmax(valid_map, dim=-1)
        
        losses = []
        for original_idx in token_indices:
            shifted_idx = original_idx - 1
            if shifted_idx < 0 or shifted_idx >= valid_map.shape[-1]: continue

            token_map = valid_map[:, :, shifted_idx]
            
            token_map = token_map.unsqueeze(0).unsqueeze(0)
            token_map = F.pad(token_map, (1, 1, 1, 1), mode='reflect')
            
            smoothed_map = smoother(token_map).squeeze()
            
            curr_max = smoothed_map.max()
            max_attns.append(curr_max.item())
            losses.append(torch.max(torch.tensor(0.0, device=device), 1.0 - curr_max))
        
        if len(losses) > 0:
            loss = torch.max(torch.stack(losses))

    return loss, max_attns


def generate_sdxl_attend_and_excite(
    prompt: str,
    token_indices: list[int],
    seed: int = 42,
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    steps: int = 30,
    max_iter_to_alter: int = 25,
    scale_factor: float = 20.0, 
    attn_res: int = 32,
    refinement_thresholds: dict = {0: 0.1, 10: 0.5, 20: 0.8},
    max_refinement_steps: int = 5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(device)
    
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name: 
            attn_procs[name] = AttendExciteSDXLAttnProcessor()
        else:
            attn_procs[name] = AttnProcessor2_0()
    pipe.unet.set_attn_processor(attn_procs)

    smoother = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5).to(device).float()
    generator = torch.Generator(device).manual_seed(seed)
    
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = \
        pipe.encode_prompt(prompt=prompt, device=device)
    
    text_len = len(pipe.tokenizer_2(prompt).input_ids)

    pipe.scheduler.set_timesteps(steps, device=device)
    
    latents = pipe.prepare_latents(
        1, pipe.unet.config.in_channels, 1024, 1024, 
        torch.float32, device, generator, None 
    )

    add_text_embeds = pooled_prompt_embeds
    projection_dim = pipe.text_encoder_2.config.projection_dim
    add_time_ids = pipe._get_add_time_ids(
        (1024, 1024), (0, 0), (1024, 1024), 
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=projection_dim
    ).to(device)

    print(f"Generating: '{prompt}'")
    
    for i, t in enumerate(tqdm(pipe.scheduler.timesteps)):
        
        if i < max_iter_to_alter:
            required_threshold = refinement_thresholds.get(i, None)
            current_max_iters = max_refinement_steps if required_threshold is not None else 1
            
            for refine_step in range(current_max_iters):
                latents = latents.detach().requires_grad_(True)
                
                loss, max_attns = get_loss_and_update(
                    pipe, latents, t, prompt_embeds, add_text_embeds, add_time_ids,
                    token_indices, smoother, attn_res, device, text_len
                )
                
                worst_attention = min(max_attns) if max_attns else 1.0
                if required_threshold is not None:
                     if worst_attention >= required_threshold:
                         break 
                
                if loss > 0:
                    grad = torch.autograd.grad(loss, latents)[0]
                    if not torch.isnan(grad).any():
                        grad_norm = grad.norm()
                        if grad_norm > 1e-5:
                            grad = grad / grad_norm 
                        decay_rate = 1.0 - (i / steps) * 0.5
                        current_scale = scale_factor * decay_rate
                        latents = latents - current_scale * grad
                
                pipe.unet.zero_grad()
                for p in pipe.unet.attn_processors.values():
                    if isinstance(p, AttendExciteSDXLAttnProcessor):
                        p.attention_probs = None
        
        latents = latents.detach()
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input.half(), t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                added_cond_kwargs={
                    "text_embeds": torch.cat([negative_pooled_prompt_embeds, add_text_embeds]),
                    "time_ids": torch.cat([add_time_ids, add_time_ids])
                }
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred.float(), t, latents).prev_sample

    print("Decoding...")
    with torch.no_grad():
        latents = torch.clamp(latents, -30, 30)
        pipe.vae.to(dtype=torch.float32)
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    
    return image
