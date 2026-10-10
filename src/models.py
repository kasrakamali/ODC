import torch
import open_clip
from diffusers import FluxPipeline, StableDiffusionXLPipeline
from transformers import (AutoModel, AutoProcessor, 
                          Blip2ForImageTextRetrieval,
                          Blip2TextModelWithProjection,
                          Blip2VisionModelWithProjection, CLIPModel,
                          CLIPProcessor)

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
BLIP_MODEL_ID = "Salesforce/blip2-itm-vit-g"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
SIGLIP2_MODEL_ID = "google/siglip2-so400m-patch14-224"
EVBCLIP_HF_ID = "hf-hub:timm/eva_giant_patch14_plus_clip_224.merged2b_s11b_b114k"


def load_sdxl_pipeline(device="cuda", compile_enabled=False):
    """
    Loads the Stable Diffusion XL pipeline with standard research settings.
    """
    print(f"Loading model: {SDXL_MODEL_ID}...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    pipe = pipe.to(device)

    if compile_enabled:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    return pipe


def load_flux_pipeline(device="cuda", compile_enabled=False):
    """
    Loads FLUX.1-schnell pipeline.
    Note: Uses bfloat16 for optimal performance/compatibility.
    """
    print(f"Loading FLUX: {FLUX_MODEL_ID}...")

    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=torch.bfloat16
    )

    pipe = pipe.to(device)
    
    if compile_enabled:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    
    return pipe


def load_blip2_retrieval_model(device="cuda"):
    """
    Loads the full BLIP-2 model for Image-Text Matching (ITM) scoring.
    Different from the individual components used for feature extraction.
    """
    print(f"Loading BLIP-2 ITM Model: {BLIP_MODEL_ID}...")

    model = Blip2ForImageTextRetrieval.from_pretrained(
        BLIP_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(BLIP_MODEL_ID)

    return processor, model


def load_blip2_components():
    """
    Loads the BLIP-2 processor, text model, and vision model 
    for Image-Text Matching (ITM).
    
    Returns:
        tuple: (processor, text_model, vision_model)
    """
    print(f"Loading BLIP-2 components from {BLIP_MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(BLIP_MODEL_ID)

    text_model = Blip2TextModelWithProjection.from_pretrained(
        BLIP_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    vision_model = Blip2VisionModelWithProjection.from_pretrained(
        BLIP_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    print("BLIP-2 components loaded successfully.")

    return processor, text_model, vision_model


def load_clip_components():
    """
    Loads CLIP processor and model for image-text similarity.
    Returns:
        tuple: (processor, model)
    """
    print(f"Loading CLIP: {CLIP_MODEL_ID}...")

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    return processor, model


def load_siglip2_components():
    """
    Loads SigLIP2 processor and model for image-text similarity.
    Returns:
        tuple: (processor, model)
    """
    print(f"Loading SigLIP2: {SIGLIP2_MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)

    model = AutoModel.from_pretrained(
        SIGLIP2_MODEL_ID,
        device_map="auto"
        ).eval()

    return processor, model


def load_evaclip_components():
    """
    Loads EVA-CLIP model, preprocess transform, and tokenizer for image-text similarity.
    Returns:
        tuple: (model, preprocess_val, tokenizer)
    """
    print(f"Loading EVA-CLIP: {EVBCLIP_HF_ID}...")

    model, _, preprocess_val = open_clip.create_model_and_transforms(EVBCLIP_HF_ID)
    
    tokenizer = open_clip.get_tokenizer(EVBCLIP_HF_ID)

    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, preprocess_val, tokenizer
