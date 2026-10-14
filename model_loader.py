import os
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
import torch

MODEL_NAME = "runwayml/stable-diffusion-inpainting"

def get_pipeline():
    """Load the inpainting pipeline. Caches it in memory."""
    if not hasattr(get_pipeline, 'pipeline'):
        print("Loading Model... This may take 1-2 minutes on first boot.")
        
        # Scheduler set karo speed ke liye
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            MODEL_NAME, subfolder="scheduler"
        )
        
        # Load Model
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_NAME,
            scheduler=scheduler,
            torch_dtype=torch.float32 # Use float32 for CPU stability
        )
        
        # Optional: Move to GPU if available (Render Free Tier has limited GPU access via env vars sometimes, but usually CPU only for free)
        # If you have GPU enabled in Render settings, uncomment below:
        # pipe.to("cuda")
        
        get_pipeline.pipeline = pipe
        print("Model Loaded Successfully.")
    
    return get_pipeline.pipeline

def generate_image(image_path, mask_path, prompt="blurry face, clean skin", negative_prompt="ugly, deformed, noisy"):
    """Generate the undressed image using ControlNet-like inpainting logic."""
    pipe = get_pipeline()
    
    # Parameters for better quality
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": 30, # Higher steps = better detail but slower
        "guidance_scale": 7.5,
        "strength": 0.85, # How much of the original image to keep/change
        "width": 512,
        "height": 512
    }
    
    # Load Images
    from PIL import Image
    init_image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("L")
    
    # Resize to standard size for faster processing (optional, can be removed for higher res)
    init_image = init_image.resize((512, 512))
    mask_image = mask_image.resize((512, 512))
    
    # Generate
    result = pipe(**params, image=init_image, mask_image=mask_image)
    
    # Save Result
    output_img = result.images[0]
    output_path = "output.png"
    output_img.save(output_path)
    
    return output_path
