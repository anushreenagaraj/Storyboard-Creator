from diffusers import StableDiffusionPipeline
import torch

print("Downloading Stable Diffusion v1-4...")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16" if torch.cuda.is_available() else "main",
    torch_dtype=torch.float32,
    use_auth_token=True  # Requires login via `huggingface-cli login`
)

pipe.save_pretrained("local_sd_model")

print("✅ Model saved to local_sd_model folder.")
