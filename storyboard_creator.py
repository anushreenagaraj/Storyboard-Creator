from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# Load the model (will download ~4GB first time)
print("⏳ Loading Stable Diffusion model (runwayml/stable-diffusion-v1-5)...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cpu")  # Use CPU

# Create output directory
os.makedirs("storyboard_frames", exist_ok=True)

# Input prompts
num_frames = int(input("Enter number of storyboard frames: "))
prompts = [input(f"Prompt for frame {i+1}: ") for i in range(num_frames)]

# Generate and save images
for i, prompt in enumerate(prompts):
    print(f"\n🎨 Generating Frame {i+1}: {prompt} ...")
    image = pipe(prompt, num_inference_steps=25).images[0]

    filename = f"storyboard_frames/frame_{i+1}.png"
    image.save(filename)
    print(f"✅ Saved: {filename}")

    # Optional: display
    image.show()

# First time download (only needed once)
pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_auth_token=False
)
pipeline.save_pretrained("local_sd_model")