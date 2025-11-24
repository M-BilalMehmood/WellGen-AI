#!/usr/bin/env python3
"""Generate wellness images using Stable Diffusion for WellGen AI."""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os
from datetime import datetime

print("="*60)
print("WellGen AI - Image Generation")
print("="*60)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

if device == "cpu":
    print("\nWARNING: Running on CPU. Generation will be slow (~2-3 min per image)")
    print("For faster generation, use WSL with CUDA support.")

print("\nLoading Stable Diffusion v1.5...")
print("(First run will download ~4GB model)")

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Use faster scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

if device == "cuda":
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

print("\nModel loaded successfully!")

# Create output directory
os.makedirs("generated_images", exist_ok=True)

# Wellness-related prompts
wellness_prompts = [
    "A healthy balanced meal plate with colorful vegetables, lean protein, whole grains, and fruits, professional food photography, high quality, detailed",
    "A nutritious breakfast bowl with oatmeal, fresh berries, nuts, and honey, morning light, appetizing, food photography",
    "A fresh green smoothie bowl topped with granola, sliced banana, and chia seeds, vibrant colors, healthy lifestyle",
    "A colorful salad with mixed greens, cherry tomatoes, avocado, and grilled chicken, professional food photography",
    "Healthy meal prep containers with balanced portions of protein, vegetables, and grains, organized, clean",
    "A person doing yoga in a peaceful natural setting, wellness lifestyle, morning meditation, serene atmosphere",
    "Fitness concept with dumbbells, water bottle, fresh fruits, and towel, healthy lifestyle, gym motivation",
    "A nutrition facts infographic showing balanced macronutrients, clean design, educational, wellness information"
]

print("\n" + "="*60)
print("Generating Wellness Images")
print("="*60)
print(f"\nGenerating {len(wellness_prompts)} images...")
print("Each image takes ~10-20 seconds on GPU\n")

for idx, prompt in enumerate(wellness_prompts, 1):
    print(f"[{idx}/{len(wellness_prompts)}] Generating: {prompt[:60]}...")
    
    # Generate image
    image = pipe(
        prompt,
        num_inference_steps=30,  # Good balance of speed/quality
        guidance_scale=7.5,
        height=512,
        width=512
    ).images[0]
    
    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_images/wellness_{idx:02d}_{timestamp}.png"
    image.save(filename)
    print(f"    ✓ Saved: {filename}\n")

print("\n" + "="*60)
print("Generation Complete!")
print("="*60)
print(f"\nGenerated {len(wellness_prompts)} wellness images")
print("Location: generated_images/")
print("\nYou can:")
print("1. Use these for your WellGen AI chatbot responses")
print("2. Generate custom images by editing wellness_prompts list")
print("3. Integrate with chat.py to show meal suggestions")
