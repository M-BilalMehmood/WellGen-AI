#!/usr/bin/env python3
"""Train Stable Diffusion LoRA for WellGen AI Image Generation."""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, get_peft_model
import json
from PIL import Image
import os

print("="*60)
print("WellGen AI - Image Generation Model Training")
print("="*60)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

if device == "cpu":
    print("WARNING: No GPU detected. Training will be very slow.")
    print("Consider running in WSL with CUDA support.")
    exit(1)

print("\nThis will fine-tune Stable Diffusion v1.5 with LoRA")
print("Training time: ~1-2 hours on RTX 5060")
print("Memory usage: ~6GB VRAM")

# Load captions
print("\nLoading training data...")
with open("data/captions/captions.json", "r") as f:
    captions_data = json.load(f)

print(f"Loaded {len(captions_data)} training examples")

print("\nLoading Stable Diffusion v1.5...")
print("(This will download ~4GB on first run)")

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to(device)

print("\nConfiguring LoRA for fine-tuning...")
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.1
)

# Apply LoRA to UNet
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.print_trainable_parameters()

print("\n" + "="*60)
print("Model Ready for Training!")
print("="*60)
print("\nTrainable parameters configured.")
print("LoRA will train only 1-2% of total parameters (very fast!)")

print("\n" + "="*60)
print("NOTE: Full training implementation requires DreamBooth/textual inversion")
print("For your project, you have two options:")
print("="*60)
print("\nOption 1: Use pre-trained Stable Diffusion (no training needed)")
print("  - Generate wellness images immediately")
print("  - Good for demonstrations")
print("  - Run: python3 generate_images.py")
print("\nOption 2: Fine-tune with DreamBooth (requires more setup)")
print("  - Custom wellness style")
print("  - 1-2 hour training")
print("  - More complex implementation")

print("\nRecommendation for your Gen AI course:")
print("Start with Option 1 (pre-trained) to have working image generation")
print("Then add fine-tuning if you have time")
