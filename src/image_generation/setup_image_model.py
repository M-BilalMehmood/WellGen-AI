#!/usr/bin/env python3
"""Setup script for WellGen AI Image Generation - Stable Diffusion LoRA fine-tuning."""

import subprocess
import sys

print("="*60)
print("WellGen AI - Image Generation Setup")
print("="*60)

print("\nThis will set up Stable Diffusion LoRA fine-tuning for:")
print("  - Meal visualization generation")
print("  - Nutrition chart creation")
print("  - Exercise diagram generation")
print("  - Health/wellness imagery")

print("\nRequired packages:")
packages = [
    "diffusers",
    "accelerate",
    "transformers",
    "bitsandbytes",
    "peft",
    "pillow",
    "datasets"
]

print("\nInstalling packages...")
for pkg in packages:
    print(f"Installing {pkg}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", pkg])

print("\n" + "="*60)
print("Setup Complete!")
print("="*60)
print("\nNext steps:")
print("1. Download food/wellness dataset from Kaggle")
print("2. Run train_image_model.py to fine-tune Stable Diffusion")
print("3. Generate custom wellness images with generate_images.py")
