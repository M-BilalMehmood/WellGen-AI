#!/usr/bin/env python3
"""Download Food-101 dataset for training wellness image generation model."""

import os
from datasets import load_dataset
import json

print("="*60)
print("Downloading Food-101 Dataset (Legitimate Kaggle/HuggingFace)")
print("="*60)

print("\nDataset: food101 - 101,000 food images across 101 categories")
print("Source: HuggingFace Datasets (from Kaggle)")
print("Perfect for wellness/meal planning image generation!")

print("\nDownloading dataset (this may take 10-15 minutes)...")
dataset = load_dataset("food101", split="train[:1000]")  # Use 1000 images for fast training

print(f"\nDataset loaded: {len(dataset)} images")

# Create directories
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/captions", exist_ok=True)

print("\nProcessing images...")
captions = []

for idx, item in enumerate(dataset):
    if idx >= 1000:  # Limit to 1000 for fast training
        break
    
    # Save image
    img = item['image']
    label = item['label']
    img_path = f"data/images/food_{idx:04d}.jpg"
    img.save(img_path)
    
    # Create caption
    caption = f"A healthy meal showing {dataset.features['label'].int2str(label)}, suitable for wellness and nutrition"
    captions.append({
        "image": img_path,
        "caption": caption
    })
    
    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1} images...")

# Save captions
with open("data/captions/captions.json", "w") as f:
    json.dump(captions, f, indent=2)

print("\n" + "="*60)
print("Dataset Ready!")
print("="*60)
print(f"Images saved: data/images/")
print(f"Captions saved: data/captions/captions.json")
print(f"Total examples: {len(captions)}")
print("\nNext: Run train_image_model.py to fine-tune Stable Diffusion")
