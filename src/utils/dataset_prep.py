#!/usr/bin/env python3
"""
Dataset Preparation Script for WellGen AI.
Helps organize images and create metadata for Stable Diffusion fine-tuning.
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def prepare_dataset(data_dir, output_dir, default_style="muscle anatomy visualization"):
    """
    Process images from subdirectories and create metadata.jsonl.
    Uses folder names as part of the caption.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    metadata = []
    
    print(f"Scanning {data_path} for images...")
    
    # Counter for processed images
    count = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(data_path):
        for file in tqdm(files, desc=f"Processing {Path(root).name}"):
            file_path = Path(root) / file
            
            if file_path.suffix.lower() in valid_extensions:
                try:
                    # Get exercise name from folder name
                    # e.g., data/Body_Exercise/bench press/img1.jpg -> "bench press"
                    exercise_name = file_path.parent.name.replace("_", " ")
                    
                    # Construct caption
                    caption = f"{exercise_name}, {default_style}"
                    
                    # Open and verify image
                    with Image.open(file_path) as img:
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save to output directory
                        new_filename = f"image_{count:04d}.jpg"
                        save_path = output_path / new_filename
                        
                        # Resize if too large (optional, but good for training speed)
                        # SD 1.5 trains on 512x512 usually
                        if max(img.size) > 1024:
                            img.thumbnail((1024, 1024))
                        
                        img.save(save_path, quality=95)
                        
                        # Add to metadata
                        metadata.append({
                            "file_name": new_filename,
                            "text": caption
                        })
                        
                        count += 1
                except Exception as e:
                    print(f"Skipping {file_path.name}: {e}")
    
    # Save metadata.jsonl
    metadata_file = output_path / "metadata.jsonl"
    with open(metadata_file, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')
            
    print(f"\nSuccess! Processed {count} images.")
    print(f"Dataset saved to: {output_path}")
    print(f"Metadata saved to: {metadata_file}")
    print("\nSample captions generated:")
    for i in range(min(5, len(metadata))):
        print(f" - {metadata[i]['text']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for SD fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing raw images (can be nested)")
    parser.add_argument("--output", type=str, default="data/processed_dataset", help="Output folder for processed dataset")
    parser.add_argument("--caption", type=str, default="muscle anatomy visualization", help="Style suffix for captions")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' not found.")
        exit(1)
        
    prepare_dataset(args.input, args.output, args.caption)
