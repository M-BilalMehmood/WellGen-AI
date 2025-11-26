#!/usr/bin/env python3
"""
Generate images using fine-tuned Stable Diffusion LoRA.
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

def generate_images(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: Generating on CPU will be slow.")

    print(f"Loading base model: {args.base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    )
    pipe = pipe.to(device)

    if args.lora_path:
        print(f"Loading LoRA weights from: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)

    print("\nGenerating images...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_images):
        seed = args.seed + i if args.seed is not None else None
        generator = torch.Generator(device).manual_seed(seed) if seed else None
        
        print(f"Generating image {i+1}/{args.num_images}...")
        image = pipe(
            args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator
        ).images[0]
        
        save_path = output_dir / f"generated_{i:04d}.png"
        image.save(save_path)
        print(f"Saved to {save_path}")

    print(f"\nDone! All images saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with LoRA")
    parser.add_argument("--base_model", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE", help="Base model path or ID")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to trained LoRA weights folder")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="low quality, bad anatomy, worst quality, deformed, disfigured", help="Negative prompt")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Output directory")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    generate_images(args)
