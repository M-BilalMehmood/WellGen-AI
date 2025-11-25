#!/usr/bin/env python3
"""Download and prepare conversational wellness dataset."""

import json
from datasets import load_dataset

print("Downloading conversational medical dataset...")
print("This uses real doctor-patient conversations\n")

# Load medical dialog dataset
dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")

print(f"Loaded {len(dataset)} conversations")

# Convert to proper format
training_data = []

for item in dataset:
    if 'Patient' in item and 'Doctor' in item:
        training_data.append({
            'text': f"USER: {item['Patient']}\nASSISTANT: {item['Doctor']}",
            'source': 'medical-chatbot'
        })

print(f"\nProcessed {len(training_data)} conversational examples")

# Save
with open('data/training_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print(f"Saved to data/training_data.json")
print("\nThis is REAL medical conversations, not templates!")
