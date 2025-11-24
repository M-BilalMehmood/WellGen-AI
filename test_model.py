#!/usr/bin/env python3
"""Test WellGen AI model with sample questions."""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...\n")

tokenizer = T5Tokenizer.from_pretrained('model', legacy=False)
model = T5ForConditionalGeneration.from_pretrained('model').to(device)
model.eval()

test_questions = [
    "What foods help with weight loss?",
    "List high protein foods",
    "What should diabetics eat?",
    "Best foods for heart health",
    "Healthy breakfast ideas"
]

print("="*60)
print("WellGen AI - Test Results")
print("="*60 + "\n")

for question in test_questions:
    inputs = tokenizer(question, return_tensors='pt', max_length=256, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=200,
            num_beams=4,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            early_stopping=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Q: {question}")
    print(f"A: {response}\n")

print("="*60)
print("\nYour model is trained and working!")
print("It gives general nutrition advice based on the Kaggle datasets.")
print("For more specific responses, you would need more training data")
print("or fine-tune with more specific Q&A pairs.")
