#!/usr/bin/env python3
"""WellGen AI - Interactive wellness coach using trained Flan-T5 model."""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

print("Loading WellGen AI model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained('model', legacy=False)
model = T5ForConditionalGeneration.from_pretrained('model').to(device)
model.eval()

print(f"Model loaded on {device}")
print("\n" + "="*60)
print("WellGen AI - Your Personal Wellness Coach")
print("="*60)
print("\nAsk me about:")
print("  - Diet plans (weight loss, diabetes, heart health, etc.)")
print("  - Nutrition advice")
print("  - Meal recommendations")
print("  - Health and wellness tips")
print("\nType 'quit' to exit\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nStay healthy! 👋")
        break
    
    if not user_input:
        continue
    
    # Tokenize input
    inputs = tokenizer(user_input, return_tensors='pt', max_length=256, truncation=True).to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=256,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            early_stopping=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nWellGen AI: {response}\n")
