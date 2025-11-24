#!/usr/bin/env python3
"""Train Flan-T5 Base for WellGen AI - Improved version with validation & sampling."""

import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

class WellnessDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=256, max_target_length=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        
        # Split USER/ASSISTANT format properly
        parts = text.split('ASSISTANT:')
        if len(parts) == 2:
            input_text = parts[0].replace('USER:', '').strip()
            target_text = parts[1].strip()
        else:
            # Skip bad examples
            input_text = "Give health advice"
            target_text = "Consult a healthcare professional"
        
        # Add task prefix for T5
        input_text = f"Answer this health question: {input_text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Replace padding token id with -100 for loss calculation
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels,
            'original_input': input_text,
            'original_target': target_text
        }

def test_model_response(model, tokenizer, test_question):
    """Test model with a sample question."""
    model.eval()
    with torch.no_grad():
        input_text = f"Answer this health question: {test_question}"
        inputs = tokenizer(input_text, return_tensors='pt', max_length=256, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model.train()
    return response

print("="*60)
print("WellGen AI - Flan-T5 Base Training v2")
print("="*60)

print("\nLoading data...")
with open('data/training_data.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)
print(f"Total available: {len(all_data)} examples")

# Use subset for faster training
SUBSET_SIZE = 5000  # Use 5K examples for fast training
data = all_data[:SUBSET_SIZE]
print(f"Using subset: {len(data)} examples for fast training")

# Split into train/val (90/10)
train_size = int(0.9 * len(data))
val_size = len(data) - train_size
train_data = data[:train_size]
val_data = data[train_size:]
print(f"Train: {len(train_data)}, Validation: {len(val_data)}")

print("\nLoading Flan-T5 Base (250M parameters)...")
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to(device)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

print("\nPreparing datasets...")
train_dataset = WellnessDataset(train_data, tokenizer)
val_dataset = WellnessDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training config
EPOCHS = 5
LEARNING_RATE = 5e-4
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=200,
    num_training_steps=total_steps
)

print(f"\nTraining Configuration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: 8")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Steps per epoch: {len(train_loader)}")
print(f"  Total steps: {total_steps}")
print("="*60 + "\n")

# Test questions to monitor progress
test_questions = [
    "I am getting fat and don't know how to control it",
    "I am 175 cm, 84 Kg male. Generate me a diet plan",
    "Hello wellgen, can you help me?",
    "I have diabetes. What should I eat?"
]

best_loss = float('inf')
model.train()

for epoch in range(EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print('='*60)
    
    # Training
    train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    model.train()
    
    print(f"\nEpoch {epoch+1} Results:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    
    # Test sample responses every 2 epochs
    if (epoch + 1) % 2 == 0:
        print(f"\n{'='*60}")
        print("Sample Responses:")
        print('='*60)
        for i, question in enumerate(test_questions[:2], 1):
            response = test_model_response(model, tokenizer, question)
            print(f"\nQ{i}: {question}")
            print(f"A{i}: {response}")
    
    # Save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        print(f"\n✓ New best model! (Val Loss: {avg_val_loss:.4f})")
        save_path = 'model'
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\nBest validation loss: {best_loss:.4f}")
print(f"Model saved to: model/")

print("\n" + "="*60)
print("Final Test - All Questions:")
print("="*60)
for i, question in enumerate(test_questions, 1):
    response = test_model_response(model, tokenizer, question)
    print(f"\nQ{i}: {question}")
    print(f"A{i}: {response}")

print("\n" + "="*60)
print("Training finished! Test with chat.py")
print("="*60)
