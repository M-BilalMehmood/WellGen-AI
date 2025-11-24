#!/usr/bin/env python3
"""Train Flan-T5 Base for WellGen AI - Optimized for wellness/diet advice."""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

class WellnessDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=512, max_target_length=512):
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
            'labels': labels
        }

print("="*60)
print("WellGen AI - Flan-T5 Base Training")
print("="*60)

print("\nLoading data...")
with open('data/training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Loaded {len(data)} examples")

print("\nLoading Flan-T5 Base (250M parameters)...")
print("This will download ~1GB model (first time only)\n")

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to(device)

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

print("\nPreparing dataset...")
dataset = WellnessDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Increased batch size

optimizer = AdamW(model.parameters(), lr=3e-4)  # Higher learning rate
total_steps = len(dataloader) * 5  # More epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)

print(f"\nTraining: 5 epochs, {len(dataloader)} batches/epoch")
print(f"Total steps: {total_steps}")
print("="*60 + "\n")

model.train()
for epoch in range(5):  # 5 epochs instead of 3
    print(f"Epoch {epoch+1}/5")
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Average loss: {avg_loss:.4f}\n")

print("Saving model...")
save_path = 'model'
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\nModel saved to: {save_path}/")
print("\nTo use the model:")
print("  from transformers import T5Tokenizer, T5ForConditionalGeneration")
print(f"  tokenizer = T5Tokenizer.from_pretrained('{save_path}')")
print(f"  model = T5ForConditionalGeneration.from_pretrained('{save_path}')")
