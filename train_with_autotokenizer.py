#!/usr/bin/env python3
"""
Training script for nano-Qwen3 on TinyStories using AutoTokenizer
"""

import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from nano_qwen3 import Qwen3Config, Qwen3ForCausalLM

class TinyStoriesDataset(Dataset):
    """Dataset for loading TinyStories JSON files"""
    
    def __init__(self, data_dir, tokenizer, max_length=512, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stories = []
        
        # Load stories from JSON files (created by tiny_stories.py)
        json_pattern = os.path.join(data_dir, "tinystories/TinyStories_all_data/*.json")
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found at {json_pattern}")
        
        if split == 'train':
            # Use files 01-49 for training (data00 is validation)
            json_files = [f for f in json_files if 'data00.json' not in f]
        else:
            # Use data00 for validation
            json_files = [f for f in json_files if 'data00.json' in f]
            
        print(f"Loading {split} data from {len(json_files)} files...")
        
        # Limit to fewer files for demo (remove this line for full training)
        json_files = json_files[:5]  # Use 5 files to get enough stories for 1800 iterations
        
        for json_file in tqdm(json_files, desc=f"Loading {split}"):
            try:
                with open(json_file, 'r') as f:
                    # Load the JSON array
                    data = json.load(f)
                    for item in data:
                        if isinstance(item, dict) and 'story' in item:
                            story = item['story'].strip()
                            if len(story) > 10:  # Filter very short stories
                                self.stories.append(story)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        print(f"Loaded {len(self.stories)} stories for {split}")
        
    def __len__(self):
        return len(self.stories)
    
    def __getitem__(self, idx):
        story = self.stories[idx]
        
        # Tokenize with AutoTokenizer
        encoding = self.tokenizer(
            story,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input_ids and clamp to vocab size to prevent index errors
        input_ids = encoding['input_ids'].squeeze(0)
        
        # Clamp token IDs to valid range [0, vocab_size-1]
        input_ids = torch.clamp(input_ids, 0, self.tokenizer.vocab_size - 1)
        
        return input_ids

def create_attention_mask(input_ids, pad_token_id):
    """Create attention mask to ignore padding tokens"""
    return (input_ids != pad_token_id).float()

def evaluate_model(model, val_dataloader, device, pad_token_id):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch.to(device)
            attention_mask = create_attention_mask(input_ids, pad_token_id).to(device)
            
            # Create labels (same as input_ids for causal LM)
            labels = input_ids.clone()
            
            # Forward pass
            logits, loss = model(input_ids, labels=labels)
            
            # Only count non-padding tokens
            mask = attention_mask.view(-1)
            if mask.sum() > 0:
                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return avg_loss, perplexity.item()

def generate_sample(model, tokenizer, device, prompt="Once upon a time", max_length=100):
    """Generate a sample story"""
    model.eval()
    
    try:
        # Encode prompt using AutoTokenizer
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']
        
        with torch.no_grad():
            generated = model.generate(input_ids, max_length=max_length, temperature=0.8)
            
        # Decode generated tokens
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        return f"Generation error: {e}"

def train_qwen3_on_tinystories(config, data_dir, num_epochs=3, batch_size=8, 
                                learning_rate=1e-4, save_dir="./checkpoints"):
    """
    Complete training pipeline for Qwen3 on TinyStories
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Set environment variable to avoid tokenizer warnings
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize AutoTokenizer
    model_name = "Qwen/Qwen3-8B"
    
    try:
        print(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"Tokenizer loaded successfully")
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        
        # Verify token IDs are within range
        print(f"Max token ID should be: {tokenizer.vocab_size - 1}")
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure you have internet connection and transformers library")
        return None
    
    # Update config with tokenizer vocab size
    config.vocab_size = tokenizer.vocab_size
    print(f"Config vocab_size updated to: {config.vocab_size}")
    
    # Verify embedding layer size matches
    print(f"Model will create embedding layer with {config.vocab_size} tokens")
    
    # Create datasets
    try:
        train_dataset = TinyStoriesDataset(data_dir, tokenizer, max_length=config.max_position_embeddings, split='train')
        val_dataset = TinyStoriesDataset(data_dir, tokenizer, max_length=config.max_position_embeddings, split='val')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo prepare TinyStories data, run:")
        print("  python tiny_stories.py")
        return None
    
    if len(train_dataset) == 0:
        print("Error: No training data loaded!")
        return None
    
    # Limit dataset size for ~1800 iterations (500 stories * 16 batch = 31 steps, need ~58x more)
    train_dataset.stories = train_dataset.stories[:28800]  # ~1800 iterations with batch_size=16
    val_dataset.stories = val_dataset.stories[:1000]       # More validation data
    print(f"Limited to {len(train_dataset.stories)} training stories, {len(val_dataset.stories)} validation stories")
    print(f"Expected iterations per epoch: {len(train_dataset.stories) // 16}")
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = Qwen3ForCausalLM(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_dataloader) * num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        epoch_loss = 0
        epoch_steps = 0
        start_time = time.time()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, input_ids in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            
            # Create labels (same as input_ids for causal language modeling)
            labels = input_ids.clone()
            
            # Forward pass
            logits, loss = model(input_ids, labels=labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{epoch_loss / epoch_steps:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log and save periodically
            if global_step % 100 == 0:  # More frequent sampling
                # Generate sample
                sample_text = generate_sample(model, tokenizer, device)
                print(f"\nStep {global_step} sample: {sample_text}")
                model.train()  # Back to training mode
        
        # End of epoch evaluation
        avg_train_loss = epoch_loss / epoch_steps
        
        if len(val_dataloader) > 0:
            val_loss, val_perplexity = evaluate_model(model, val_dataloader, device, tokenizer.pad_token_id)
        else:
            val_loss, val_perplexity = float('inf'), float('inf')
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Perplexity: {val_perplexity:.2f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'config': config,
            'global_step': global_step,
            'tokenizer_name': model_name
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print("  ✓ New best model saved!")
        
        model.train()  # Back to training mode
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Generate final samples
    print("\n=== Final Generation Samples ===")
    for prompt in ["Once upon a time", "The little girl", "In a magical forest"]:
        sample = generate_sample(model, tokenizer, device, prompt=prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {sample}")
    
    return model

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = Qwen3Config()
    
    # Adjust for MacBook Air (smaller model)
    config.hidden_size = 256
    config.num_hidden_layers = 4
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.intermediate_size = 1024
    config.max_position_embeddings = 128  # Much shorter sequences for speed
    
    print("Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  KV heads: {config.num_key_value_heads}")
    print(f"  Max sequence length: {config.max_position_embeddings}")
    
    # Train model
    data_dir = "."  # Current directory (should contain tinystories/ folder)
    trained_model = train_qwen3_on_tinystories(
        config=config,
        data_dir=data_dir,
        num_epochs=1,  # Reduced from 2 to 1
        batch_size=16,  # Larger batch size for fewer steps
        learning_rate=2e-4  # Slightly higher learning rate
    )