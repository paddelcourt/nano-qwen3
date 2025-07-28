#!/usr/bin/env python3
"""
Quick inference example for nano-Qwen3
"""

import torch
from transformers import AutoTokenizer
from nano_qwen3 import Qwen3ForCausalLM

# Quick function to test your model
def quick_test(checkpoint_path="checkpoints/best_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint - add safe globals for PyTorch 2.6+ compatibility
    from nano_qwen3 import Qwen3Config
    torch.serialization.add_safe_globals([Qwen3Config])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Load model
    model = Qwen3ForCausalLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a magical forest"
    ]
    
    print(f"🤖 Testing nano-Qwen3 (epoch {checkpoint['epoch']}, loss {checkpoint['train_loss']:.3f})")
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            generated = model.generate(inputs['input_ids'], max_length=80, temperature=0.8)
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {text}")

if __name__ == "__main__":
    quick_test()