#!/usr/bin/env python3
"""
Inference script for trained nano-Qwen3 model
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from nano_qwen3 import Qwen3Config, Qwen3ForCausalLM

def load_trained_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint"""
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Add safe globals for PyTorch 2.6+ compatibility
    from nano_qwen3 import Qwen3Config
    torch.serialization.add_safe_globals([Qwen3Config])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config = checkpoint['config']
    print(f"Model config: {config.hidden_size}d, {config.num_hidden_layers} layers")
    
    # Initialize model
    model = Qwen3ForCausalLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer_name = checkpoint.get('tokenizer_name', 'Qwen/Qwen3-8B')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Training epoch: {checkpoint['epoch']}")
    print(f"   Training loss: {checkpoint['train_loss']:.4f}")
    
    return model, tokenizer, config

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, 
                  top_k=50, top_p=0.9, do_sample=True, device='cpu'):
    """Generate text from a prompt"""
    
    model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    
    print(f"Prompt: '{prompt}'")
    print(f"Input tokens: {input_ids.shape[1]}")
    print("Generating...")
    
    with torch.no_grad():
        generated_ids = input_ids.clone()
        
        for step in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = model.model(generated_ids)
            
            if model.lm_head is not None:
                logits = model.lm_head(outputs)
            else:
                logits = F.linear(outputs, model.model.embed_tokens.weight)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Add to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # Print progress
            if step % 10 == 0:
                current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                print(f"Step {step}: ...{current_text[-50:]}")
    
    # Decode final result
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

def interactive_chat(model, tokenizer, config, device='cpu'):
    """Interactive chat interface"""
    
    print("\n🤖 Nano-Qwen3 Interactive Chat")
    print("Type 'quit' to exit, 'clear' to reset")
    print("Settings: temp=0.8, max_len=200")
    print("-" * 50)
    
    conversation_history = ""
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'clear':
                conversation_history = ""
                print("Conversation cleared!")
                continue
            elif not user_input:
                continue
            
            # Build prompt with history
            if conversation_history:
                prompt = f"{conversation_history}\nHuman: {user_input}\nAssistant:"
            else:
                prompt = f"Human: {user_input}\nAssistant:"
            
            # Generate response
            generated = generate_text(
                model, tokenizer, prompt,
                max_length=min(200, config.max_position_embeddings),
                temperature=0.8,
                do_sample=True,
                device=device
            )
            
            # Extract assistant response
            if "Assistant:" in generated:
                assistant_response = generated.split("Assistant:")[-1].strip()
                # Stop at next "Human:" if it appears
                if "Human:" in assistant_response:
                    assistant_response = assistant_response.split("Human:")[0].strip()
            else:
                assistant_response = generated[len(prompt):].strip()[:200]
            
            print(f"\nAssistant: {assistant_response}")
            
            # Update conversation history (keep it short)
            conversation_history = f"{conversation_history}\nHuman: {user_input}\nAssistant: {assistant_response}"
            # Limit history length
            if len(conversation_history) > 1000:
                conversation_history = conversation_history[-800:]
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

def run_story_generation(model, tokenizer, config, device='cpu'):
    """Generate sample stories"""
    
    print("\n📚 Story Generation Samples")
    print("-" * 50)
    
    story_prompts = [
        "Once upon a time, there was a little girl named",
        "In a magical forest, a brave knight discovered",
        "The friendly dragon lived in a castle where",
        "Every morning, the cat would wake up and",
        "Long ago, in a village by the sea,",
    ]
    
    for i, prompt in enumerate(story_prompts, 1):
        print(f"\n--- Story {i} ---")
        
        generated = generate_text(
            model, tokenizer, prompt,
            max_length=150,
            temperature=0.9,  # More creative
            top_p=0.9,
            device=device
        )
        
        print(f"Prompt: {prompt}")
        print(f"Story: {generated}")
        print()

def main():
    """Main inference function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = input("Enter checkpoint path (or press Enter for 'checkpoints/best_model.pth'): ").strip()
    if not checkpoint_path:
        checkpoint_path = "checkpoints/best_model.pth"
    
    try:
        model, tokenizer, config = load_trained_model(checkpoint_path, device)
    except FileNotFoundError:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        import os
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith('.pth'):
                    print(f"  - checkpoints/{f}")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Choose mode
    print("\nSelect mode:")
    print("1. Interactive chat")
    print("2. Story generation samples")
    print("3. Single prompt")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        interactive_chat(model, tokenizer, config, device)
    elif choice == '2':
        run_story_generation(model, tokenizer, config, device)
    elif choice == '3':
        prompt = input("Enter your prompt: ").strip()
        if prompt:
            generated = generate_text(model, tokenizer, prompt, max_length=200, device=device)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated}")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()