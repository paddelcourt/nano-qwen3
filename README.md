# Nano-Qwen3

A minimal Qwen3 transformer implementation for learning. Train your own language model based on Tiny Story!

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (5 min)
python tiny_stories.py

# 3. Train model (45-60 min)
python train_with_autotokenizer.py

# 4. Test your model
python quick_inference.py

# 5. Chat with the model or generate short stories
python inference.py
```

## What You Get

- **Small but functional**: 2M parameters, trains in ~1 hour
- **Real text generation**: Learns to write simple stories
- **Educational code**: Clean implementation of transformer concepts
- **Interactive testing**: Chat with your trained model

## Tutorial

On `qwen3-toy-model-tutorial.html`, you will get a step-by-step tutorial on implementing the toy model
to better grasp the inner workings and key features.


## Key Features

- Grouped Query Attention (GQA) 
- Rotary Position Embeddings (RoPE)
- RMS Normalization
- Complete training pipeline

## Training Configuration

**Current setup** (~1800 iterations, 45-60 min):
```python
num_epochs=1          # Training passes through data
batch_size=16         # Stories processed together
json_files[:5]        # Number of data files used
stories[:28800]       # Total stories for training
max_position_embeddings=128  # Sequence length
```

**Iterations calculation:**
```
Iterations per epoch = stories ÷ batch_size
                     = 28,800 ÷ 16 = 1,800
Total iterations     = iterations_per_epoch × num_epochs
                     = 1,800 × 1 = 1,800
```

**To train longer/better:**
- Change `num_epochs=2` (1,800 × 2 = **3,600 iterations**)
- Increase `stories[:50000]` (50,000 ÷ 16 = **3,125 iterations per epoch**)
- Use `json_files[:10]` (more variety, needs more stories)

**To train faster:**
- Reduce `stories[:10000]` (10,000 ÷ 16 = **625 iterations**)
- Increase `batch_size=32` (28,800 ÷ 32 = **900 iterations**, faster)
- Use `json_files[:2]` (less data variety)

## File Structure

```
nano-qwen3/
├── nano_qwen3.py            # Complete model implementation (Qwen3Config, Qwen3ForCausalLM) 
├── tokenizer.py             # Qwen3-compatible tokenizer using tiktoken
├── tiny_stories.py          # Dataset download and processing pipeline
├── train_with_autotokenizer.py  # Training script (~1800 iterations, 45-60 min)
├── inference.py             # Full inference with chat/story modes
├── quick_inference.py       # Simple 3-prompt test
├── nano-qwen3.ipynb         # Model implementation notebook (in progress)
├── tinystories/             # Generated dataset directory
│   ├── *.json              # 50 TinyStories JSON files
│   ├── train.bin           # Tokenized training data
│   └── val.bin             # Tokenized validation data
└── checkpoints/             # Model checkpoints
    ├── best_model.pth      # Best performing checkpoint
    └── latest_checkpoint.pth  # Most recent checkpoint
```

After training, your model will generate basic stories like:
> "Once upon a time there was a little girl who lived in a small house..."

Perfect for understanding how modern language models work!


## Special Thanks
- Andrej Kaparthy: Source of inspiration for the project and also wrote the `tiny_stories.py`
- Qwen 3 Team: For making the model and open sourcing the techniques
