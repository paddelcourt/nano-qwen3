#!/usr/bin/env python3
"""
Nano Qwen3 Model Implementation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Qwen3Config:
    vocab_size: int = 1500
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 1536
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    qk_norm: bool = True
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def rotate_half(x):
    """Rearrange tensor for rotation"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to Q and K"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding implementation"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len):
        device = x.device
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention implementation"""
    
    def __init__(self, config: Qwen3Config, layer_idx: int = 0):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

        self.rotary_emb = RotaryPositionEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
            
        self.attention_dropout = config.attention_dropout
        self.layer_idx = layer_idx

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match number of Q heads for GQA"""
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2)
        x = x.expand(batch_size, num_kv_heads, self.num_kv_groups, seq_len, head_dim)
        x = x.reshape(batch_size, num_kv_heads * self.num_kv_groups, seq_len, head_dim)
        return x
        
    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)
            
        cos, sin = self.rotary_emb(values, seq_len)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        if self.num_kv_groups > 1:
            keys = self._repeat_kv(keys)
            values = self._repeat_kv(values)

        attn_weights = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is None:
            attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1)
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

class Qwen3DecoderLayer(nn.Module):
    """Single transformer decoder layer"""
    
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.mlp = SwiGLU(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class Qwen3Model(nn.Module):
    """Core Qwen3 transformer model"""
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Qwen3ForCausalLM(nn.Module):
    """Qwen3 model for causal language modeling"""
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids, labels=None, attention_mask=None):
        outputs = self.model(input_ids, attention_mask)
        
        if self.lm_head is not None:
            logits = self.lm_head(outputs)
        else:
            # Tied embeddings
            logits = F.linear(outputs, self.model.embed_tokens.weight)
        
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=100, temperature=1.0, do_sample=True):
        """Simple generation method"""
        self.eval()
        
        for _ in range(max_length - input_ids.size(1)):
            outputs = self.model(input_ids)
            
            if self.lm_head is not None:
                logits = self.lm_head(outputs)
            else:
                logits = F.linear(outputs, self.model.embed_tokens.weight)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Add to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS (you'd need to define this)
            # if next_token.item() == eos_token_id:
            #     break
        
        return input_ids