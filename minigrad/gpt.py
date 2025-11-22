import numpy as np
from minigrad.tensor import Tensor
from minigrad.backend import xp, to_device
import minigrad.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, max_len=1024):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        # Causal mask 
        # Ensure it's on the backend device
        self.bias = to_device(np.tril(np.ones((max_len, max_len))).reshape(1, 1, max_len, max_len))
        
    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.data.shape
        
        qkv = self.c_attn(x) # (B, T, 3*C)
        
        # Split qkv into q, k, v
        qkv = qkv.reshape((B, T, 3, self.n_head, self.head_dim))
        qkv = qkv.transpose(2, 0, 3, 1, 4) # (3, B, n_head, T, head_dim)
        
        q = qkv[0] # (B, n_head, T, head_dim)
        k = qkv[1]
        v = qkv[2]
        
        # Attention scores
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(k.data.shape[-1]))
        
        # Apply Mask
        mask = self.bias[:, :, :T, :T]
        
        att = att + Tensor((1.0 - mask) * -1e10)
        att = att.softmax(dim=-1)
        
        y = att @ v 
        
        y = y.transpose(0, 2, 1, 3) # (B, T, n_head, head_dim)
        y = y.reshape((B, T, C))
        
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_len=1024):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(max_len, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx):
        B, T = idx.shape
        
        pos = xp.arange(0, T, dtype=int) # (T) on device
        pos = Tensor(xp.broadcast_to(pos, (B, T))) # (B, T)
        
        # Embeddings
        tok_emb = self.wte(idx) # (B, T, C)
        pos_emb = self.wpe(pos) # (B, T, C)
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Weight tying implementation
        logits = x.matmul(self.wte.weight.transpose())
        
        return logits
