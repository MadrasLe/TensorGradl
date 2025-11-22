import numpy as np
from minigrad.tensor import Tensor

def cross_entropy(logits, target):
    # logits: (B, T, vocab_size)
    # target: (B, T) indices inteiros
    
    B, T, V = logits.shape
    
    # 1. Softmax
    probs = logits.softmax(dim=-1)
    
    # 2. Flatten
    probs_flat = probs.reshape((B*T, V))
    targets_flat = target.data.reshape((B*T)).astype(int)
    
    # 3. Gather correct probabilities
    # We want: correct_probs[i] = probs_flat[i, targets_flat[i]]
    # We can use integer array indexing on the Tensor itself since we implemented __getitem__ with autograd.
    # Tensor.__getitem__ accepts tuple indices.
    
    # Indices for rows: 0..N-1
    row_indices = np.arange(B*T)
    
    # We pass a tuple (rows, cols) to __getitem__
    # probs_flat is a Tensor.
    # __getitem__ logic: self.data[idx]. idx can be tuple.
    
    correct_probs = probs_flat[(row_indices, targets_flat)]
    
    # 4. Loss = -mean(log(correct_probs + epsilon))
    loss = -(correct_probs + 1e-8).log().mean()
    
    return loss
