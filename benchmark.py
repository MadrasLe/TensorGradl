import time
import numpy as np
from minigrad.tensor import Tensor
from minigrad.gpt import GPT
from minigrad.loss import cross_entropy
import minigrad.optim as optim
from minigrad.backend import xp, backend_name

def benchmark():
    print(f"Running benchmark on: {backend_name.upper()}")
    
    # Model Config
    vocab_size = 50257 # GPT-2 size
    n_embd = 64 # Small for testing
    n_head = 4
    n_layer = 2
    block_size = 32
    batch_size = 8
    
    model = GPT(vocab_size, n_embd, n_head, n_layer, max_len=block_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy Data
    inputs_np = np.random.randint(0, vocab_size, (batch_size, block_size))
    targets_np = np.random.randint(0, vocab_size, (batch_size, block_size))
    
    x = Tensor(inputs_np)
    y = Tensor(targets_np)
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        logits = model(x)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Benchmark
    steps = 10
    start_time = time.time()
    
    print(f"Running {steps} steps...")
    for i in range(steps):
        logits = model(x)
        loss = cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Ensure sync if CUDA
        if backend_name == 'cuda':
            xp.cuda.Stream.null.synchronize()
            
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Total time: {duration:.4f}s")
    print(f"Steps per second: {steps/duration:.2f}")

if __name__ == "__main__":
    benchmark()
