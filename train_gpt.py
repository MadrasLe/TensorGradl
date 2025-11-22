import numpy as np
from minigrad.tensor import Tensor
from minigrad.gpt import GPT
from minigrad.loss import cross_entropy
import minigrad.optim as optim
from minigrad.backend import to_device, to_cpu

def train():
    # Hyperparameters
    vocab_size = 6
    n_embd = 16
    n_head = 2
    n_layer = 2
    block_size = 4
    batch_size = 8
    steps = 300
    
    # Model
    model = GPT(vocab_size, n_embd, n_head, n_layer, max_len=block_size)
    
    # Adam optimizer usually works best for Transformers
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Dataset: Repeated sequence
    seq = [0, 1, 2, 3, 4, 5]
    raw_data = np.array(seq * 50)
    
    print(f"Starting training GPT on repeated sequence...")
    print(f"Params: {len(model.parameters())} tensors found.")
    
    for step in range(steps):
        inputs = []
        targets = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(raw_data) - block_size - 1)
            inputs.append(raw_data[idx:idx+block_size])
            targets.append(raw_data[idx+1:idx+block_size+1])
            
        # Create Tensors (will move to device automatically if configured)
        x = Tensor(np.array(inputs))
        y = Tensor(np.array(targets))
        
        logits = model(x)
        loss = cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            # Move data to CPU for printing if needed (handled by Tensor.__repr__ mostly)
            # But loss.data might be cupy array
            loss_val = to_cpu(loss.data)
            print(f"Step {step}: Loss = {loss_val:.4f}")

    final_loss = to_cpu(loss.data)
    print(f"Final Loss: {final_loss:.4f}")
    
    # Verification
    if final_loss < 0.1:
        print("SUCCESS: Model learned the pattern!")
        
        # Test generation (greedy)
        print("\nGenerating...")
        context = [0, 1]
        for _ in range(10):
            x_cond = Tensor(np.array([context[-block_size:]]))
            logits = model(x_cond)
            # Greedy decoding
            last_logits = logits.data[0, -1, :]
            # Move to CPU for argmax if it's not already handled
            last_logits_cpu = to_cpu(last_logits)
            next_token = np.argmax(last_logits_cpu)
            context.append(int(next_token))
        print(f"Generated sequence: {context}")
    else:
        print("FAIL: Model did not converge.")

if __name__ == "__main__":
    train()
