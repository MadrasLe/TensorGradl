import numpy as np
from minigrad.tensor import Tensor
from minigrad.gpt import GPT
import minigrad.optim as optim

def test_gpt_forward():
    vocab_size = 100
    n_embd = 32
    n_head = 4
    n_layer = 2
    block_size = 16
    
    model = GPT(vocab_size, n_embd, n_head, n_layer, max_len=block_size)
    
    # Dummy input
    idx = np.array([[1, 5, 10, 2]], dtype=int) # (1, 4)
    
    logits = model(Tensor(idx))
    
    print("Logits shape:", logits.data.shape)
    assert logits.data.shape == (1, 4, vocab_size)

    # Check backward
    loss = logits.sum()
    loss.backward()
    
    print("Gradients computed.")
    assert model.wte.weight.grad is not None
    assert np.any(model.wte.weight.grad != 0)

if __name__ == "__main__":
    test_gpt_forward()
