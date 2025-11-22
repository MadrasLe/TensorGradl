import numpy as np
from minigrad.tensor import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, Tensor):
                # If we attach Tensors directly (like weights), include them? 
                # Usually weights are wrapped in Parameter class in PyTorch, 
                # but here we use bare Tensors often inside Modules.
                # But let's stick to sub-modules to be safe for now unless we know it's a parameter.
                # For "minigrad", let's assume direct Tensors in __dict__ are NOT parameters unless explicitly returned by subclass.
                # Wait, this recursive search is for subclasses that DON'T override parameters().
                pass
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Tensor(np.random.uniform(-1, 1, (in_features, out_features)) / np.sqrt(in_features))
        self.bias = Tensor(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x.matmul(self.weight)
        if self.bias:
            out = out + self.bias
        return out
        
    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias else [])

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        # Use smaller initialization (scaled normal) similar to GPT-2
        self.weight = Tensor(np.random.normal(0, 0.1, (num_embeddings, embedding_dim)))
        
    def forward(self, idx):
        # Optimized forward pass using slicing (enabled by Tensor.__getitem__)
        # idx is a Tensor of indices. We need simple numpy slicing.
        # Tensor.__getitem__ handles backward via np.add.at
        return self.weight[idx.data.astype(int)]

    def parameters(self):
        return [self.weight]

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = Tensor(np.ones(dim))
        self.bias = Tensor(np.zeros(dim))
        
    def forward(self, x):
        # x: (B, T, D)
        mean = x.sum(axis=-1, keepdims=True) / x.data.shape[-1]
        variance = ((x - mean) ** 2).sum(axis=-1, keepdims=True) / x.data.shape[-1]
        x_norm = (x - mean) * ((variance + self.eps) ** -0.5)
        return x_norm * self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]

class ReLU(Module):
    def forward(self, x):
        # x * (x > 0)
        mask = (x.data > 0).astype(np.float64)
        # We need to make sure we don't break the graph. 
        # We can implement ReLU as a Tensor op or compose it.
        # Since Tensor doesn't have > operator returning Tensor (returns bool array), 
        # we can't purely compose yet.
        # Let's add a specialized op or implement it here.
        
        out = Tensor(np.maximum(0, x.data), (x,), 'ReLU')
        
        def _backward():
            x.grad += (x.data > 0) * out.grad
        out._backward = _backward
        return out

class GELU(Module):
    def forward(self, x):
        # Approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Implementing exact or approx? Let's do the simple exact one: 0.5 * x * (1 + erf(x / sqrt(2)))
        # But we don't have erf in Tensor.
        # Let's use the approximation used in GPT-2 (often).
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        
        # We need Tanh.
        # Let's implement Tanh in Tensor or here.
        
        return x * 0.5 * (1.0 + ((((x ** 3) * 0.044715) + x) * np.sqrt(2.0 / np.pi)).tanh())

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
