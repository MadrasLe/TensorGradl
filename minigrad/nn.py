import numpy as np
from minigrad.tensor import Tensor
from minigrad.backend import xp

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = xp.zeros_like(p.data)

    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, Tensor):
                pass
        return params
        
    def to(self, device):
        # Move all parameters to device (if we implemented full .to() logic)
        # For now, assume global device config or manual Tensor movement
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        # Init on CPU (numpy) then Tensor will move to device if configured
        # Ideally we initialize directly on device to save memory
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
        self.weight = Tensor(np.random.normal(0, 0.1, (num_embeddings, embedding_dim)))
        
    def forward(self, idx):
        return self.weight[idx.data.astype(int)]

    def parameters(self):
        return [self.weight]

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = Tensor(np.ones(dim))
        self.bias = Tensor(np.zeros(dim))
        
    def forward(self, x):
        mean = x.sum(axis=-1, keepdims=True) / x.data.shape[-1]
        variance = ((x - mean) ** 2).sum(axis=-1, keepdims=True) / x.data.shape[-1]
        x_norm = (x - mean) * ((variance + self.eps) ** -0.5)
        return x_norm * self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]

class ReLU(Module):
    def forward(self, x):
        out = Tensor(xp.maximum(0, x.data), (x,), 'ReLU')
        
        def _backward():
            x.grad += (x.data > 0) * out.grad
        out._backward = _backward
        return out

class GELU(Module):
    def forward(self, x):
        # Using numpy pi/sqrt. If data is cupy, these scalars are fine (broadcast).
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
