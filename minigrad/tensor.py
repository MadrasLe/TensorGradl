from minigrad.backend import xp, to_device, to_cpu, get_array_module

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        # Ensure data is on the correct device
        self.data = to_device(data) if not isinstance(data, xp.ndarray) else data
        self.grad = xp.zeros_like(self.data, dtype=float)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        
    def __repr__(self):
        # Move to CPU for printing
        data_cpu = to_cpu(self.data)
        return f"Tensor(data={data_cpu}, shape={data_cpu.shape})"

    @property
    def shape(self):
        return self.data.shape

    def backward(self):
        # Topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = xp.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
            
    def to(self, device):
        """Dummy method for compatibility, actual movement handled globally for now"""
        return self

    # --- Operations ---

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
            
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)
            
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += unbroadcast((other * self.data**(other-1)) * out.grad, self.data.shape)
            
        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            if self.data.ndim == 2 and other.data.ndim == 2:
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
            else:
                 self.grad += unbroadcast(out.grad @ other.data.swapaxes(-1, -2), self.data.shape)
                 other.grad += unbroadcast(self.data.swapaxes(-1, -2) @ out.grad, other.data.shape)

        out._backward = _backward
        return out
        
    def sum(self, axis=None, keepdims=False):
        out = Tensor(xp.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        
        def _backward():
            grad_output = out.grad
            if not keepdims and axis is not None:
                 grad_output = xp.expand_dims(out.grad, axis=axis)
            elif not keepdims and axis is None:
                 grad_output = xp.full(self.data.shape, out.grad) 
                 self.grad += grad_output
                 return 

            self.grad += grad_output * xp.ones_like(self.data)
            
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(xp.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(xp.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(xp.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')
        
        def _backward():
            if axis is None:
                N = self.data.size
            else:
                if isinstance(axis, int):
                    N = self.data.shape[axis]
                else:
                    N = 1
                    for a in axis:
                        N *= self.data.shape[a]
            
            grad_output = out.grad / N
            
            if not keepdims and axis is not None:
                 grad_output = xp.expand_dims(grad_output, axis=axis)
            elif not keepdims and axis is None:
                 grad_output = xp.full(self.data.shape, grad_output)
                 self.grad += grad_output
                 return
                 
            self.grad += grad_output * xp.ones_like(self.data)
            
        out._backward = _backward
        return out

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), (self,), 'reshape')
        
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,), 'transpose')
        
        def _backward():
            if not axes:
                 self.grad += out.grad.transpose()
            else:
                 import numpy as np 
                 inv_axes = np.argsort(axes)
                 # -------------------------
                 self.grad += out.grad.transpose(*inv_axes)
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), 'getitem')
        
        def _backward():
            grad_update = xp.zeros_like(self.data)
            xp.add.at(grad_update, idx, out.grad)
            self.grad += grad_update
            
        out._backward = _backward
        return out
        
    def __matmul__(self, other):
        return self.matmul(other)
        
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
        
    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
        
    def __rsub__(self, other):
        return other + (-self)
        
    def __rtruediv__(self, other):
        return other * self**-1

    def tanh(self):
        t = xp.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def softmax(self, dim=-1):
        x_max = xp.max(self.data, axis=dim, keepdims=True)
        e_x = xp.exp(self.data - x_max)
        out_data = e_x / xp.sum(e_x, axis=dim, keepdims=True)
        
        out = Tensor(out_data, (self,), 'softmax')
        
        def _backward():
            s = out.data
            g = out.grad
            sg = xp.sum(g * s, axis=dim, keepdims=True)
            self.grad += s * (g - sg)

        out._backward = _backward
        return out

def unbroadcast(grad, shape):
    if grad.shape == shape:
        return grad
        
    ndims_added = grad.ndim - len(shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
        
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
            
    return grad
