import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.data.shape})"

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
        
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    # --- Operations ---

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Gradient flows equally to both inputs
            # Handle broadcasting
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
            # grad = (B, M, P)
            # self = (B, M, N), other = (B, N, P) (simplified view)
            # self.grad = out.grad @ other.T
            # other.grad = self.T @ out.grad
            # Need to handle matrix multiplication rules carefully regarding shapes
            
            # Simple case (2D)
            if self.data.ndim == 2 and other.data.ndim == 2:
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
            else:
                 # Handle batched matmul if needed or implement general case
                 # For now, let's assume standard numpy behavior handles dimensions, 
                 # but gradients need transpose of the last two dims.
                 self.grad += unbroadcast(out.grad @ other.data.swapaxes(-1, -2), self.data.shape)
                 other.grad += unbroadcast(self.data.swapaxes(-1, -2) @ out.grad, other.data.shape)

        out._backward = _backward
        return out
        
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        
        def _backward():
            # Need to broadcast gradient back to input shape
            # If keepdims was False, we need to add the dimensions back to broadcast
            grad_output = out.grad
            if not keepdims and axis is not None:
                 # Expand dims for the reduced axes
                 grad_output = np.expand_dims(out.grad, axis=axis)
            elif not keepdims and axis is None:
                 grad_output = np.full(self.data.shape, out.grad) # Scalar expanded to full shape
                 self.grad += grad_output
                 return 

            # Now grad_output should be broadcastable
            self.grad += grad_output * np.ones_like(self.data)
            
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        
        def _backward():
            # d/dx ln(x) = 1/x
            # grad = (1 / self.data) * out.grad
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')
        
        def _backward():
            # grad = out.grad / N
            # We need N corresponding to the reduced dimensions
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
            
            # Same broadcasting logic as sum
            if not keepdims and axis is not None:
                 grad_output = np.expand_dims(grad_output, axis=axis)
            elif not keepdims and axis is None:
                 grad_output = np.full(self.data.shape, grad_output)
                 self.grad += grad_output
                 return
                 
            self.grad += grad_output * np.ones_like(self.data)
            
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
            # Inverse permutation for transpose? 
            # If axes is None (default), it reverses dims.
            if not axes:
                 self.grad += out.grad.transpose()
            else:
                 # argsort of axes gives the inverse permutation
                 inv_axes = np.argsort(axes)
                 self.grad += out.grad.transpose(*inv_axes)
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), 'getitem')
        
        def _backward():
            # Add gradient to the specific slice
            # We need to create a zero tensor of same shape as self.data
            # and add out.grad to the position idx.
            # Since idx can contain duplicates (especially in Embedding), 
            # we must accumulate gradients using np.add.at.
            grad_update = np.zeros_like(self.data)
            np.add.at(grad_update, idx, out.grad)
            self.grad += grad_update
            
        out._backward = _backward
        return out
        
    # Aliases and helper methods
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
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def softmax(self, dim=-1):
        # exp(x - max(x)) / sum(exp(x - max(x)))
        # Stable softmax
        x_max = np.max(self.data, axis=dim, keepdims=True)
        e_x = np.exp(self.data - x_max)
        out_data = e_x / np.sum(e_x, axis=dim, keepdims=True)
        
        out = Tensor(out_data, (self,), 'softmax')
        
        def _backward():
            s = out.data
            g = out.grad
            
            # sum(grad_output * S, dim)
            sg = np.sum(g * s, axis=dim, keepdims=True)
            self.grad += s * (g - sg)

        out._backward = _backward
        return out

def unbroadcast(grad, shape):
    # Helper to sum gradients when broadcasting occurred
    if grad.shape == shape:
        return grad
        
    # Sum across the extra dimensions at the front
    ndims_added = grad.ndim - len(shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
        
    # Sum across dimensions where input has size 1 but grad has size > 1
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
            
    return grad
