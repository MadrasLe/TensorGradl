from minigrad.backend import xp

class Optimizer:
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            # Ensure zeroing happens on the backend array
            p.grad[:] = 0

class SGD(Optimizer):
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        # Use xp (numpy/cupy) to initialize state on the correct device
        self.m = [xp.zeros_like(p.data) for p in params]
        self.v = [xp.zeros_like(p.data) for p in params]
        self.t = 0
        
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            
            # Calculations use xp via operator overloading or explicit calls
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (p.grad ** 2)
            
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            
            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
