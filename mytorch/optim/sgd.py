from .base_optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001, weight_decay=0.0):
        self.params = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            g = p.grad
            
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            
            p.data -= self.lr * g
    
    def zero_grad(self):
        for p in self.params:
            if p.requires_grad:
                p.grad = None