import mytorch
import numpy as np

class Optimizer:
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError
    
    def _init_optimizer_state(self):
        raise NotImplementedError
    
    def state_dict(self):
        return None
    
    def load_state(self):
        pass
    
    def _update_lr(self, lr):
        if hasattr(self, "param_groups"):
            for group in self.param_groups:
                group["lr"] = lr
        elif hasattr(self, "lr"):
            self.lr = lr

    def __repr__(self):
        raise NotImplementedError
