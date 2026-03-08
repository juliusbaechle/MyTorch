from .base_optimizer import Optimizer
from typing import Dict, Any

class LRScheduler():
    def __init__(self, optimizer : Optimizer, last_epoch : int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        
    def get_last_lr(self):
        return self.optimizer.lr
    
    def get_lr(self):
        raise NotImplementedError()

    def state_dict():
        raise NotImplementedError()
    
    def load_state_dict(state_dict : Dict[str, Any]):
        raise NotImplementedError()
    
    def step():
        raise NotImplementedError()