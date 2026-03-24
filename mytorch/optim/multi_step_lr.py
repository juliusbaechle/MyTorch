from .scheduler import LRScheduler
from .base_optimizer import Optimizer

class MultiStepLR(LRScheduler):
    def __init__(self, optimizer : Optimizer, milestones : list[int], gamma : float):
        super().__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma
        self.step_count = 0
        self.initial_lr = optimizer.lr

    def get_lr(self):
        return self.optimizer.lr

    def step(self):
        self.step_count += 1
        exponent = 0
        while self.step_count >= self.milestones[exponent]:
            exponent += 1
        self.optimizer.lr = self.initial_lr * self.gamma ** exponent