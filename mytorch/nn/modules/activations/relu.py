from ..base_module import Module
import mytorch.nn.functional as F

class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = F.relu(x)
        return y