from ..base_module import Module
import mytorch.nn.functional as F

class GELU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = F.gelu(x)
        return y