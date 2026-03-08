import math
import mytorch
from ..base_module import Module
import mytorch.nn.functional as F
from ... import initializations as init

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = mytorch.zeros((out_features, in_features), True, device, dtype)
        k = math.sqrt(1 / in_features)
        init.uniform_(self.weight, -k, k)

        if self.use_bias:
            self.bias = mytorch.zeros((out_features,), True, device, dtype)
            init.uniform_(self.bias)
        else:
            self.bias=None

    def forward(self, input):
        output = F.linear(input, weight=self.weight, bias=self.bias)
        return output