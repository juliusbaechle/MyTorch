import numpy as np
from mytorch import Tensor

def relu(input : Tensor):
    return input.clamp(0)