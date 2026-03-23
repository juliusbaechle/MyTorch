from ..tensor import Tensor
import math
import numpy as np
from ..factory import rand_like

def uniform_(tensor, low=0.0, high=1.0):
    arr = rand_like(tensor).data * (high - low) + low
    tensor.data = arr

def kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan = _calculate_fan(tensor.shape, mode)
    gain = calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0) * gain / math.sqrt(fan)
    uniform_(tensor, -bound, bound)

def _calculate_fan_in_out(shape):
    if len(shape) == 2:  # linear layer weight
        fan_in, fan_out = shape[1], shape[0]
    elif len(shape) in {3, 4, 5}:  # conv weights
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        fan_in = fan_out = 1
    return fan_in, fan_out

def _calculate_fan(shape, mode="fan_in"):
    fan_in, fan_out = _calculate_fan_in_out(shape)
    return fan_in if mode == "fan_in" else fan_out

def calculate_gain(nonlinearity, param=None):
    if nonlinearity == "linear" or nonlinearity == "conv1d":
        return 1.0
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1 + param**2)) if param is not None else math.sqrt(2.0)
    return 1.0