from ..tensor import Tensor
from ..factory import zeros_like, ones_like, rand_like, randn_like

def uniform_(tensor, low=0.0, high=1.0):
    arr = rand_like(tensor).data * (high - low) + low
    tensor.data = arr