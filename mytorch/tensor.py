from __future__ import annotations
import numpy as np
import weakref
from .topo_sort import build_topo
from mytorch.array import Array
from mytorch.dtypes import *

class Tensor:
    _build_graph = True

    def __init__(self, 
                 data, 
                 requires_grad=False, 
                 grad_fn=None,
                 device=None, 
                 dtype=None,
                 parents=()):
        self._data = Array(data, device, dtype)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None
        self._is_leaf = requires_grad and (grad_fn is None)
        self._set_parents(parents)

    @property
    def dtype(self):
        return self._data.dtype
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = Array(value)

    @property
    def device(self):
        return self._data.device
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def size(self):
        return self._data.size
    
    @property
    def ndim(self):
        return self._data.ndim

    @property
    def is_leaf(self):
        return self._is_leaf
    
    def item(self):
        if self.size > 1:
            raise RuntimeError("Item only works for size 1")
        return self.data._array.item()

    @classmethod
    def build_graph_enabled(cls):
        return cls._build_graph
    
    def __repr__(self): ...
    
    def to(self, device):
        self.data = self.data.to(device)
        return self

    def _add_grad(self, grad):
        assert grad.shape == self.shape
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        grad = None

    def _set_parents(self, parents):
        if not isinstance(parents, (list, tuple)):
            parents = (parents,)
        self._parents = tuple(weakref.ref(p) for p in parents)

    def backward(self, grad=None, retain_graph=False):
        if grad is None:
            grad = Array.ones_like(self.data, dtype=self.dtype, device=self.device)
        self.grad = grad

        topo = build_topo(self)
        for i, t in enumerate(reversed(topo)):
            if t.grad_fn is not None:
                t.grad_fn(t.grad)

                if not retain_graph:
                    t.grad_fn = None
                    t._parents = ()
                    if not t.is_leaf:
                        t.grad = None

    ### BINARY OPS ###

    def __add__(self, val): ...    
    def __radd__(self, val): return self + val
    def __sub__(self, val): return self + -val    
    def __rsub__(self, val): return -self + val
    
    def __mul__(self, val): ...    
    def __rmul__(self, val): return self * val
    def __neg__(self): return self * -1
    
    def __matmul__(self, val): ...
    
    def __truediv__(self, val): ...
    def __rtruediv__(self, val): return val / self
    
    ### UNARY OPS ###
    def __pow__(self, exponent): ...
    def sqrt(self):
        if (self < 0).data.any():
            raise ValueError("Cannot compute sqrt of negative values")
        return self ** 0.5
    
    def sin(self): ...
    def cos(self): ...
    def tan(self): ...

    def exp(self): ...
    def log(self): ...
    def abs(self): ...
    def clamp(self, min_val = None, max_val = None): ...
    
    def sigmoid(self): ...

    def _compare(self, other, op):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(op(self.data, other_data), requires_grad=False, device=self.device)
    
    def __eq__(self, other):
        return self._compare(other, lambda a,b: a == b)
    
    def __ne__(self, other):
        return self._compare(other, lambda a,b: a != b)
    
    def __lt__(self, other):
        return self._compare(other, lambda a,b: a < b)
    
    def __le__(self, other):
        return self._compare(other, lambda a,b: a <= b)
    
    def __gt__(self, other):
        return self._compare(other, lambda a,b: a > b)
    
    def __ge__(self, other):
        return self._compare(other, lambda a,b: a >= b)
    
    def any(self):
        return self.data.any()
    
    def all(self):
        return self.data.all()
    
    ### INDEXING / RESHAPING ###

    def __getitem__(self, idx): ...        
    def __setitem__(self, idx, value): ...
    def permute(self, *dims): ...
    def transpose(self, dim1=-1, dim2=-2): ...
    def reshape(self, *shape): ...
    def broadcast_to(self, shape): ...    
    def flatten(self, start_dim=0, end_dim=-1): ...    
    def squeeze(self, dim = None): ...
    def unsqueeze(self, dim = 0): ...
    def chunk(self, chunks, dim=0): ...
    
    ### REDUCTIONS ###

    def sum(self, dim=None, keepdims=False): ...
    def cumsum(self, dim=None): ...    
    def mean(self, dim=None, keepdims=False): ...    
    def var(self, dim=None, keepdims=False): ...    
    def max(self, dim=None, keepdims=False): ...
    def min(self, dim=None, keepdims=False): ...

    def argmax(self, dim=-1): 
        return Tensor(np.argmax(self.data, axis=dim), device=self.device)
    
    def argmin(self, dim=-1): 
        return Tensor(np.argmin(self.data, axis=dim), device=self.device)
    
    ### OTHER OPS ###

    def masked_fill(self, mask, value): ...    
    def sort(self, dim=-1, descending=False): ...
    def argsort(self, dim=-1, descending=False): ...
    
    def item(self):
        if self.size != 1:
            raise ValueError("Only tensors with one element can be converted to a Python scalar")
        return self.data.item()
    
    def astype(self, dtype):
        self.data = self.data.astype(dtype)
        return self
    
    def detatch(self):
        output = Tensor(self.data, requires_grad=False, device=self.device)
        return output
    
    def numpy(self):
        return self.data.asnumpy()
    
    def __len__(self):
        return self.shape[0]

from .tensor_ops.binary import *
Tensor.__add__ = add
Tensor.__mul__ = mul
Tensor.__matmul__ = matmul
Tensor.__truediv__ = divide

from .tensor_ops.unary import *
Tensor.__pow__ = pow
Tensor.sin = sin
Tensor.cos = cos
Tensor.tan = tan
Tensor.exp = exp
Tensor.log = log
Tensor.abs = abs
Tensor.clamp = clamp
Tensor.sigmoid = sigmoid

from .tensor_ops.repr import *
Tensor.__repr__ = tensor2string

from .tensor_ops.reduction import *
Tensor.sum = sum
Tensor.cumsum = cumsum
Tensor.mean = mean
Tensor.var = var
Tensor.max = max
Tensor.min = min

from .tensor_ops.shape import *
Tensor.__getitem__ = get_item
Tensor.__setitem__ = set_item
Tensor.permute = permute
Tensor.transpose = transpose
Tensor.reshape = reshape
Tensor.broadcast_to = broadcast_to
Tensor.flatten = flatten
Tensor.squeeze = squeeze
Tensor.unsqueeze = unsqueeze
Tensor.chunk = chunk

from .tensor_ops.other import *
Tensor.masked_fill = masked_fill
Tensor.sort = sort
Tensor.argsort = argsort