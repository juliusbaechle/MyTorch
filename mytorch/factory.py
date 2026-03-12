from .tensor import Tensor
from .array import Array

def zeros(shape, requires_grad=False, device=None, dtype=None):
    data = Array.zeros(shape, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def ones(shape, requires_grad=False, device=None, dtype=None):
    data = Array.ones(shape, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def empty(shape, requires_grad=False, device=None, dtype=None):
    data = Array.empty(shape, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def full(shape, fill_value, requires_grad=False, device=None, dtype=None):
    data = Array.full(shape, fill_value, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def arange(start, end=None, step=1, requires_grad=False, device=None, dtype=None):
    data = Array.arange(start, end, step=step, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def linspace(start, end=None, num=50, requires_grad=False, device=None):
    data = Array.linspace(start, end, num=num, device=device)
    return Tensor(data, requires_grad=requires_grad)

def eye(N, M=None, k=0, requires_grad=False, device=None, dtype=None):
    data = Array.eye(N, M=M, k=k, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def randn(shape, requires_grad=False, device=None, dtype=None):
    data = Array.randn(shape, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def rand(shape, requires_grad=False, device=None, dtype=None):
    data = Array.rand(shape, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def randint(low, high, shape, requires_grad=False, device=None, dtype=None):
    data = Array.randint(low, high=high, shape=shape, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def tril(x, k=0, requires_grad=False, device=None, dtype=None):
    data = Array.tril(x.data, k=k, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def triu(x, k=0, requires_grad=False, device=None, dtype=None):
    data = Array.triu(x.data, k=k, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def zeros_like(other, requires_grad=False, device=None, dtype=None):
    data = Array.zeros_like(other.data, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def ones_like(other, requires_grad=False, device=None, dtype=None):
    data = Array.ones_like(other.data, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def full_like(other, fill_value, requires_grad=False, device=None, dtype=None): 
    data = Array.full_like(other.data, fill_value, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def randn_like(other, requires_grad=False, device=None, dtype=None):
    data = Array.randn_like(other.data, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def rand_like(other, requires_grad=False, device=None, dtype=None):
    data = Array.rand_like(other.data, device=device, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)