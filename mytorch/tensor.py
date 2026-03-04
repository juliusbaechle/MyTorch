from __future__ import annotations
import numpy as np
import weakref
from functools import wraps
from contextlib import contextmanager
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
                 parents=None):
        self._data = Array(data, device, dtype)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None
        self._is_leaf = self.requires_grad and (self.grad_fn is None)
        self._parents = self._set_parents(parents)

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
    
    @classmethod
    def build_graph_enabled(cls):
        return cls._build_graph
    
    def __repr__(self): ...
    
    def to(self, device):
        self.data = self.data.to(device)
        return self

    def _add_grad(self, grad):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        grad = None

    def _set_parents(self, parents):
        if not isinstance(parents, (list, tuple)):
            parents = (parents)
        if parents is not None:
            self._parents = tuple(weakref.ref(p) for p in parents if p is not None)

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

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.data

        out_data = self.data[idx]

        def _getitem_backward(grad):
            if self.requires_grad:
                assert np.unique(idx).size == len(idx)
                self_grad = Array.zeros_like(self.data)
                self_grad[idx] = grad
                self._add_grad(self_grad)

        out_requires_grad = self.requires_grad and Tensor._build_graph
        return Tensor(out_data,
                        requires_grad=out_requires_grad,
                        grad_fn=_getitem_backward if out_requires_grad else None,
                        parents=(self,) if out_requires_grad else None,
                        device=self.device)
    
    def __setitem__(self, idx, value):
        if isinstance(idx, Tensor):
            idx = idx.data
        if isinstance(value, Tensor):
            value = value.data
        self.data[idx] = value
    
    def permute(self, *dims):
        out_data = np.transpose(self.data, axes=dims)

        def _permute_backward(grad):
            if self.requires_grad:
                inverse_dims = np.argsort(dims)  # (2, 0, 1) -> (1, 2, 0)
                self_grad = np.transpose(grad, axes=inverse_dims)
                self._add_grad(self_grad)

        out_requires_grad = self.requires_grad and Tensor._build_graph
        return Tensor(out_data,
                        requires_grad=out_requires_grad,
                        grad_fn=_permute_backward if out_requires_grad else None,
                        parents=(self,) if out_requires_grad else None,
                        device=self.device)

    def transpose(self, dim1=-1, dim2=-2):
        axes = list(range(self.ndim))
        dim1 %= self.ndim
        dim2 %= self.ndim
        axes[dim1] = dim2
        axes[dim2] = dim1
        return self.permute(*axes)

    def reshape(self, *shape):
        out_data = self.data.reshape(shape)

        def _reshape_backward(grad):
            if self.requires_grad:
                self_grad = grad.reshape(self.shape)
                self._add_grad(self_grad)

        out_requires_grad = self.requires_grad and Tensor._build_graph
        return Tensor(out_data,
                        requires_grad=out_requires_grad,
                        grad_fn=_reshape_backward if out_requires_grad else None,
                        parents=self if out_requires_grad else None,
                        device=self.device)
    
    def broadcast_to(self, shape):
        if self.shape == shape:
            return self
        
        out_data = np.broadcast_to(self.data, shape)

        def _broadcast_backward(grad): 
            if self.requires_grad:
                added_axes = tuple(i for i in range(grad.ndim - self.ndim))
                self_grad = grad.sum(axis=added_axes, keepdims=False)
                expanded_axes = tuple(i for i in range(self_grad.ndim) if self_grad.shape[i] != self.shape[i])
                self_grad = self_grad.sum(axis=expanded_axes, keepdims=True)
                self._add_grad(self_grad)

        out_requires_grad = self.requires_grad and Tensor._build_graph
        return Tensor(out_data,
                        requires_grad=out_requires_grad,
                        grad_fn=_broadcast_backward if out_requires_grad else None,
                        parents=self if out_requires_grad else None,
                        device=self.device)
    
    def flatten(self, start_dim=0, end_dim=-1):
        start_dim %= self.data.ndim
        end_dim %= self.data.ndim
        new_shape = self.shape[:start_dim] + (-1,) + self.shape[end_dim+1:]
        return self.reshape(*new_shape)
    
    def squeeze(self, dim = None):
        if dim is None:
            new_shape = tuple(s for s in self.shape if s != 1)
        else:
            dim %= self.data.ndim
            if self.shape[dim] != 1:
                raise ValueError(f"Cannot squeeze dimension {dim} with size {self.shape[dim]}")
            new_shape = self.shape[:dim] + self.shape[dim+1:]
        return self.reshape(*new_shape)
    
    def unsqueeze(self, dim = 0):
        dim %= self.data.ndim + 1
        new_shape = self.shape[:dim] + (1,) + self.shape[dim:]
        return self.reshape(*new_shape)
    
    def chunk(self, chunks, dim=0):
        dim %= self.data.ndim
        size = self.shape[dim]
        if size % chunks != 0:
            raise ValueError(f"Cannot split dimension of size {size} into {chunks} chunks")
        chunk_size = size // chunks

        out_tensors = []
        for i in range(chunks):
            start, end = i*chunk_size, (i+1)*chunk_size
            idx = [slice(None)] * self.ndim
            idx[dim] = slice(start, end)
            out_data = self.data[tuple(idx)]

            def _chunk_backward(grad, start=start, end=end):
                if self.requires_grad:
                    self_grad = Array.zeros_like(self.data)
                    grad_idx = [slice(None)] * self.ndim
                    grad_idx[dim] = slice(start, end)
                    self_grad[tuple(grad_idx)] = grad
                    self._add_grad(self_grad)

            out_requires_grad = self.requires_grad and Tensor._build_graph
            output = Tensor(out_data,
                            requires_grad=out_requires_grad,
                            grad_fn=_chunk_backward if out_requires_grad else None,
                            parents=(self,) if out_requires_grad else None,
                            device=self.device)            
            out_tensors.append(output)
        return out_tensors
    
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

    def masked_fill(self, mask, value):
        out_data = np.where(mask.data, value, self.data)

        def _masked_fill_backward(grad):
            if self.requires_grad:
                self_grad = np.where(mask.data, 0, grad)
                self._add_grad(self_grad)

        out_requires_grad = self.requires_grad and Tensor._build_graph
        return Tensor(out_data,
                        requires_grad=out_requires_grad,
                        grad_fn=_masked_fill_backward if out_requires_grad else None,
                        parents=(self,) if out_requires_grad else None,
                        device=self.device)
    
    def sort(self, dim=-1, descending=False):
        out_data = np.sort(self.data, axis=dim)
        if descending:
            out_data = np.flip(out_data, axis=dim)

        def _sort_backward(grad):
            if self.requires_grad:
                sorted_indices = np.argsort(self.data, axis=dim)
                if descending:
                    sorted_indices = np.flip(sorted_indices, axis=dim)
                self_grad = np.zeros_like(self.data)
                np.put_along_axis(self_grad, sorted_indices, grad, axis=dim)
                self._add_grad(self_grad)

        out_requires_grad = self.requires_grad and Tensor._build_graph
        return Tensor(out_data,
                        requires_grad=out_requires_grad,
                        grad_fn=_sort_backward if out_requires_grad else None,
                        parents=(self,) if out_requires_grad else None,
                        device=self.device)
    
    def argsort(self, dim=-1, descending=False):
        out_data = np.argsort(self.data, axis=dim)
        if descending:
            out_data = np.flip(out_data, axis=dim)

        def _argsort_backward(input_grad):
            if self.requires_grad:
                return Array.zeros_like(self.data)

        out_requires_grad = self.requires_grad and Tensor._build_graph
        return Tensor(out_data,
                        requires_grad=out_requires_grad,
                        grad_fn=_argsort_backward if out_requires_grad else None,
                        parents=(self,) if out_requires_grad else None,
                        device=self.device)
    
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

from .ops_binary import *
Tensor.__add__ = add
Tensor.__mul__ = mul
Tensor.__matmul__ = matmul
Tensor.__truediv__ = divide

from .ops_unary import *
Tensor.__pow__ = pow
Tensor.sin = sin
Tensor.cos = cos
Tensor.tan = tan
Tensor.exp = exp
Tensor.log = log
Tensor.abs = abs
Tensor.clamp = clamp

from .ops_repr import *
Tensor.__repr__ = tensor2string

from .ops_reduction import *
Tensor.sum = sum
Tensor.cumsum = cumsum
Tensor.mean = mean
Tensor.var = var
Tensor.max = max
Tensor.min = min