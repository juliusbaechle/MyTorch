from .tensor import Tensor
import numpy as np

def transpose(input : Tensor, dim1, dim2):
    return input.transpose(dim1, dim2)

def permute(input : Tensor, *dims):
    return input.permute(dims)

def reshape(input : Tensor, *shape):
    return input.reshape(shape)

def exp(input : Tensor):
    return input.exp()

def log(input : Tensor):
    return input.log()

def sum(input : Tensor, dim=None, keepdims=False):
    return input.sum(dim, keepdims)

def cumsum(input : Tensor, dim=None):
    return input.cumsum(dim)

def mean(input : Tensor, dim=None, keepdims=False):
    return input.mean(dim, keepdims)

def var(input : Tensor, dim=None, keepdims=False):
    return input.var(dim, keepdims)

def max(input : Tensor, dim=None, keepdims=False):
    return input.max(dim, keepdims)

def argmax(input : Tensor, dim=None):
    return input.argmax(dim)

def masked_fill(input : Tensor, mask, value):
    return input.masked_fill(mask, value)

def abs(input : Tensor):
    return input.abs()

def clamp(input : Tensor, min, max):
    return input.clamp(min, max)

def sqrt(input : Tensor):
    return input.sqrt()

def chunk(input : Tensor, chunks, dim=0):
    return input.chunk(chunks=chunks, dim=dim)

def concatenate(tensors, dim=0):
    if len(tensors) == 0:
        raise ValueError("No tensors to concatenate")
    
    tensor_list = [t.data for t in tensors]
    out_data = np.concatenate(tensor_list, axis=dim)
    sizes = [t.shape[dim] for t in tensors]

    def _concat_backward(input_grad):
        offset = 0
        for t, size in zip(tensors, sizes):
            if t.requires_grad:
                grad_idx = [slice(None)] * out_data.ndim
                grad_idx[dim] = slice(offset, offset + size)
                t._add_grad(input_grad[tuple(grad_idx)])
            offset += size

    out_requires_grad = any(t.requires_grad for t in tensors) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_concat_backward if out_requires_grad else None,
                    parents=tensors if out_requires_grad else None,
                    device=tensors[0].device)

def stack(tensors, dim=0):
    if len(tensors) == 0:
        raise ValueError("No tensors to stack")
    
    tensor_list = [t.data for t in tensors]
    out_data = np.stack(tensor_list, axis=dim)

    def _stack_backward(input_grad):
        for i, t in enumerate(tensors):
            if t.requires_grad:
                grad_idx = [slice(None)] * out_data.ndim
                grad_idx[dim] = i
                t._add_grad(input_grad[tuple(grad_idx)])

    out_requires_grad = any(t.requires_grad for t in tensors) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_stack_backward if out_requires_grad else None,
                    parents=tensors if out_requires_grad else None,
                    device=tensors[0].device)