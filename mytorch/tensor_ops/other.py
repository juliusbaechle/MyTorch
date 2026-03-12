from ..tensor import Tensor
from ..array import Array
import numpy as np

def masked_fill(input, mask, value):
    out_data = np.where(mask.data, value, input.data)

    def _masked_fill_backward(grad):
        if input.requires_grad:
            input_grad = np.where(mask.data, 0, grad)
            input._add_grad(input_grad)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_masked_fill_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else (),
                    device=input.device)

def sort(input, dim=-1, descending=False):
    out_data = np.sort(input.data, axis=dim)
    if descending:
        out_data = np.flip(out_data, axis=dim)

    def _sort_backward(grad):
        if input.requires_grad:
            sorted_indices = np.argsort(input.data, axis=dim)
            if descending:
                sorted_indices = np.flip(sorted_indices, axis=dim)
            input_grad = np.zeros_like(input.data)
            np.put_along_axis(input_grad, sorted_indices, grad, axis=dim)
            input._add_grad(input_grad)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_sort_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else (),
                    device=input.device)

def argsort(input, dim=-1, descending=False):
    out_data = np.argsort(input.data, axis=dim)
    if descending:
        out_data = np.flip(out_data, axis=dim)

    assert input.requires_grad == False
    return Tensor(out_data, device=input.device)

def concatenate(tensors, dim=0):
    if len(tensors) == 0:
        raise ValueError("No tensors to concatenate")
    
    tensor_list = [t.data for t in tensors]
    out_data = np.concatenate(tensor_list, axis=dim)
    sizes = [t.shape[dim] for t in tensors]

    def _concat_backward(grad):
        offset = 0
        for t, size in zip(tensors, sizes):
            if t.requires_grad:
                grad_idx = [slice(None)] * out_data.ndim
                grad_idx[dim] = slice(offset, offset + size)
                t._add_grad(grad[tuple(grad_idx)])
            offset += size

    out_requires_grad = any(t.requires_grad for t in tensors) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_concat_backward if out_requires_grad else None,
                    parents=tensors if out_requires_grad else (),
                    device=tensors[0].device)

def stack(tensors, dim=0):
    if len(tensors) == 0:
        raise ValueError("No tensors to stack")
    
    tensor_list = [t.data for t in tensors]
    out_data = np.stack(tensor_list, axis=dim)

    def _stack_backward(grad):
        for i, t in enumerate(tensors):
            if t.requires_grad:
                grad_idx = [slice(None)] * out_data.ndim
                grad_idx[dim] = i
                t._add_grad(grad[tuple(grad_idx)])

    out_requires_grad = any(t.requires_grad for t in tensors) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_stack_backward if out_requires_grad else None,
                    parents=tensors if out_requires_grad else (),
                    device=tensors[0].device)