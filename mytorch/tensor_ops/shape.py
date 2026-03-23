from ..tensor import Tensor
from ..array import Array
import numpy as np

def get_item(input, idx):
    if isinstance(idx, Tensor):
        idx = idx.data

    output = input.data[idx]

    def _getitem_backward(grad):
        if input.requires_grad:
            assert np.unique(idx).size == len(idx)
            input_grad = Array.zeros_like(input.data)
            input_grad[idx] = grad
            input._add_grad(input_grad)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(output,
                    requires_grad=out_requires_grad,
                    grad_fn=_getitem_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else (),
                    device=input.device)

def set_item(input, idx, value):
    if isinstance(idx, Tensor):
        idx = idx.data
    if isinstance(value, Tensor):
        value = value.data
    input.data[idx] = value

def permute(input, *dims):
    output = np.transpose(input.data, axes=dims)

    def _permute_backward(grad):
        if input.requires_grad:
            inverse_dims = np.argsort(dims)  # (2, 0, 1) -> (1, 2, 0)
            input_grad = np.transpose(grad, axes=inverse_dims)
            input._add_grad(input_grad)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(output,
                    requires_grad=out_requires_grad,
                    grad_fn=_permute_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else (),
                    device=input.device)

def transpose(input, dim1=-1, dim2=-2):
    axes = list(range(input.ndim))
    dim1 %= input.ndim
    dim2 %= input.ndim
    axes[dim1] = dim2
    axes[dim2] = dim1
    return input.permute(*axes)

def reshape(input, *shape):
    output = input.data.reshape(shape)

    def _reshape_backward(grad):
        if input.requires_grad:
            input_grad = np.reshape(grad, input.shape)
            input._add_grad(input_grad)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(output,
                    requires_grad=out_requires_grad,
                    grad_fn=_reshape_backward if out_requires_grad else None,
                    parents=input if out_requires_grad else (),
                    device=input.device)

def broadcast_to(input, shape):
    if input.shape == shape:
        return input
    
    output = np.broadcast_to(input.data, shape)

    def _broadcast_backward(grad): 
        if input.requires_grad:
            added_axes = tuple(i for i in range(grad.ndim - input.ndim))
            input_grad = grad.sum(axis=added_axes, keepdims=False)
            expanded_axes = tuple(i for i in range(input_grad.ndim) if input_grad.shape[i] != input.shape[i])
            input_grad = input_grad.sum(axis=expanded_axes, keepdims=True)
            input._add_grad(input_grad)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(output,
                    requires_grad=out_requires_grad,
                    grad_fn=_broadcast_backward if out_requires_grad else None,
                    parents=input if out_requires_grad else (),
                    device=input.device)

def flatten(input, start_dim=0, end_dim=-1):
    start_dim %= input.data.ndim
    end_dim %= input.data.ndim
    new_shape = input.shape[:start_dim] + (-1,) + input.shape[end_dim+1:]
    return input.reshape(*new_shape)

def squeeze(input, dim = None):
    if dim is None:
        new_shape = tuple(s for s in input.shape if s != 1)
    else:
        dim %= input.data.ndim
        if input.shape[dim] != 1:
            raise ValueError(f"Cannot squeeze dimension {dim} with size {input.shape[dim]}")
        new_shape = input.shape[:dim] + input.shape[dim+1:]
    return input.reshape(*new_shape)

def unsqueeze(input, dim = 0):
    dim %= input.data.ndim + 1
    new_shape = input.shape[:dim] + (1,) + input.shape[dim:]
    return input.reshape(*new_shape)

def chunk(input, chunks, dim=0):
    dim %= input.data.ndim
    size = input.shape[dim]
    if size % chunks != 0:
        raise ValueError(f"Cannot split dimension of size {size} into {chunks} chunks")
    chunk_size = size // chunks

    out_tensors = []
    for i in range(chunks):
        start, end = i*chunk_size, (i+1)*chunk_size
        idx = [slice(None)] * input.ndim
        idx[dim] = slice(start, end)
        output = input.data[tuple(idx)]

        def _chunk_backward(grad, start=start, end=end):
            if input.requires_grad:
                input_grad = Array.zeros_like(input.data)
                grad_idx = [slice(None)] * input.ndim
                grad_idx[dim] = slice(start, end)
                input_grad[tuple(grad_idx)] = grad
                input._add_grad(input_grad)

        out_requires_grad = input.requires_grad and Tensor._build_graph
        output = Tensor(output,
                        requires_grad=out_requires_grad,
                        grad_fn=_chunk_backward if out_requires_grad else None,
                        parents=(input,) if out_requires_grad else (),
                        device=input.device)            
        out_tensors.append(output)
    return out_tensors