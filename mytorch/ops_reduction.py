from .tensor import Tensor
from .array import Array
import numpy as np

def sum(input, dim=None, keepdims=False):
    out_data = input.data.sum(axis=dim, keepdims=keepdims)

    def _sum_backward(grad):
        if input.requires_grad:
            grad_input = np.broadcast_to(grad, input.shape)
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_sum_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else None,
                    device=input.device)

def cumsum(input, dim=None):
    out_data = input.data.cumsum(axis=dim)

    def _cumsum_backward(grad):
        if input.requires_grad:
            grad_input = np.flip(grad, axis=dim)
            grad_input = np.cumsum(grad_input, axis=dim)
            grad_input = np.flip(grad_input, axis=dim)
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_cumsum_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else None,
                    device=input.device)

def mean(input, dim=None, keepdims=False):
    if dim is None:
        dim = tuple(range(input.ndim))
    if isinstance(dim, int):
        dim = (dim,)

    out_data = input.data.mean(axis=dim, keepdims=keepdims)

    def _mean_backward(grad):
        if input.requires_grad:
            grad_input = np.broadcast_to(grad, input.shape)
            grad_input /= np.float32(np.prod([input.shape[d] for d in dim]))
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_mean_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else None,
                    device=input.device)

def var(input, dim=None, keepdims=False):
    if dim is None:
        dim = tuple(range(input.ndim))
    if isinstance(dim, int):
        dim = (dim,)

    out_data = input.data.var(axis=dim, keepdims=keepdims)

    def _var_backward(grad):
        if input.requires_grad:
            mean = input.data.mean(axis=dim, keepdims=True)
            num_values = np.prod([input.shape[d] for d in dim])
            grad_input = 2 * (input.data - mean) * grad / num_values
            grad_input = np.broadcast_to(grad_input, input.shape)
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_var_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else None,
                    device=input.device)

def _max_min(input, dim=None, keepdims=False, is_max=True):
    if dim is None:
        dim = tuple(range(input.ndim))

    if is_max:
        out_data = np.max(input.data, axis=dim, keepdims=keepdims)
    else:
        out_data = np.min(input.data, axis=dim, keepdims=keepdims)

    def _backward(grad):
        if input.requires_grad:
            if dim is not None and not keepdims:
                grad = np.expand_dims(Array(grad), axis=dim)
                out_vals = np.expand_dims(Array(out_data), axis=dim)
            input_grad = grad * (input.data == out_vals)
            input._add_grad(input_grad)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else None,
                    device=input.device)

def max(input, dim=None, keepdims=False):
    return _max_min(input, dim, keepdims, is_max=True)

def min(input, dim=None, keepdims=False):
    return _max_min(input, dim, keepdims, is_max=False)