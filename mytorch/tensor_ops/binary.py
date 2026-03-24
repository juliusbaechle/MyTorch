from ..tensor import Tensor
import numpy as np

def _coerce_types(input, val, func_name):
    if isinstance(input, Tensor) and np.isscalar(val):
        val = Tensor(val, device=input.device, dtype=input.dtype) 
    if isinstance(val, Tensor) and np.isscalar(input):
        input = Tensor(input, device=val.device, dtype=val.dtype) 
    if np.isscalar(input) and np.isscalar(val):
        input = Tensor(input)
        val = Tensor(val)
    if not isinstance(input, Tensor):
        raise TypeError(f"{func_name}(): argument 'input' (position 1) must be a Tensor, not {type(input).__name__}")
    if not isinstance(val, Tensor):
        raise TypeError(f"{func_name}(): argument 'val' (position 2) must be a Tensor, not {type(val).__name__}")
    return input, val

def _broadcast_tensors(input, val):
    if input.shape != val.shape:
        try:
            broadcast_shape = np.broadcast_shapes(input.shape, val.shape)
        except ValueError:
            raise ValueError(f"operands could not be broadcast together with shapes {input.shape} and {val.shape}")
        input = input.broadcast_to(broadcast_shape)
        val = val.broadcast_to(broadcast_shape)
    return input, val

def add(input, val):
    input, val = _coerce_types(input, val, "add")
    input, val = _broadcast_tensors(input, val)
    out_data = input.data + val.data

    def _add_backward(grad):
        if input.requires_grad:
            input._add_grad(grad)
        if val.requires_grad:
            val._add_grad(grad)

    out_requires_grad = (input.requires_grad or val.requires_grad) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_add_backward if out_requires_grad else None,
                    parents=(input, val) if out_requires_grad else (),
                    device=input.device)

def mul(input, val):
    input, val = _coerce_types(input, val, "mul")
    input, val = _broadcast_tensors(input, val)
    out_data = input.data * val.data

    def _mul_backward(grad):
        if input.requires_grad:
            input._add_grad(grad * val.data)
        if val.requires_grad:
            val._add_grad(grad * input.data)

    out_requires_grad = (input.requires_grad or val.requires_grad) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_mul_backward if out_requires_grad else None,
                    parents=(input, val) if out_requires_grad else (),
                    device=input.device)

def matmul(input : Tensor, val : Tensor):
    if not isinstance(input, Tensor):
        raise TypeError(f"matmul(): argument 'input' (position 1) must be a Tensor, not {type(input).__name__}")
    if not isinstance(val, Tensor):
        raise TypeError(f"matmul(): argument 'val' (position 2) must be a Tensor, not {type(val).__name__}")
    
    out_data = input.data @ val.data
    
    def _matmul_backward(grad):
        if input.requires_grad:
            grad_input = grad @ np.swapaxes(val.data, -1, -2)
            input._add_grad(grad_input)
        if val.requires_grad:
            grad_val = np.swapaxes(input.data, -1, -2) @ grad
            val._add_grad(grad_val)

    out_requires_grad = (input.requires_grad or val.requires_grad) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_matmul_backward if out_requires_grad else None,
                    parents=(input, val) if out_requires_grad else (),
                    device=input.device)

def divide(input, val):
    input, val = _coerce_types(input, val, "divide")    
    input, val = _broadcast_tensors(input, val)
    out_data = input.data / val.data

    def _truediv_backward(grad):
        if input.requires_grad:
            input_grad = grad / val.data
            input._add_grad(input_grad)
        if val.requires_grad:
            val_grad = -grad * input.data / val.data ** 2
            val._add_grad(val_grad)

    out_requires_grad = (input.requires_grad or val.requires_grad) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_truediv_backward if out_requires_grad else None,
                    parents=(input, val) if out_requires_grad else (),
                    device=input.device)