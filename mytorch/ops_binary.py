from .tensor import Tensor
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

def add(input, val):
    input, val = _coerce_types(input, val, "add")
    input._check_broadcast(input, val)
    out_data = input.data + val.data

    def _add_backward(grad):
        if input.requires_grad:
            input_grad = input._broadcasted_grad_accumulate(input.shape, grad)
            input._add_grad(input_grad)
        if val.requires_grad:
            val_grad = input._broadcasted_grad_accumulate(val.shape, grad)
            val._add_grad(val_grad)

    out_requires_grad = (input.requires_grad or val.requires_grad) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_add_backward if out_requires_grad else None,
                    parents=(input, val) if out_requires_grad else None,
                    device=input.device)

def mul(input, val):
    input, val = _coerce_types(input, val, "mul")
    input._check_broadcast(input, val)
    out_data = input.data * val.data

    def _mul_backward(grad):
        if input.requires_grad:
            input_grad = grad * val.data
            input_grad = input._broadcasted_grad_accumulate(input.shape, input_grad)
            input._add_grad(input_grad)
        if val.requires_grad:
            val_grad = grad * input.data
            val_grad = input._broadcasted_grad_accumulate(val.shape, val_grad)
            val._add_grad(val_grad)

    out_requires_grad = (input.requires_grad or val.requires_grad) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_mul_backward if out_requires_grad else None,
                    parents=(input, val) if out_requires_grad else None,
                    device=input.device)

def matmul(input : Tensor, val : Tensor):
    if not isinstance(input, Tensor):
        raise TypeError(f"matmul(): argument 'input' (position 1) must be a Tensor, not {type(input)}")
    if not isinstance(val, Tensor):
        raise TypeError(f"matmul(): argument 'val' (position 2) must be a Tensor, not {type(val)}")
    
    out_data = input.data @ val.data
    
    def _matmul_backward(grad):
        if input.requires_grad:
            grad_input = grad @ val.data.T
            input._add_grad(grad_input)
        if val.requires_grad:
            grad_val = input.data.T @ grad
            val._add_grad(grad_val)

    out_requires_grad = (input.requires_grad or val.requires_grad) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_matmul_backward if out_requires_grad else None,
                    parents=(input, val) if out_requires_grad else None,
                    device=input.device)

def divide(input, val):
    input, val = _coerce_types(input, val, "divide")
    input._check_broadcast(input, val)
    out_data = input.data / val.data

    def _truediv_backward(grad):
        if input.requires_grad:
            input_grad = grad / val.data
            input_grad = input._broadcasted_grad_accumulate(input.shape, input_grad)
            input._add_grad(input_grad)
        if val.requires_grad:
            val_grad = -grad * input.data / val.data ** 2
            val_grad = input._broadcasted_grad_accumulate(val.shape, val_grad)
            val._add_grad(val_grad)

    out_requires_grad = (input.requires_grad or val.requires_grad) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_truediv_backward if out_requires_grad else None,
                    parents=(input, val) if out_requires_grad else None,
                    device=input.device)