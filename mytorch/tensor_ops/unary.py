from ..tensor import Tensor
import numpy as np

def pow(input, exponent):
    if not isinstance(exponent, Tensor):
        exponent = Tensor(exponent, dtype=input.dtype, device=input.device)
    if exponent.ndim != 0:
        raise ValueError("Exponent must be a scalar")
    out_data = input.data ** exponent.data

    def _pow_backward(grad):
        if input.requires_grad:
            grad_input = grad * exponent.data * (input.data ** (exponent.data - 1))
            input._add_grad(grad_input)
        if exponent.requires_grad:
            grad_exponent = grad * input.data ** exponent.data * np.log(input.data)
            exponent._add_grad(grad_exponent.sum())

    out_requires_grad = (input.requires_grad or exponent.requires_grad) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_pow_backward if out_requires_grad else None,
                    parents=(input, exponent) if out_requires_grad else (),
                    device=input.device)

def sin(input : Tensor):
    output = np.sin(input.data)

    def _sin_backward(grad):
        if input.requires_grad:
            grad_input = grad * np.cos(input.data)
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(output,
            requires_grad=out_requires_grad,
            grad_fn=_sin_backward if out_requires_grad else None,
            parents=input if out_requires_grad else (),
            device=input.device)

def cos(input : Tensor):
    output = np.cos(input.data)

    def _cos_backward(grad):
        if input.requires_grad:
            grad_input = -grad * np.sin(input.data)
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(output,
            requires_grad=out_requires_grad,
            grad_fn=_cos_backward if out_requires_grad else None,
            parents=input if out_requires_grad else (),
            device=input.device)

def tan(input : Tensor):
    output = np.tan(input.data)

    def _tan_backward(grad):
        if input.requires_grad:
            grad_input = grad * (1 + output ** 2)
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(output,
            requires_grad=out_requires_grad,
            grad_fn=_tan_backward if out_requires_grad else None,
            parents=input if out_requires_grad else (),
            device=input.device)

def exp(input):
    out_data = np.exp(input.data)

    def _exp_backward(grad):
        if input.requires_grad:
            grad_input = grad * out_data
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_exp_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else (),
                    device=input.device)

def log(input):
    out_data = np.log(input.data)

    def _log_backward(grad):
        if input.requires_grad:
            grad_input = grad / input.data
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_log_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else (),
                    device=input.device)

def abs(input):
    out_data = np.abs(input.data)

    def _abs_backward(grad):
        if input.requires_grad:
            grad_input = grad * np.sign(input.data)
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_abs_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else (),
                    device=input.device)

def clamp(input, min_val = None, max_val = None):
    out_data = np.clip(input.data, min_val, max_val)

    def _clamp_backward(grad):
        if input.requires_grad:
            grad_input = grad
            if min_val is not None:
                grad_input = np.where(input.data < min_val, 0, grad_input)
            if max_val is not None:
                grad_input = np.where(input.data > max_val, 0, grad_input)
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_clamp_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else (),
                    device=input.device)

def sigmoid(input):
    output = 1 / (1 + np.exp(-input.data))

    def _sigmoid_backward(grad):
        if input.requires_grad:
            grad_input = grad * output * (1 - output)
            input._add_grad(grad_input)

    out_requires_grad = input.requires_grad and Tensor._build_graph
    return Tensor(output,
                    requires_grad=out_requires_grad,
                    grad_fn=_sigmoid_backward if out_requires_grad else None,
                    parents=(input,) if out_requires_grad else (),
                    device=input.device)