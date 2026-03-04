from .tensor import Tensor
import numpy as np

def pow(self, exponent):
    if not isinstance(exponent, Tensor):
        exponent = Tensor(exponent, dtype=self.dtype, device=self.device)
    if exponent.ndim != 0:
        raise ValueError("Exponent must be a scalar")
    out_data = self.data ** exponent.data

    def _pow_backward(grad):
        if self.requires_grad:
            grad_self = grad * exponent.data * (self.data ** (exponent.data - 1))
            self._add_grad(grad_self)
        if exponent.requires_grad:
            grad_exponent = grad * self.data ** exponent.data * np.log(self.data)
            exponent._add_grad(grad_exponent.sum())

    out_requires_grad = (self.requires_grad or exponent.requires_grad) and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_pow_backward if out_requires_grad else None,
                    parents=(self, exponent) if out_requires_grad else None,
                    device=self.device)

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
            parents=input if out_requires_grad else None,
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
            parents=input if out_requires_grad else None,
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
            parents=input if out_requires_grad else None,
            device=input.device)

def exp(self):
    out_data = np.exp(self.data)

    def _exp_backward(grad):
        if self.requires_grad:
            grad_self = grad * out_data
            self._add_grad(grad_self)

    out_requires_grad = self.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_exp_backward if out_requires_grad else None,
                    parents=(self,) if out_requires_grad else None,
                    device=self.device)

def log(self):
    out_data = np.log(self.data)

    def _log_backward(grad):
        if self.requires_grad:
            grad_self = grad / self.data
            self._add_grad(grad_self)

    out_requires_grad = self.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_log_backward if out_requires_grad else None,
                    parents=(self,) if out_requires_grad else None,
                    device=self.device)

def abs(self):
    out_data = np.abs(self.data)

    def _abs_backward(grad):
        if self.requires_grad:
            grad_self = grad * np.sign(self.data)
            self._add_grad(grad_self)

    out_requires_grad = self.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_abs_backward if out_requires_grad else None,
                    parents=(self,) if out_requires_grad else None,
                    device=self.device)

def clamp(self, min_val = None, max_val = None):
    out_data = np.clip(self.data, min_val, max_val)

    def _clamp_backward(grad):
        if self.requires_grad:
            grad_self = grad
            if min_val is not None:
                grad_self = np.where(self.data < min_val, 0, grad_self)
            if max_val is not None:
                grad_self = np.where(self.data > max_val, 0, grad_self)
            self._add_grad(grad_self)

    out_requires_grad = self.requires_grad and Tensor._build_graph
    return Tensor(out_data,
                    requires_grad=out_requires_grad,
                    grad_fn=_clamp_backward if out_requires_grad else None,
                    parents=(self,) if out_requires_grad else None,
                    device=self.device)