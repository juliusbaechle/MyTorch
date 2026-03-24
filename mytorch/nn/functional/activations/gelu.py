import numpy as np
from mytorch import Tensor

def gelu(input):
    
    """
    gelu as described in https://arxiv.org/pdf/2305.12073

    Forward method is Equation 24
    Backward method is Equation 42-43
    """

    data = input.data

    # Constants
    sqrt_2_over_pi = 0.7978845608 # xp.sqrt(2 / xp.pi).astype(x.data.dtype)
    coeff = 0.044715

    #inner = sqrt_2_over_pi * (x + coeff * x^3)
    x_squared = np.power(data, 2)
    x_cubed = x_squared * data

    inner = sqrt_2_over_pi * (data + coeff * x_cubed)

    ### Tanh out = tanh(inner) ###
    tanh_out = np.tanh(inner)
    out_data = 0.5 * data * (1.0 + tanh_out)

    # Backward
    def _gelu_backward(grad):
        if input.requires_grad:
            inner_grad = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_squared)
            # derivative of GELU approximation (sech^2(x) = 1 - tanh^2(x))
            sech2 = 1 - np.power(tanh_out, 2)  # derivative of tanh
            grad_x = 0.5 * (1.0 + tanh_out + data * sech2 * inner_grad) * grad
            input._add_grad(grad_x)

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    return Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_gelu_backward if requires_grad else None,
        parents=(input) if requires_grad else (),
        device=input.device
    )