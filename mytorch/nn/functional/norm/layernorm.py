import numpy as np
from mytorch import Tensor

def reshape_for_layernorm(x):
    reshaped = False
    *dims, embed_dim = x.shape

    ### If we have more than 1 dim, we have to flatten ###
    if len(dims) > 1:
        reshaped = True

    x = np.reshape(x, [-1, embed_dim])

    return x, dims, reshaped

def layernorm(input, weight, bias, eps=1e-5, *args):
    input_arr, dims, reshaped_flag = reshape_for_layernorm(input.data)
    embed_dim = input_arr.shape[-1]

    weight_arr = weight.data if weight is not None else None
    beta_arr = bias.data if bias is not None else None

    ### Compute Mean and Var Along Last Dimension ###
    mean = np.mean(input_arr, axis=-1, keepdims=True)
    var = np.var(input_arr, axis=-1, keepdims=True)
    inv_std = np.reciprocal(np.sqrt(var + eps))

    ### Store copy of x_hat for the input backward ###
    x_hat = (input_arr - mean) * inv_std
    
    ### Scale if weight is used ###
    if weight_arr is not None:
        output = x_hat * weight_arr.reshape(1,-1)
    else:
        output = x_hat

    ### Add bias if available ###
    if beta_arr is not None:
        output += beta_arr.reshape(1,-1)

    ### Reshape Back if Needed ###
    if reshaped_flag:
        output = output.reshape(*dims, embed_dim)

    def _layernorm_backward(grad):
        
        ### Reshape Grad Output as its currently (*, I) ###
        if reshaped_flag:
            grad = np.reshape(grad, [-1, embed_dim])

        if weight is not None:
            if weight.requires_grad:
                grad_weight = np.sum(grad * x_hat, axis=0)
                weight._add_grad(grad_weight)
        
        if bias is not None:
            if bias.requires_grad:
                grad_bias = np.sum(grad, axis=0)
                bias._add_grad(grad_bias)

        if input.requires_grad:
            # y = x_hat * gamma + beta
            # where x_hat = (x - mu) / (var + eps)
            # dL/dx = dL/dy * dy/dx_hat * dx_hat / dx
            # = inv_std * (grad_output * gamma - mean(grad_output*gamma) - x_hat*mean(grad_output * weight_arr * x_hat))
            # sum up grads over the batch dim
            dx_hat = grad * weight_arr if weight_arr is not None else grad
            mean_dx_hat = np.mean(dx_hat, axis=-1, keepdims=True)
            mean_mean_dx_hat_x_hat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
            grad_input = inv_std * (dx_hat - mean_dx_hat - x_hat * mean_mean_dx_hat_x_hat) 

            ### Put Back into Original Shape ###
            if reshaped_flag:
                grad_input = np.reshape(grad_input, [*dims, embed_dim])
            input._add_grad(grad_input)

    requires_grad = (input.requires_grad or \
                    (weight is not None and weight.requires_grad) or \
                    (bias is not None and bias.requires_grad)) and \
                    Tensor.build_graph_enabled()
    return Tensor(
        output, 
        requires_grad=requires_grad,
        grad_fn=_layernorm_backward if requires_grad else None, 
        parents=(input,weight,bias) if requires_grad else None
    )