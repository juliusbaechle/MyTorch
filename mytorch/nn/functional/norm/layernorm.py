def reshape_for_layernorm(x):
    reshaped = False
    *dims, embed_dim = x.shape

    ### If we have more than 1 dim, we have to flatten ###
    if len(dims) > 1:
        reshaped = True

    x = x.reshape(-1, embed_dim)

    return x, dims, reshaped

def layernorm(input, weight, bias, eps=1e-5, *args):

    input, dims, reshaped_flag = reshape_for_layernorm(input)
    embed_dim = input.shape[-1]

    var_x = (input.var(dim=-1, keepdims=True) + eps)
    norm_x = (input - input.mean(dim=-1, keepdims=True)) / var_x**0.5
    
    if weight:
        scale_shifted_x = norm_x * weight.reshape(1,-1) 
    else:
        scale_shifted_x = norm_x
    
    if bias:
        scale_shifted_x = scale_shifted_x + bias.reshape(1,-1)

    if reshaped_flag:
        scale_shifted_x = scale_shifted_x.reshape(*dims, embed_dim)

    return scale_shifted_x