import dpnp.random as random

def dropout(input, dropout_p, training=True):
    if not training or dropout_p == 0.0:
        return input
    
    mask = (random.random_sample(input.data.shape, device=input.device) >= dropout_p).astype(input.dtype, copy=False)
    ratio = 1 / (1 - dropout_p)
    mask *= ratio

    return input * mask