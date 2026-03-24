import dpnp.random as random
import mytorch

def dropout(input, dropout_p, training=True):
    if not training or dropout_p == 0.0:
        return input
    
    mask = (mytorch.rand_like(input, device=input.device) >= dropout_p).astype(input.dtype)
    ratio = 1 / (1 - dropout_p)
    mask *= ratio

    return input * mask