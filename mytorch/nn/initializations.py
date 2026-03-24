import math
import numpy as np
from ..factory import rand_like
import mytorch

def uniform_(tensor, low=0.0, high=1.0):
    arr = rand_like(tensor).data * (high - low) + low
    tensor.data = arr

def kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan = _calculate_fan(tensor.shape, mode)
    gain = calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0) * gain / math.sqrt(fan)
    uniform_(tensor, -bound, bound)

def _calculate_fan_in_out(shape):
    if len(shape) == 2:  # linear layer weight
        fan_in, fan_out = shape[1], shape[0]
    elif len(shape) in {3, 4, 5}:  # conv weights
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        fan_in = fan_out = 1
    return fan_in, fan_out

def _calculate_fan(shape, mode="fan_in"):
    fan_in, fan_out = _calculate_fan_in_out(shape)
    return fan_in if mode == "fan_in" else fan_out

def calculate_gain(nonlinearity, param=None):
    if nonlinearity == "linear" or nonlinearity == "conv1d":
        return 1.0
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1 + param**2)) if param is not None else math.sqrt(2.0)
    return 1.0

def zeros_(tensor):
    tensor.data = mytorch.zeros_like(tensor).data

def ones_(tensor):
    tensor.data = mytorch.ones_like(tensor).data

def normal_(tensor, mean=0.0, std=1.0):
    arr = mytorch.randn_like(tensor).data * std + mean
    tensor.data = arr

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    tensor.data = trunc_normal(tensor.shape, mean, std, a, b)

def trunc_normal(shape, mean=0.0, std=1.0, a=-2.0, b=2.0, dtype=np.float32):
    """
    Sample from a truncated normal distribution using NumPy only.
    Matches PyTorch's trunc_normal_ behavior.

    Args:
        shape:  Output array shape.
        mean:   Mean of the underlying normal distribution.
        std:    Std of the underlying normal distribution.
        a:      Lower truncation bound (in original, non-standardized space).
        b:      Upper truncation bound (in original, non-standardized space).
        dtype:  Output dtype (default float32 to match PyTorch).

    Returns:
        NumPy array of the given shape sampled from the truncated normal.
    """
    # Standardize bounds to the standard normal
    alpha = (a - mean) / std
    beta  = (b - mean) / std

    # CDF of the standard normal at the bounds
    # We use the relationship: Phi(x) = 0.5 * erfc(-x / sqrt(2))
    def phi(x):
        erf = np.vectorize(math.erf)
        return 0.5 * (1.0 - erf(-x / np.sqrt(2)))

    p_low  = phi(alpha)
    p_high = phi(beta)

    # Draw uniform samples in [p_low, p_high], then invert via the
    # percent-point function (inverse CDF):  x = sqrt(2) * erfinv(2p - 1)
    u = np.random.uniform(p_low, p_high, size=shape)
    samples = np.sqrt(2) * _erfinv_approx(2 * u - 1)

    # Re-scale to the requested mean / std and clip for numerical safety
    samples = np.clip(samples * std + mean, a, b)
    return samples.astype(dtype)

def _erfinv_approx(x):
    """
    Pure-NumPy inverse error function.
    Winitzki (2008) rational approximation — max absolute error ≈ 3e-4.
    """
    a = 0.147
    ln1mx2 = np.log1p(-x * x)           # log(1 - x²)
    t1 = 2 / (np.pi * a) + ln1mx2 / 2
    t2 = ln1mx2 / a
    return np.sign(x) * np.sqrt(np.sqrt(t1 * t1 - t2) - t1)