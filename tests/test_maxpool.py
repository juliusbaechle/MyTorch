import dpnp as dp
import mytorch
from mytorch import Array
import mytorch.nn.functional as F

def naive_maxpool2d(input, kernel_size, stride=None, padding=0, dilation=1):
    """
    Naive 2D max-pooling (forward).
 
    Args:
        input:       dp array of shape (N, C, H, W)
        kernel_size: int – pooling window size (square)
        stride:      int – defaults to kernel_size when None
        padding:     int – zero-padding added to both spatial dims
        dilation:    int – spacing between kernel elements
 
    Returns:
        out:         dp array of shape (N, C, H_out, W_out)
    """
    N, C, H, W = input.shape
    K = kernel_size
    S = stride if stride is not None else K
    P, D = padding, dilation
 
    # Work on padded input so index arithmetic stays identical to forward
    input = dp.pad(input, ((0,0), (0,0), (P,P), (P,P)),
                    mode='constant', constant_values=-dp.inf)
    H = H + 2 * P
    W = W + 2 * P
 
    K_eff  = D * (K - 1) + 1
    H_out  = (H - K_eff) // S + 1
    W_out  = (W - K_eff) // S + 1
 
    out = dp.zeros((N, C, H_out, W_out), dtype=input.dtype, device=input.device)
 
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    patch = input[n, c,
                                  i*S : i*S + K_eff : D,
                                  j*S : j*S + K_eff : D]   # (K, K)
                    out[n, c, i, j] = patch.max()
 
    return out
 
 
def naive_maxpool2d_backward(input, grad, kernel_size, stride=None, padding=0, dilation=1):
    """
    Naive 2D max-pooling (backward).
 
    Gradients are routed to the single max element inside each window
    (ties broken by the first occurrence, matching dpnp/numpy argmax).
 
    Args:
        input:       dp array of shape (N, C, H, W) – original (pre-pad) input
        grad:        dp array of shape (N, C, H_out, W_out) – upstream gradient
        kernel_size: int
        stride:      int – defaults to kernel_size when None
        padding:     int
        dilation:    int
 
    Returns:
        di:          dp array of shape (N, C, H, W) – gradient w.r.t. input
    """
    N, C, H, W = input.shape
    K = kernel_size
    S = stride if stride is not None else K
    P, D = padding, dilation
    _, _, H_out, W_out = grad.shape
    K_eff = D * (K - 1) + 1
 
    # Work on padded input so index arithmetic stays identical to forward
    input_pad = dp.pad(input, ((0,0), (0,0), (P,P), (P,P)), mode='constant', constant_values=-dp.inf)
 
    H_pad = H + 2 * P
    W_pad = W + 2 * P
    di_pad = dp.zeros((N, C, H_pad, W_pad), dtype=input.dtype, device=input.device)
 
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    patch = input_pad[n, c,
                                      i*S : i*S + K_eff : D,
                                      j*S : j*S + K_eff : D]   # (K, K)
 
                    # Flat index of the maximum element inside the patch
                    flat_idx = int(patch.argmax())
                    ki, kj   = flat_idx // K, flat_idx % K
 
                    # Map back to padded-input coordinates
                    hi = i*S + ki*D
                    wj = j*S + kj*D
                    di_pad[n, c, hi, wj] += grad[n, c, i, j]
 
    # Strip padding to recover gradient in original input space
    if P > 0:
        di = di_pad[:, :, P:P+H, P:P+W]
    else:
        di = di_pad
 
    return di
 
 
def test_maxpool2d():
    input  = mytorch.rand([2, 3, 28, 28], True)
 
    out1 = F.maxpool2d(input, 3, 2, 1, 1)
    out2 = naive_maxpool2d(input.data._array, 3, 2, 1, 1)
    assert dp.allclose(out1.data._array, out2).item(), "Forward mismatch"
 
    grad = Array.rand(out1.shape)
    out1.backward(grad)
 
    di = naive_maxpool2d_backward(input.data._array, grad._array, 3, 2, 1, 1)
    assert dp.allclose(input.grad._array, di).item(), "Backward mismatch"
 
    print("All max_pool2d tests passed.")