from mytorch import Tensor, Array
import dpnp as dp
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build flat gather indices for a sliding window
# ─────────────────────────────────────────────────────────────────────────────
 
def _pool_idx(H_pad, W_pad, K, H_out, W_out, S, D, device):
    """
    Returns idx of shape (H_out*W_out, K*K).
 
    idx[p, q] is the flat index into a (H_pad * W_pad) spatial map
    for output position p and kernel offset q.
    """
    # Kernel offsets within a single row of the padded map
    ki = dp.arange(K, device=device)                          # (K,)
    kj = dp.arange(K, device=device)                          # (K,)
    # Flat kernel offsets: row * row-stride + col * dilation
    k_off = (ki[:, None] * D * W_pad + kj[None, :] * D).reshape(-1)   # (K*K,)
 
    # Top-left corner of each output window
    oi = dp.arange(H_out, device=device)                      # (H_out,)
    oj = dp.arange(W_out, device=device)                      # (W_out,)
    o_start = (oi[:, None] * S * W_pad + oj[None, :] * S).reshape(-1) # (H_out*W_out,)
 
    # Combine: (H_out*W_out, K*K)
    idx = o_start[:, None] + k_off[None, :]
    return idx                                                 # (P, KK) where P=H_out*W_out
 
def maxpool2d_forward(input, kernel_size, stride=None, padding=0, dilation=1):
    """
    Vectorized 2D max-pooling forward pass.
 
    Args:
        input:       dp array  (N, C, H, W)
        kernel_size: int
        stride:      int  (default: kernel_size)
        padding:     int
        dilation:    int
 
    Returns:
        out:         dp array  (N, C, H_out, W_out)
        argmax_flat: dp array  (N, C, H_out*W_out)  – flat argmax per window
                     (saved for backward)
        idx:         dp array  (H_out*W_out, K*K)   – gather indices
                     (saved for backward)
    """
    N, C, H, W = input.shape
    K  = kernel_size
    S  = stride if stride is not None else K
    P, D = padding, dilation
 
    # 1. Pad with -inf so padded positions never win the max
    if P > 0:
        input_pad = dp.pad(
            input, ((0,0),(0,0),(P,P),(P,P)),
            mode='constant', constant_values=-dp.inf
        )
    else:
        input_pad = input
 
    H_pad, W_pad = H + 2*P, W + 2*P
    K_eff  = D * (K - 1) + 1
    H_out  = (H_pad - K_eff) // S + 1
    W_out  = (W_pad - K_eff) // S + 1
    P_out  = H_out * W_out                                     # num output positions
    KK     = K * K                                             # kernel elements
 
    # 2. Build gather indices once  →  (P_out, KK)
    idx = _pool_idx(H_pad, W_pad, K, H_out, W_out, S, D, input.device)
 
    # 3. Gather all window values
    #    input_pad : (N, C, H_pad*W_pad)
    #    windows   : (N, C, P_out, KK)
    input_flat = input_pad.reshape(N, C, -1)
    windows    = input_flat[:, :, idx]                         # (N, C, P_out, KK)
 
    # 4. Max and argmax over the kernel dimension
    out_flat      = windows.max(axis=-1)                       # (N, C, P_out)
    argmax_flat   = windows.argmax(axis=-1)                    # (N, C, P_out)
 
    out = out_flat.reshape(N, C, H_out, W_out)
 
    return out, argmax_flat, idx
 
def maxpool2d_backward(input_shape, argmax_flat, idx, grad,
                       kernel_size, stride=None, padding=0, dilation=1):
    """
    Vectorized 2D max-pooling backward pass.
 
    Routes each upstream gradient scalar to the single max element
    (the winner identified by argmax_flat) via a one-hot mask and scatter-add.
 
    Args:
        input_shape:  (N, C, H, W)  – shape of the original input
        argmax_flat:  dp array  (N, C, H_out*W_out)  – from forward
        idx:          dp array  (H_out*W_out, K*K)   – from forward
        grad:         dp array  (N, C, H_out, W_out) – upstream gradient
        kernel_size, stride, padding, dilation: same as forward
 
    Returns:
        di:  dp array  (N, C, H, W)
    """
    N, C, H, W = input_shape
    K = kernel_size
    S = stride if stride is not None else K
    P, D = padding, dilation
 
    H_pad, W_pad = H + 2*P, W + 2*P
    K_eff  = D * (K - 1) + 1
    H_out  = (H_pad - K_eff) // S + 1
    W_out  = (W_pad - K_eff) // S + 1
    P_out  = H_out * W_out
    KK     = K * K
 
    # 1. Flatten upstream gradient  →  (N, C, P_out)
    grad_flat = grad.reshape(N, C, -1)
 
    # 2. One-hot mask over kernel positions  →  (N, C, P_out, KK)
    #    mask[n, c, p, k] = 1 iff k == argmax_flat[n, c, p]
    mask = (dp.arange(KK, device=grad.device)[None, None, None, :]   # (1,1,1,KK)
            == argmax_flat[:, :, :, None])                            # (N,C,P_out,1)
    mask = mask.astype(grad.dtype)
 
    # 3. Weighted mask  →  (N, C, P_out, KK)
    weighted = mask * grad_flat[:, :, :, None]
 
    # 4. Scatter-add via matmul
    #    Build a sparse scatter matrix S of shape (P_out*KK, H_pad*W_pad):
    #      S[q, idx.reshape(-1)[q]] = 1  for each q in 0..P_out*KK-1
    #    Then di_pad = weighted.reshape(N,C, P_out*KK) @ S
    HW_pad   = H_pad * W_pad
    flat_idx = idx.reshape(-1)                                 # (P_out*KK,)
    rows     = dp.arange(P_out * KK, device=grad.device)       # (P_out*KK,)
 
    S = dp.zeros((P_out * KK, HW_pad), dtype=grad.dtype, device=grad.device)
    S[rows, flat_idx] = 1                                      # (P_out*KK, HW_pad)
 
    # (N, C, P_out*KK) @ (P_out*KK, HW_pad) → (N, C, HW_pad)
    flat_weights = weighted.reshape(N, C, -1)                  # (N, C, P_out*KK)
    di_pad = flat_weights @ S                                  # (N, C, HW_pad)
 
    # 5. Reshape and strip padding
    di_pad = di_pad.reshape(N, C, H_pad, W_pad)
    if P > 0:
        di = di_pad[:, :, P:P+H, P:P+W]
    else:
        di = di_pad
 
    return di

def maxpool2d(input, kernel_size, stride=None, padding=0, dilation=1, **args):
    input_arr = input.data._array
 
    out_arr, argmax_flat, idx = maxpool2d_forward(
        input_arr, kernel_size, stride, padding, dilation
    )
 
    def _maxpool2d_backward(grad):
        grad_arr = grad._array
        if input.requires_grad:
            di = maxpool2d_backward(
                input_arr.shape, argmax_flat, idx, grad_arr,
                kernel_size, stride, padding, dilation
            )
            input._add_grad(Array(di))
 
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
 
    return Tensor(
        out_arr,
        requires_grad=requires_grad,
        grad_fn=_maxpool2d_backward if requires_grad else None,
        parents=(input,) if requires_grad else (),
        device=input.device
    )