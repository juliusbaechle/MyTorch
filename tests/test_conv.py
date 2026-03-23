import dpnp as dp
import numpy as np
import mytorch
from mytorch import Array
import mytorch.nn.functional as F

def naive_conv2d(input, weight, bias, stride=1, padding=0, dilation=1):
    N, C_in, H, W = input.shape
    C_out, _, K, _ = weight.shape
    S, P, D = stride, padding, dilation

    if P > 0:
        input = dp.pad(input, ((0,0), (0,0), (P,P), (P,P)), mode='constant')
        H = H + 2 * P
        W = W + 2 * P

    K_eff = D * (K - 1) + 1
    H_out = (H - K_eff)//S + 1
    W_out = (W - K_eff)//S + 1

    out = dp.zeros((N, C_out, H_out, W_out), dtype=input.dtype, device=input.device)
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, co, i, j] = bias[co] + (
                        input[n, :,
                              i*S : i*S + K_eff : D,
                              j*S : j*S + K_eff : D]
                              * weight[co]).sum()
    return out

def naive_conv2d_backward(input, weight, grad, stride=1, padding=0, dilation=1):
    S, P, D = stride, padding, dilation

    N, C_in, H, W = input.shape
    C_out, _, K, K = weight.shape
    _, _, H_out, W_out = grad.shape
    K_eff = D * (K - 1) + 1

    if P > 0:
        input = dp.pad(input, ((0,0), (0,0), (P,P), (P,P)), mode='constant')

    di = dp.zeros_like(input)
    dw = dp.zeros_like(weight)

    # db: sum over N, H_out, W_out
    db = grad.sum(axis=(0, 2, 3))              # (C_out,)

    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    g = grad[n, co, i, j]      # scalar

                    # x patch: (C_in, kH, kW)
                    patch = input[n, :,
                                  i*S : i*S + K_eff : D,
                                  j*S : j*S + K_eff : D]

                    dw[co] += g * patch        # (C_in, kH, kW)

                    di[n, :,
                           i*S : i*S + K_eff : D,
                           j*S : j*S + K_eff : D] += g * weight[co]

    di = di[:, :, P:P+H, P:P+W]
    return di, dw, db

def test_conv2d():
    input = mytorch.rand([2, 3, 28, 28], True)
    weight = mytorch.rand([4, 3, 3, 3], True)
    bias = mytorch.rand([4], True)

    out1 = F.conv2d(input, weight, bias, 2, 1, 2)
    out2 = naive_conv2d(input.data._array, weight.data._array, bias.data._array, 2, 1, 2)
    assert(dp.allclose(out1.data._array, out2).item())

    grad = Array.rand(out1.shape)
    out1.backward(grad)
    di, dw, db = naive_conv2d_backward(input.data._array, weight.data._array, grad._array, 2, 1, 2)
    assert(dp.allclose(input.grad._array, di).item())
    assert(dp.allclose(weight.grad._array, dw).item())
    assert(dp.allclose(bias.grad._array, db).item())