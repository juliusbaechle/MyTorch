from mytorch import Tensor, Array
import dpnp as dp

def conv2d_forward(input, weight, bias=None, stride=1, padding=0, dilation=1):
    B, C_in, H, W = input.shape
    C_out, _, K, _ = weight.shape
    S, P, D = stride, padding, dilation

    # 0. Pad array
    if P > 0:
        input = dp.pad(input, ((0,0), (0,0), (P,P), (P,P)), mode='constant')
        H = H + 2 * P
        W = W + 2 * P

    K_eff = D * (K - 1) + 1
    H_out = (H - K_eff)//S + 1
    W_out = (W - K_eff)//S + 1

    # 1. Build im2col index arrays (done once on CPU, shape-only work)
    kernel_offsets = dp.arange(K, device=input.device)[:, None] * D * W + dp.arange(K, device=input.device)[None, :] * D
    row_starts = dp.arange(H_out, device=input.device)[:, None] * S * W + dp.arange(W_out, device=input.device)[None, :] * S
    idx = (row_starts.reshape(-1, 1) + kernel_offsets.reshape(1, -1))

    # 2. im2col  →  col : (B, H_out*W_out, C_in*kH*kW)
    input_flat = input.reshape(B, C_in, -1)  # (B, C_in, H*W)
    col = input_flat[:, :, idx]           # (B, C_in, H_out*W_out, kH*kW)
    col = col.transpose(0, 2, 1, 3)       # (B, H_out*W_out, C_in, kH*kW)
    col = col.reshape(B, H_out * W_out, C_in * K * K) # (B, H_out*W_out, C_in*kH*kW)

    # 3. Flatten kernel  →  (C_out, C_in*kH*kW)
    w_flat = weight.reshape(C_out, -1)

    # 4. Batched matmul: (B, H_out*W_out, C_in*kH*kW) @ (C_in*kH*kW, C_out) → (B, H_out*W_out, C_out)
    out = col @ w_flat.T
    if bias is not None:
        out += bias

    # 5. Reshape back to BCHW
    out = out.transpose(0, 2, 1).reshape(B, C_out, H_out, W_out)
    return out

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, **args):
    input_arr = input.data._array
    weight_arr = weight.data._array
    if bias is not None:
        bias_arr = bias.data._array

    out = conv2d_forward(input_arr, weight_arr, bias_arr, stride, padding, dilation)

    def _conv2d_backward(grad):
        grad = grad._array

        B, C_in, H, W = input_arr.shape
        C_out, _, K, _ = weight_arr.shape
        _, _, H_out, W_out = grad.shape
        S, P, D = stride, padding, dilation

        # 4. db
        if bias is not None and bias.requires_grad:
            db = grad.sum(axis=(0, 2, 3))
            bias._add_grad(Array(db))     # (C_out,)

        # 5. dw
        if weight.requires_grad:
            # 1. Pad input
            input_pad = dp.pad(input_arr, ((0,0),(0,0),(P,P), (P,P)))
            W_pad = W + 2 * P

            # 1. Build im2col index arrays (done once on CPU, shape-only work)
            kernel_offsets = dp.arange(K, device=input.device)[:, None] * D * W_pad + dp.arange(K, device=input.device)[None, :] * D
            row_starts = dp.arange(H_out, device=input.device)[:, None] * S * W_pad + dp.arange(W_out, device=input.device)[None, :] * S
            idx = (row_starts.reshape(-1, 1) + kernel_offsets.reshape(1, -1))

            # 2. im2col  →  col : (B, H_out*W_out, C_in*kH*kW)
            input_flat = input_pad.reshape(B, C_in, -1)  # (B, C_in, H*W)
            col = input_flat[:, :, idx]           # (B, C_in, H_out*W_out, kH*kW)
            col = col.transpose(0, 2, 1, 3)       # (B, H_out*W_out, C_in, kH*kW)
            col = col.reshape(B, H_out * W_out, C_in * K * K) # (B, H_out*W_out, C_in*kH*kW)

            # 3. Flatten dout for matmul: (N, H_out*W_out, C_out)
            grad_flat = grad.transpose(0, 2, 3, 1).reshape(B, H_out * W_out, C_out)

            dw_flat = dp.einsum('npc,npk->ck', grad_flat, col)
            dw = dw_flat.reshape(C_out, C_in, K, K)
            weight._add_grad(Array(dw))

        if input.requires_grad:
            # Flip the kernel spatially: (C_out, C_in, kH, kW) -> (C_in, C_out, kH, kW)
            w_flipped = weight_arr[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
            
            # The transposed convolution padding and stride:
            #   - dilation stays the same
            #   - new padding = kH_eff - 1 - pH (to undo the original padding)
            #   - stride > 1 requires inserting (s-1) zeros between dout elements
            #     (i.e. "dilating" dout) — handle by upsampling dout first
            P_t = D * (K - 1) - P

            # If stride > 1, upsample dout by inserting (s-1) zeros between elements
            if S > 1:
                grad_up = dp.zeros(
                    (B, C_out, S * (H_out - 1) + 1, S * (W_out - 1) + 1),
                    dtype=grad.dtype,
                    device=input.device
                )
                grad_up[:, :, ::S, ::S] = grad
            else:
                grad_up = grad

            # Pad grad_up so that the output of the transposed conv is exactly (H, W)
            K_eff = D * (K - 1) + 1
            P_t = K_eff - 1 - P

            # The "missing" pixels after strided upsampling
            pH_left  = P_t
            pH_right = H - (grad_up.shape[2] + 2 * P_t - K_eff + 1) + P_t
            pW_left  = P_t
            pW_right = W - (grad_up.shape[3] + 2 * P_t - K_eff + 1) + P_t

            grad_up = dp.pad(grad_up, ((0,0),(0,0),(pH_left, pH_right),(pW_left, pW_right)))

            di = conv2d_forward(grad_up, w_flipped, None, stride=1, padding=0, dilation=D)
            input._add_grad(Array(di))

    requires_grad = (input.requires_grad or weight.requires_grad or \
                        (bias is not None and bias.requires_grad)) \
                        and Tensor.build_graph_enabled()

    return Tensor(
        out,
        requires_grad=requires_grad,
        grad_fn=_conv2d_backward if requires_grad else None,
        parents=(input, weight, bias) if requires_grad else (),
        device=input.device
    )